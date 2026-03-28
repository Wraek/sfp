"""Tier 1: Episodic Memory — structured buffer with hot/cold storage and admission gates.

Implements the episodic memory tier from sfp-hierarchical-memory-implementation.md.
Episodes are stored with full provenance and integrity hashing. Admission is gated
by surprise threshold, cosine deduplication, and cryptographic integrity. Eviction
is stratified by attractor basin to maintain coverage across the concept manifold.
"""

from __future__ import annotations

import random
import time
from collections import defaultdict

import torch

from sfp.config import Tier1Config
from sfp.exceptions import MemoryTierError
from sfp.memory.integrity import verify_episode_integrity
from sfp.types import Episode
from sfp.utils.logging import get_logger
from sfp.utils.math import cosine_similarity_matrix

logger = get_logger("memory.episodic")


class EpisodicMemory:
    """Tier 1: Structured episodic memory buffer with hot/cold storage.

    Hot buffer (GPU): Fast access for recent, high-priority episodes.
    Cold buffer (CPU): Overflow storage for older episodes.

    Admission is gated by three mechanisms:
      1. Surprise threshold — only store genuinely surprising episodes.
      2. Cosine deduplication — reject near-duplicates of recent episodes.
      3. Integrity hashing — every episode carries a SHA-256 integrity hash.

    Eviction is stratified by attractor basin to maintain coverage across
    all known concept regions.

    Args:
        config: Tier1Config with capacity, thresholds, etc.
        d_model: Dimensionality of embeddings.
    """

    def __init__(self, config: Tier1Config | None = None, d_model: int = 512) -> None:
        self._config = config or Tier1Config()
        self._d_model = d_model
        self._hot: list[Episode] = []
        self._cold: list[Episode] = []
        self._next_id: int = 0
        self._last_episode_id: int | None = None

    def maybe_store(self, episode: Episode) -> bool:
        """Attempt to store an episode, subject to admission gates.

        Args:
            episode: The episode to store.

        Returns:
            True if the episode was admitted, False if rejected.
        """
        cfg = self._config

        # Gate 1: Surprise threshold
        if episode.surprise_at_storage < cfg.surprise_threshold:
            return False

        # Gate 2: Cosine deduplication against recent hot episodes
        if self._hot:
            recent = self._hot[-min(50, len(self._hot)) :]
            recent_embeddings = torch.stack([e.input_embedding for e in recent])
            query = episode.input_embedding.unsqueeze(0)
            # Move to same device for comparison
            if recent_embeddings.device != query.device:
                recent_embeddings = recent_embeddings.to(query.device)
            sims = cosine_similarity_matrix(query, recent_embeddings)
            max_sim = sims.max().item()
            if max_sim > cfg.dedup_threshold:
                return False

        # Gate 3: Integrity hash must be valid
        if not verify_episode_integrity(
            episode.input_embedding,
            episode.logit_snapshot,
            episode.weight_hash_at_storage,
            episode.integrity_hash,
        ):
            logger.warning("Episode %d failed integrity check — rejected", episode.id)
            return False

        # Admitted — add to hot buffer
        self._hot.append(episode)
        self._last_episode_id = episode.id

        # Evict to cold if hot buffer meets or exceeds capacity
        if len(self._hot) >= self._config.hot_capacity:
            self._evict_to_cold()

        return True

    def force_store(self, episode: Episode) -> bool:
        """Force-store an episode, bypassing the surprise threshold gate.

        Used for episode boundary storage when the world model's continue_head
        detects a sequence transition.  Still applies dedup and integrity checks.

        Args:
            episode: The episode to store.

        Returns:
            True if stored, False if rejected by dedup or integrity.
        """
        # Gate 1 (surprise): SKIPPED for forced storage

        # Gate 2: Cosine deduplication against recent hot episodes
        if self._hot:
            recent = self._hot[-min(50, len(self._hot)) :]
            recent_embeddings = torch.stack([e.input_embedding for e in recent])
            query = episode.input_embedding.unsqueeze(0)
            if recent_embeddings.device != query.device:
                recent_embeddings = recent_embeddings.to(query.device)
            sims = cosine_similarity_matrix(query, recent_embeddings)
            max_sim = sims.max().item()
            if max_sim > self._config.dedup_threshold:
                return False

        # Gate 3: Integrity hash must be valid
        if not verify_episode_integrity(
            episode.input_embedding,
            episode.logit_snapshot,
            episode.weight_hash_at_storage,
            episode.integrity_hash,
        ):
            logger.warning(
                "Episode %d failed integrity check — rejected (forced)",
                episode.id,
            )
            return False

        # Admitted
        self._hot.append(episode)
        self._last_episode_id = episode.id

        if len(self._hot) >= self._config.hot_capacity:
            self._evict_to_cold()

        logger.debug("Force-stored episode %d (episode boundary)", episode.id)
        return True

    def sample_for_replay(self, batch_size: int = 32) -> list[Episode]:
        """Sample a stratified batch of episodes for consolidation replay.

        Sampling is stratified by attractor basin to ensure diverse concept coverage.
        Hot episodes are preferred over cold ones.

        Args:
            batch_size: Number of episodes to sample.

        Returns:
            List of sampled episodes (may be fewer than batch_size if buffer is small).
        """
        all_episodes = self._hot + self._cold
        if not all_episodes:
            return []

        if len(all_episodes) <= batch_size:
            return list(all_episodes)

        # Group by basin
        basins: dict[int, list[Episode]] = defaultdict(list)
        for ep in all_episodes:
            basins[ep.attractor_basin_id].append(ep)

        # Stratified sampling: round-robin across basins with surprise weighting
        sampled: list[Episode] = []
        basin_ids = list(basins.keys())
        random.shuffle(basin_ids)

        per_basin = max(1, batch_size // len(basin_ids))
        for bid in basin_ids:
            candidates = basins[bid]
            n = min(per_basin, len(candidates))
            # Weight by surprise_at_storage: higher surprise → more likely to be replayed
            weights = [max(ep.surprise_at_storage, 0.01) for ep in candidates]
            picks = random.choices(candidates, weights=weights, k=n)
            # Deduplicate (choices can repeat)
            seen = {id(e) for e in sampled}
            for ep in picks:
                if id(ep) not in seen:
                    sampled.append(ep)
                    seen.add(id(ep))
            if len(sampled) >= batch_size:
                break

        # If we still need more, sample from the remainder weighted by surprise
        if len(sampled) < batch_size:
            sampled_set = {id(e) for e in sampled}
            remaining = [ep for ep in all_episodes if id(ep) not in sampled_set]
            if remaining:
                extra = min(batch_size - len(sampled), len(remaining))
                weights = [max(ep.surprise_at_storage, 0.01) for ep in remaining]
                picks = random.choices(remaining, weights=weights, k=extra)
                seen = {id(e) for e in sampled}
                for ep in picks:
                    if id(ep) not in seen:
                        sampled.append(ep)
                        seen.add(id(ep))

        return sampled[:batch_size]

    def validate_integrity(self) -> list[Episode]:
        """Validate integrity hashes of all stored episodes.

        Returns:
            List of episodes that failed integrity checks.
        """
        failed: list[Episode] = []
        for ep in self._hot + self._cold:
            if not verify_episode_integrity(
                ep.input_embedding,
                ep.logit_snapshot,
                ep.weight_hash_at_storage,
                ep.integrity_hash,
            ):
                ep.flagged = True
                failed.append(ep)

        if failed:
            logger.warning("%d episodes failed integrity validation", len(failed))

        return failed

    def _evict_to_cold(self) -> None:
        """Evict episodes from hot to cold buffer using stratified eviction.

        Preferentially evicts from over-represented basins, scoring by
        age * (1 + consolidation_count) to keep unconsolidated, diverse episodes hot.
        """
        cfg = self._config
        n_to_evict = min(cfg.eviction_batch_size, len(self._hot) - cfg.hot_capacity)
        if n_to_evict <= 0:
            return

        # Score episodes for eviction (higher score = more evictable)
        now = time.monotonic()
        basin_counts: dict[int, int] = defaultdict(int)
        for ep in self._hot:
            basin_counts[ep.attractor_basin_id] += 1

        scored: list[tuple[float, int, Episode]] = []
        for i, ep in enumerate(self._hot):
            age = now - ep.timestamp
            consolidation_factor = 1.0 + ep.consolidation_count
            # Over-represented basins get higher eviction scores
            basin_factor = basin_counts[ep.attractor_basin_id] / max(cfg.min_per_basin, 1)
            score = age * consolidation_factor * basin_factor
            scored.append((score, i, ep))

        # Sort by score descending (most evictable first)
        scored.sort(key=lambda t: t[0], reverse=True)

        evict_indices: set[int] = set()
        for score, idx, ep in scored:
            if len(evict_indices) >= n_to_evict:
                break
            # Check basin minimum constraint
            remaining_in_basin = basin_counts[ep.attractor_basin_id] - sum(
                1 for j in evict_indices if self._hot[j].attractor_basin_id == ep.attractor_basin_id
            )
            if remaining_in_basin > cfg.min_per_basin:
                evict_indices.add(idx)

                # Move tensors to CPU for cold storage
                cold_ep = Episode(
                    id=ep.id,
                    timestamp=ep.timestamp,
                    modality=ep.modality,
                    provenance_hash=ep.provenance_hash,
                    input_embedding=ep.input_embedding.cpu(),
                    working_memory_state=ep.working_memory_state.cpu(),
                    logit_snapshot=ep.logit_snapshot.cpu(),
                    surprise_at_storage=ep.surprise_at_storage,
                    attractor_basin_id=ep.attractor_basin_id,
                    attractor_distance=ep.attractor_distance,
                    preceding_episode_id=ep.preceding_episode_id,
                    following_episode_id=ep.following_episode_id,
                    integrity_hash=ep.integrity_hash,
                    weight_hash_at_storage=ep.weight_hash_at_storage,
                    consolidation_count=ep.consolidation_count,
                    last_consolidated=ep.last_consolidated,
                    flagged=ep.flagged,
                )
                self._cold.append(cold_ep)

            if len(evict_indices) >= n_to_evict:
                break

        # Remove evicted episodes from hot buffer (reverse order to preserve indices)
        for idx in sorted(evict_indices, reverse=True):
            self._hot.pop(idx)

        # Trim cold buffer if over capacity
        if len(self._cold) >= self._config.cold_capacity:
            # Remove oldest cold episodes
            excess = len(self._cold) - self._config.cold_capacity
            self._cold = self._cold[excess:]

        logger.debug(
            "Evicted %d episodes to cold storage (hot=%d, cold=%d)",
            len(evict_indices),
            len(self._hot),
            len(self._cold),
        )

    def promote_to_hot(self, episode_ids: list[int]) -> int:
        """Move specific episodes from cold back to hot buffer.

        Args:
            episode_ids: IDs of episodes to promote.

        Returns:
            Number of episodes actually promoted.
        """
        id_set = set(episode_ids)
        promoted = 0
        remaining_cold: list[Episode] = []

        for ep in self._cold:
            if ep.id in id_set and len(self._hot) < self._config.hot_capacity:
                self._hot.append(ep)
                promoted += 1
            else:
                remaining_cold.append(ep)

        self._cold = remaining_cold
        return promoted

    def allocate_id(self) -> int:
        """Allocate the next episode ID."""
        eid = self._next_id
        self._next_id += 1
        return eid

    @property
    def hot_count(self) -> int:
        """Number of episodes in the hot (GPU) buffer."""
        return len(self._hot)

    @property
    def cold_count(self) -> int:
        """Number of episodes in the cold (CPU) buffer."""
        return len(self._cold)

    @property
    def total_count(self) -> int:
        """Total number of stored episodes."""
        return len(self._hot) + len(self._cold)

    @property
    def last_episode_id(self) -> int | None:
        """ID of the most recently stored episode."""
        return self._last_episode_id

    def retrieve_by_location(
        self,
        position: tuple[float, float, float],
        *,
        radius: float = 50.0,
        max_results: int = 10,
        embedding: torch.Tensor | None = None,
        spatial_weight: float = 0.7,
    ) -> list[tuple[Episode, float]]:
        """Retrieve episodes near a spatial position.

        "I was here before" — returns episodes stored at nearby positions,
        optionally blended with embedding similarity.

        Args:
            position: (x, y, z) query position.
            radius: Maximum Euclidean distance to consider.
            max_results: Maximum number of episodes to return.
            embedding: Optional (d_model,) query embedding for semantic
                blending.  If provided, the final score is a weighted
                average of spatial proximity and cosine similarity.
            spatial_weight: Weight for spatial proximity score vs. semantic
                similarity (default 0.7 = mostly spatial).

        Returns:
            List of (episode, score) tuples sorted by descending score.
            Score is in [0, 1].
        """
        px, py, pz = position
        candidates: list[tuple[Episode, float]] = []

        for ep in self._hot + self._cold:
            if ep.spatial_position is None:
                continue

            ex, ey, ez = ep.spatial_position
            dist = ((px - ex) ** 2 + (py - ey) ** 2 + (pz - ez) ** 2) ** 0.5

            if dist > radius:
                continue

            # Spatial proximity score: 1.0 at dist=0, 0.0 at dist=radius
            spatial_score = 1.0 - (dist / radius)

            if embedding is not None:
                # Blend spatial + semantic similarity
                ep_emb = ep.input_embedding
                if ep_emb.device != embedding.device:
                    ep_emb = ep_emb.to(embedding.device)
                cos_sim = torch.nn.functional.cosine_similarity(
                    embedding.unsqueeze(0), ep_emb.unsqueeze(0),
                ).item()
                semantic_score = max(0.0, (cos_sim + 1.0) / 2.0)  # [0, 1]
                score = (
                    spatial_weight * spatial_score
                    + (1.0 - spatial_weight) * semantic_score
                )
            else:
                score = spatial_score

            candidates.append((ep, score))

        # Sort by score descending
        candidates.sort(key=lambda t: t[1], reverse=True)
        return candidates[:max_results]

    @property
    def basin_distribution(self) -> dict[int, int]:
        """Distribution of episodes across attractor basins (hot + cold)."""
        counts: dict[int, int] = defaultdict(int)
        for ep in self._hot + self._cold:
            counts[ep.attractor_basin_id] += 1
        return dict(counts)
