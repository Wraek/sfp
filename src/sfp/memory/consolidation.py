"""Consolidation Engine — moves knowledge between memory tiers on schedule.

Implements the three consolidation cycles from sfp-hierarchical-memory-implementation.md:
  - Mini-consolidation (Tier 0 -> Tier 1): every ~100 steps, ~2ms
  - Standard consolidation (Tier 1 -> Tier 2): every ~1000 steps, ~50-200ms
  - Deep consolidation (Tier 2 -> Tier 3): every ~10000 steps, ~1-5s
"""

from __future__ import annotations

import random
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from sfp.affect.valence import ValenceSystem

from sfp.config import ConsolidationConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor
from sfp.memory.core import CoreMemory
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.integrity import compute_episode_hash, compute_weight_hash
from sfp.types import ConsolidationMode, Episode
from sfp.utils.logging import get_logger
from sfp.utils.math import cosine_similarity_matrix

logger = get_logger("memory.consolidation")


class ConsolidationEngine:
    """Manages knowledge transfer between the four memory tiers.

    Called periodically by the orchestrator (HierarchicalMemoryProcessor).
    Each consolidation mode runs independently on its own schedule.

    Args:
        config: ConsolidationConfig with intervals and thresholds.
        tier0: The StreamingProcessor (Tier 0 working memory).
        tier1: The EpisodicMemory (Tier 1 episodic buffer).
        tier2: The EssentialMemory (Tier 2 concept basins).
        tier3: The CoreMemory (Tier 3 axioms).
    """

    def __init__(
        self,
        config: ConsolidationConfig | None = None,
        tier0: StreamingProcessor | None = None,
        tier1: EpisodicMemory | None = None,
        tier2: EssentialMemory | None = None,
        tier3: CoreMemory | None = None,
    ) -> None:
        self._config = config or ConsolidationConfig()
        self._tier0 = tier0
        self._tier1 = tier1
        self._tier2 = tier2
        self._tier3 = tier3
        self._valence: ValenceSystem | None = None

        # Tracking
        self._last_mini: int = 0
        self._last_standard: int = 0
        self._last_deep: int = 0
        self._total_mini: int = 0
        self._total_standard: int = 0
        self._total_deep: int = 0

        # A4: New basin keys from last consolidation (read by processor)
        self._last_new_basin_keys: list[torch.Tensor] = []

        # G3: Topology urgency — Betti B0 count
        self._topology_b0: int = 0

    def set_valence_system(self, valence_system: ValenceSystem | None) -> None:
        """Set the valence system reference for weighted consolidation sampling."""
        self._valence = valence_system

    def set_topology_urgency(self, b0: int) -> None:
        """Set the current Betti B0 count for topology-informed urgency."""
        self._topology_b0 = b0

    def should_consolidate(self, step_count: int) -> ConsolidationMode | None:
        """Determine which consolidation mode to run based on step count.

        Returns the highest-priority mode that is due, or None if nothing is due.
        Deep > Standard > Mini in priority.

        When Betti B0 exceeds threshold (many disconnected components),
        standard consolidation interval is reduced to merge them faster.
        """
        cfg = self._config

        if step_count - self._last_deep >= cfg.deep_interval:
            return ConsolidationMode.DEEP

        # G3: Betti urgency — reduce standard interval when B0 is high
        effective_standard_interval = cfg.standard_interval
        if self._topology_b0 >= cfg.betti_b0_consolidation_threshold:
            effective_standard_interval = int(
                cfg.standard_interval * cfg.betti_consolidation_interval_reduction
            )

        if step_count - self._last_standard >= effective_standard_interval:
            return ConsolidationMode.STANDARD
        if step_count - self._last_mini >= cfg.mini_interval:
            return ConsolidationMode.MINI
        return None

    def consolidate(self, mode: ConsolidationMode, step_count: int) -> None:
        """Run the specified consolidation mode.

        Higher modes include all lower modes:
          - DEEP runs standard then deep
          - STANDARD runs mini then standard
          - MINI runs only mini
        """
        if mode == ConsolidationMode.MINI:
            self.mini_consolidate(step_count)
        elif mode == ConsolidationMode.STANDARD:
            self.mini_consolidate(step_count)
            self.standard_consolidate(step_count)
        elif mode == ConsolidationMode.DEEP:
            self.mini_consolidate(step_count)
            self.standard_consolidate(step_count)
            self.deep_consolidate(step_count)

    def mini_consolidate(self, step_count: int) -> None:
        """Mini-consolidation: Tier 0 -> Tier 1.

        Snapshots the current working memory state and creates an episode
        from the most recent high-surprise input. Stores it in episodic memory.
        """
        if self._tier0 is None or self._tier1 is None:
            return

        self._last_mini = step_count
        self._total_mini += 1

        # Get recent high-surprise inputs from Tier 0 history
        history = self._tier0.surprise_history
        if not history:
            return

        # Find the highest-surprise metric since last mini-consolidation
        recent = history[-(self._config.mini_interval) :]
        if not recent:
            return

        # Only create episodes from inputs that actually triggered updates
        updated_metrics = [m for m in recent if m.updated]
        if not updated_metrics:
            return

        # Use the highest-surprise updated metric
        best = max(updated_metrics, key=lambda m: m.grad_norm)

        # Snapshot working memory state
        field = self._tier0.field
        weight_summary = field.get_weight_summary()
        weight_hash = compute_weight_hash(field)

        # Create a synthetic embedding from the weight summary
        # (In the full orchestrator, the actual input embedding is passed through)
        d_model = field.config.dim
        if weight_summary.shape[0] < d_model:
            embedding = F.pad(weight_summary, (0, d_model - weight_summary.shape[0]))
        else:
            embedding = weight_summary[:d_model]

        logit_snapshot = embedding.clone()  # Placeholder — real logits come from forward pass

        integrity_hash = compute_episode_hash(embedding, logit_snapshot, weight_hash)

        episode = Episode(
            id=self._tier1.allocate_id(),
            timestamp=time.monotonic(),
            modality="working_memory_snapshot",
            provenance_hash=weight_hash[:16],
            input_embedding=embedding.detach().cpu(),
            working_memory_state=weight_summary.detach().cpu(),
            logit_snapshot=logit_snapshot.detach().cpu(),
            surprise_at_storage=best.grad_norm,
            attractor_basin_id=0,  # Updated during standard consolidation
            attractor_distance=0.0,
            preceding_episode_id=self._tier1.last_episode_id,
            following_episode_id=None,
            integrity_hash=integrity_hash,
            weight_hash_at_storage=weight_hash,
        )

        self._tier1.maybe_store(episode)

    def standard_consolidate(self, step_count: int) -> None:
        """Standard consolidation: Tier 1 -> Tier 2.

        Samples replay batches from episodic memory, matches episodes to existing
        Tier 2 basins (or creates new ones), and updates basin keys/values via EMA.
        Also runs topological integrity checks.
        """
        if self._tier1 is None or self._tier2 is None:
            return

        self._last_standard = step_count
        self._total_standard += 1

        cfg = self._config

        # Sample replay batch (optionally valence-weighted)
        if (
            cfg.valence_weighted_sampling
            and self._valence is not None
            and self._tier2 is not None
            and self._tier2.n_active > 0
        ):
            episodes = self._valence_weighted_sample(cfg.replay_batch_size)
        else:
            episodes = self._tier1.sample_for_replay(cfg.replay_batch_size)
        if not episodes:
            return

        # Process each episode
        new_concept_candidates: list[Episode] = []

        for ep in episodes:
            embedding = ep.input_embedding
            if embedding.device != self._tier2.keys.device:
                embedding = embedding.to(self._tier2.keys.device)

            if self._tier2.n_active == 0:
                # No basins yet — everything is a new concept
                new_concept_candidates.append(ep)
                continue

            # Retrieve nearest basin
            with torch.no_grad():
                output, basin_id, attn = self._tier2.retrieve(embedding)

            bid = basin_id.item() if basin_id.dim() == 0 else basin_id[0].item()

            # Check if the episode is close enough to an existing basin
            active_keys = self._tier2.active_keys_tensor
            if active_keys.shape[0] > 0:
                # Find distance to nearest active key
                sims = cosine_similarity_matrix(embedding.unsqueeze(0), active_keys)
                max_sim = sims.max().item()

                if max_sim < cfg.distillation_threshold:
                    # Too far from any existing basin — candidate for new concept
                    new_concept_candidates.append(ep)
                    continue

            # Update existing basin with EMA
            # Modality bit: hash the modality string to a bit position
            modality_bit = 1 << (hash(ep.modality) % 8)
            # Spatial evidence counts as an additional modality (bit 8)
            if ep.spatial_position is not None:
                modality_bit |= 1 << 8

            self._tier2.update_slot(
                bid,
                key_delta=embedding,
                value_delta=ep.working_memory_state.to(self._tier2.values.device)[: self._tier2._config.d_value]
                if ep.working_memory_state.shape[0] >= self._tier2._config.d_value
                else F.pad(
                    ep.working_memory_state.to(self._tier2.values.device),
                    (0, self._tier2._config.d_value - ep.working_memory_state.shape[0]),
                ),
                confidence_update=0.01,
                episode_increment=1,
                modality_bit=modality_bit,
                importance_update=ep.surprise_at_storage * 0.1,
            )

            # Mark episode as consolidated
            ep.consolidation_count += 1
            ep.last_consolidated = time.monotonic()

            # Update episode's basin assignment
            ep.attractor_basin_id = bid

        # Create new concept basins for novel clusters
        if len(new_concept_candidates) >= cfg.new_concept_threshold:
            self._create_new_basins(new_concept_candidates)

        logger.debug(
            "Standard consolidation: processed %d episodes, %d new concept candidates",
            len(episodes),
            len(new_concept_candidates),
        )

        # Field replay phase (co-adaptation): replay episodes through Tier 0
        if cfg.replay_through_field_enabled and self._tier0 is not None:
            replay_batch = self._tier1.sample_for_replay(
                cfg.replay_through_field_batch_size,
            )
            if replay_batch:
                device = next(self._tier0.field.parameters()).device
                replay_losses = []
                for ep in replay_batch:
                    inp = ep.input_embedding.to(device)
                    tgt = ep.logit_snapshot.to(device)
                    rloss = self._tier0.replay_episode(
                        inp, tgt, lr_scale=cfg.replay_lr_scale,
                    )
                    replay_losses.append(rloss)
                logger.debug(
                    "Field replay: %d episodes, mean_loss=%.4f",
                    len(replay_losses),
                    sum(replay_losses) / len(replay_losses),
                )

    def _valence_weighted_sample(self, batch_size: int) -> list[Episode]:
        """Sample episodes with valence-weighted basin probabilities.

        Uses ValenceSystem.get_consolidation_sampling_weights() to bias
        sampling toward basins with high absolute valence (emotionally
        significant experiences).

        Falls back to uniform sampling if valence weights cannot be computed.
        """
        if self._tier1 is None:
            return []

        all_episodes = list(self._tier1._hot) + list(self._tier1._cold)
        if not all_episodes:
            return []

        if len(all_episodes) <= batch_size:
            return list(all_episodes)

        # Group by basin (skip unassigned episodes with basin_id < 0)
        basins: dict[int, list[Episode]] = defaultdict(list)
        unassigned: list[Episode] = []
        for ep in all_episodes:
            if ep.attractor_basin_id >= 0:
                basins[ep.attractor_basin_id].append(ep)
            else:
                unassigned.append(ep)

        basin_ids = list(basins.keys())
        if not basin_ids or self._valence is None:
            return self._tier1.sample_for_replay(batch_size)

        # Get valence-based weights
        weights = self._valence.get_consolidation_sampling_weights(basin_ids)
        weights_list = weights.tolist()

        sampled: list[Episode] = []
        for _ in range(batch_size):
            chosen_basin = random.choices(basin_ids, weights=weights_list, k=1)[0]
            candidates = basins[chosen_basin]
            if candidates:
                sampled.append(random.choice(candidates))

        return sampled

    def deep_consolidate(self, step_count: int) -> None:
        """Deep consolidation: Tier 2 -> Tier 3.

        Scans Tier 2 basins for promotion candidates that meet all Tier 3
        criteria. Emits promotion requests via the event system.
        """
        if self._tier2 is None or self._tier3 is None:
            return

        self._last_deep = step_count
        self._total_deep += 1

        promoted = 0
        active_idx = self._tier2.active_indices

        for slot_tensor in active_idx:
            slot = slot_tensor.item()
            success = self._tier3.promote_from_tier2(self._tier2, slot)
            if success:
                promoted += 1

        # Verify core memory integrity
        failed = self._tier3.verify_integrity()
        if failed:
            logger.error(
                "Deep consolidation: %d core memory slots failed integrity check",
                len(failed),
            )

        logger.info(
            "Deep consolidation: promoted %d basins to core, %d integrity failures",
            promoted,
            len(failed),
        )

    def _create_new_basins(self, candidates: list[Episode]) -> None:
        """Create new Tier 2 basins from a cluster of novel episodes.

        Groups candidates by embedding similarity and creates one basin per cluster.
        Also stores new basin keys for A4 consolidation notification.
        """
        if not candidates or self._tier2 is None:
            return

        self._last_new_basin_keys = []

        device = self._tier2.keys.device

        # Collect embeddings
        embeddings = torch.stack(
            [ep.input_embedding.to(device) for ep in candidates]
        )

        # Simple greedy clustering: assign each embedding to the nearest existing
        # cluster center, or create a new cluster if too far from all
        cluster_centers: list[torch.Tensor] = []
        cluster_members: list[list[Episode]] = []

        for i, ep in enumerate(candidates):
            emb = embeddings[i]

            if not cluster_centers:
                cluster_centers.append(emb)
                cluster_members.append([ep])
                continue

            # Find nearest cluster
            centers = torch.stack(cluster_centers)
            sims = cosine_similarity_matrix(emb.unsqueeze(0), centers)
            max_sim, max_idx = sims.max(dim=-1)

            if max_sim.item() > self._config.distillation_threshold:
                # Join existing cluster
                idx = max_idx.item()
                cluster_members[idx].append(ep)
                # Update center (running mean)
                n = len(cluster_members[idx])
                cluster_centers[idx] = (
                    cluster_centers[idx] * (n - 1) / n + emb / n
                )
            else:
                # Start new cluster
                cluster_centers.append(emb)
                cluster_members.append([ep])

        # Create basins from clusters
        for center, members in zip(cluster_centers, cluster_members):
            if len(members) < 2:
                continue  # Need at least 2 episodes to form a basin

            # Compute mean value from episode working memory states
            values = torch.stack(
                [ep.working_memory_state.to(device) for ep in members]
            )
            d_value = self._tier2._config.d_value
            if values.shape[-1] < d_value:
                values = F.pad(values, (0, d_value - values.shape[-1]))
            else:
                values = values[..., :d_value]
            mean_value = values.mean(dim=0)

            slot = self._tier2.allocate_slot(center, mean_value)
            self._last_new_basin_keys.append(center.detach().clone())

            # Set initial metadata
            modality_bits = 0
            for ep in members:
                modality_bits |= 1 << (hash(ep.modality) % 8)
                # Spatial evidence counts as an additional modality (bit 8)
                if ep.spatial_position is not None:
                    modality_bits |= 1 << 8

            self._tier2.update_slot(
                slot,
                confidence_update=0.1 * len(members),
                episode_increment=len(members),
                modality_bit=modality_bits,
                importance_update=sum(ep.surprise_at_storage for ep in members) * 0.1,
            )

            logger.debug("Created new basin at slot %d from %d episodes", slot, len(members))

    def replay_skim_buffer(
        self,
        salience_gate,
        tier0: StreamingProcessor,
    ) -> None:
        """D2: Replay recent skim buffer entries through the field.

        During standard consolidation, we replay observations that the salience
        gate previously skimmed (low-priority but not entirely irrelevant).
        Uses autoassociative targets since we lack explicit logit snapshots.

        Args:
            salience_gate: SalienceGate with skim_buffer attribute.
            tier0: StreamingProcessor for replay_episode().
        """
        if not hasattr(salience_gate, '_skim_buffer'):
            return
        buffer = salience_gate._skim_buffer
        if not buffer:
            return

        device = next(tier0.field.parameters()).device
        lr_scale = self._config.skim_replay_lr_scale
        replayed = 0

        for entry in buffer[-16:]:  # Replay most recent 16
            emb = entry.to(device) if hasattr(entry, 'to') else entry
            if not isinstance(emb, torch.Tensor):
                continue
            # Autoassociative: use field's own output as target
            with torch.no_grad():
                target = tier0.field(emb).detach()
            tier0.replay_episode(emb, target, lr_scale=lr_scale)
            replayed += 1

        if replayed > 0:
            logger.debug("Skim buffer replay: %d entries, lr_scale=%.3f", replayed, lr_scale)

    @property
    def stats(self) -> dict[str, int]:
        """Consolidation statistics."""
        return {
            "total_mini": self._total_mini,
            "total_standard": self._total_standard,
            "total_deep": self._total_deep,
            "last_mini_step": self._last_mini,
            "last_standard_step": self._last_standard,
            "last_deep_step": self._last_deep,
        }
