"""Transition Learning — discovers and creates edges during consolidation.

Implements the transition learning algorithms from sfp-reasoning-chains.md:
  - Temporal co-occurrence mining
  - Causal asymmetry detection
  - Compositional relation discovery
  - Analogical structure detection
  - Inhibitory relation inference
  - Chain shortcut creation
"""

from __future__ import annotations

from collections import defaultdict

import torch

from sfp.config import ReasoningChainConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import Episode, RelationType
from sfp.utils.logging import get_logger
from sfp.utils.math import cosine_similarity_matrix

logger = get_logger("reasoning.learning")


class TransitionLearner:
    """Discovers typed relations between concept basins from episodic memory patterns.

    Called during standard consolidation to grow the transition graph.
    """

    def __init__(
        self,
        tier2: EssentialMemory,
        transitions: TransitionStructure,
        temporal_window: int = 5,
        similarity_threshold: float = 0.3,
        causal_asymmetry_ratio: float = 2.0,
        compositional_distance: float = 0.7,
    ) -> None:
        self._tier2 = tier2
        self._transitions = transitions
        self._temporal_window = temporal_window
        self._similarity_threshold = similarity_threshold
        self._causal_asymmetry_ratio = causal_asymmetry_ratio
        self._compositional_distance = compositional_distance

    def learn_from_episodes(self, episodes: list[Episode]) -> int:
        """Mine temporal co-occurrence and causal relations from a batch of episodes.

        Episodes within the temporal window are considered potentially related.
        Temporal ordering provides directionality for causal inference.

        Args:
            episodes: List of episodes from Tier 1 replay.

        Returns:
            Number of new edges created or updated.
        """
        if len(episodes) < 2:
            return 0

        # Sort episodes by timestamp
        sorted_eps = sorted(episodes, key=lambda e: e.timestamp)
        edges_created = 0

        # Temporal co-occurrence: episodes within the window are linked
        for i, ep_a in enumerate(sorted_eps):
            basin_a = ep_a.attractor_basin_id
            if basin_a < 0:
                continue

            for j in range(i + 1, min(i + 1 + self._temporal_window, len(sorted_eps))):
                ep_b = sorted_eps[j]
                basin_b = ep_b.attractor_basin_id
                if basin_b < 0 or basin_a == basin_b:
                    continue

                # Temporal relation: A precedes B
                self._transitions.add_edge(
                    basin_a, basin_b,
                    relation=RelationType.TEMPORAL,
                    weight=0.1,
                    confidence=0.05,
                )
                edges_created += 1

                # Check for causal asymmetry: if A->B is much stronger than B->A
                # we infer a causal direction
                ab_targets, ab_weights, _ = self._transitions.get_outgoing(basin_a)
                ba_targets, ba_weights, _ = self._transitions.get_outgoing(basin_b)

                ab_strength = 0.0
                for t, w in zip(ab_targets.tolist(), ab_weights.tolist()):
                    if t == basin_b:
                        ab_strength = w
                        break

                ba_strength = 0.0
                for t, w in zip(ba_targets.tolist(), ba_weights.tolist()):
                    if t == basin_a:
                        ba_strength = w
                        break

                if ab_strength > 0 and ba_strength > 0:
                    if ab_strength / (ba_strength + 1e-8) > self._causal_asymmetry_ratio:
                        self._transitions.add_edge(
                            basin_a, basin_b,
                            relation=RelationType.CAUSAL,
                            weight=0.2,
                            confidence=0.1,
                        )
                        edges_created += 1

        return edges_created

    def learn_compositional_relations(self) -> int:
        """Discover compositional (part-of, is-a) relations from geometric proximity.

        Two basins with high embedding similarity that are also close in key space
        are candidates for compositional relations.

        Returns:
            Number of compositional edges created.
        """
        if self._tier2.n_active < 2:
            return 0

        active_keys = self._tier2.active_keys_tensor
        active_idx = self._tier2.active_indices

        # Compute pairwise cosine similarity
        sims = cosine_similarity_matrix(active_keys, active_keys)

        edges_created = 0
        n = active_keys.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j].item()

                # High but not identical similarity suggests compositional relation
                if self._compositional_distance < sim < 0.95:
                    basin_i = active_idx[i].item()
                    basin_j = active_idx[j].item()

                    # The basin with fewer episodes is more likely the "part"
                    count_i = self._tier2.episode_count[basin_i].item()
                    count_j = self._tier2.episode_count[basin_j].item()

                    if count_i > count_j:
                        # j is part of i
                        self._transitions.add_edge(
                            basin_j, basin_i,
                            relation=RelationType.COMPOSITIONAL,
                            weight=sim * 0.5,
                            confidence=0.1,
                        )
                    else:
                        self._transitions.add_edge(
                            basin_i, basin_j,
                            relation=RelationType.COMPOSITIONAL,
                            weight=sim * 0.5,
                            confidence=0.1,
                        )
                    edges_created += 1

        return edges_created

    def learn_inhibitory_relations(self) -> int:
        """Discover inhibitory relations from prediction errors.

        Basins that are geometrically close but semantically distant (one
        predicting the other poorly) suggest inhibitory relations.

        Returns:
            Number of inhibitory edges created.
        """
        if self._tier2.n_active < 2:
            return 0

        active_keys = self._tier2.active_keys_tensor
        active_values = self._tier2.active_values_tensor
        active_idx = self._tier2.active_indices

        edges_created = 0
        n = active_keys.shape[0]

        for i in range(n):
            for j in range(i + 1, n):
                key_sim = torch.nn.functional.cosine_similarity(
                    active_keys[i].unsqueeze(0),
                    active_keys[j].unsqueeze(0),
                ).item()

                value_sim = torch.nn.functional.cosine_similarity(
                    active_values[i].unsqueeze(0),
                    active_values[j].unsqueeze(0),
                ).item()

                # Keys are similar but values are very different = inhibitory
                if key_sim > 0.5 and value_sim < 0.2:
                    basin_i = active_idx[i].item()
                    basin_j = active_idx[j].item()

                    self._transitions.add_edge(
                        basin_i, basin_j,
                        relation=RelationType.INHIBITORY,
                        weight=0.3,
                        confidence=0.1,
                    )
                    self._transitions.add_edge(
                        basin_j, basin_i,
                        relation=RelationType.INHIBITORY,
                        weight=0.3,
                        confidence=0.1,
                    )
                    edges_created += 2

        return edges_created


class ChainShortcutLearner:
    """Creates shortcut edges for frequently-traversed reasoning chains.

    When the same chain of basins (A -> B -> C -> D) is traversed more than
    a configurable number of times, a direct shortcut edge (A -> D) is created.
    This implements the expertise gradient: novice chains become expert shortcuts.

    Args:
        transitions: The TransitionStructure.
        config: ReasoningChainConfig (uses shortcut_min_traversals).
    """

    def __init__(
        self,
        transitions: TransitionStructure,
        config: ReasoningChainConfig | None = None,
    ) -> None:
        self._transitions = transitions
        self._config = config or ReasoningChainConfig()
        # Track observed chain fragments: (start, end) -> count
        self._chain_counts: dict[tuple[int, int], int] = defaultdict(int)
        # Track quality scores
        self._chain_quality: dict[tuple[int, int], float] = defaultdict(float)

    def observe_chain(self, visited_basins: list[int], quality_score: float = 1.0) -> None:
        """Record a reasoning chain traversal for potential shortcutting.

        Args:
            visited_basins: Ordered list of basin IDs visited during the chain.
            quality_score: How useful the chain result was (0-1).
        """
        if len(visited_basins) < 3:
            # Need at least 3 hops to create a meaningful shortcut
            return

        start = visited_basins[0]
        end = visited_basins[-1]
        pair = (start, end)

        self._chain_counts[pair] += 1
        # Running average quality
        n = self._chain_counts[pair]
        self._chain_quality[pair] = (
            self._chain_quality[pair] * (n - 1) / n + quality_score / n
        )

    def create_shortcuts(self) -> int:
        """Create shortcut edges for chains that exceed the traversal threshold.

        Returns:
            Number of shortcut edges created.
        """
        min_traversals = self._config.shortcut_min_traversals
        created = 0

        pairs_to_remove: list[tuple[int, int]] = []

        for (start, end), count in self._chain_counts.items():
            if count >= min_traversals:
                quality = self._chain_quality[(start, end)]
                if quality > 0.3:  # Only shortcut useful chains
                    self._transitions.add_edge(
                        start, end,
                        relation=RelationType.ASSOCIATIVE,
                        weight=quality,
                        confidence=min(1.0, count / (min_traversals * 3)),
                    )
                    created += 1
                    logger.info(
                        "Created shortcut edge %d -> %d (traversals=%d, quality=%.2f)",
                        start, end, count, quality,
                    )
                pairs_to_remove.append((start, end))

        # Reset counters for created shortcuts
        for pair in pairs_to_remove:
            del self._chain_counts[pair]
            del self._chain_quality[pair]

        return created
