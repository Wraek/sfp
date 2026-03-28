"""Defense Layer 5: Topological manifold monitoring for corruption detection.

Monitors the concept manifold for structural anomalies that indicate poisoning:
  - Basin merging/splitting (unexpected component count changes)
  - Transition graph corruption (implausible edges)
  - Reasoning loop detection (cycles in the traversal graph)
"""

from __future__ import annotations

import torch

from sfp.config import DefenseConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import RelationType
from sfp.utils.logging import get_logger
from sfp.utils.math import cosine_similarity_matrix

logger = get_logger("defense.topology")


class ManifoldIntegrityMonitor:
    """Monitors the concept manifold and transition graph for structural corruption.

    Runs periodic integrity checks that detect:
      1. Unexpected basin merging (basins becoming too similar)
      2. Unexpected basin splitting (new isolated components appearing)
      3. Transition graph corruption (strong edges between semantically distant basins)
      4. Reasoning loops (cycles that could trap the reasoning chain)

    Args:
        config: DefenseConfig with monitoring thresholds.
        merge_alarm_threshold: Max number of merge alarms before raising alert.
        topology_change_threshold: Max unexpected topology changes per check.
    """

    def __init__(self, config: DefenseConfig | None = None) -> None:
        cfg = config or DefenseConfig()
        self._config = cfg
        self._merge_alarm_threshold = cfg.merge_alarm_threshold
        self._topology_change_threshold = cfg.topology_change_threshold
        self._previous_n_components: int | None = None
        self._check_count: int = 0

    def check_basin_integrity(self, tier2: EssentialMemory) -> list[str]:
        """Check Tier 2 basins for anomalous structural changes.

        Detects:
          - Basin merging: pairs of basins with cosine similarity > 0.95
          - Basin isolation: basins with no similarity > 0.1 to any other basin

        Returns:
            List of alert messages. Empty list = no anomalies.
        """
        alerts: list[str] = []

        if tier2.n_active < 2:
            return alerts

        active_keys = tier2.active_keys_tensor
        active_idx = tier2.active_indices
        n = active_keys.shape[0]

        # Pairwise cosine similarity
        sims = cosine_similarity_matrix(active_keys, active_keys)

        # Check for merging basins (dangerously high similarity)
        merge_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                sim = sims[i, j].item()
                if sim > 0.95:
                    basin_i = active_idx[i].item()
                    basin_j = active_idx[j].item()
                    merge_count += 1
                    if merge_count <= 5:  # Don't flood with alerts
                        alerts.append(
                            f"Basin merge warning: slots {basin_i} and {basin_j} "
                            f"have cosine similarity {sim:.4f}"
                        )

        if merge_count > self._merge_alarm_threshold:
            alerts.append(
                f"ALERT: {merge_count} basin pairs exceed merge threshold — "
                f"potential manifold corruption"
            )

        # Estimate connected components via thresholded adjacency
        n_components = self._estimate_components(sims, threshold=0.3)
        if self._previous_n_components is not None:
            change = abs(n_components - self._previous_n_components)
            if change > self._topology_change_threshold:
                alerts.append(
                    f"ALERT: Component count changed from {self._previous_n_components} "
                    f"to {n_components} (change={change})"
                )
        self._previous_n_components = n_components

        self._check_count += 1
        return alerts

    def check_transition_integrity(
        self, transitions: TransitionStructure, tier2: EssentialMemory
    ) -> list[str]:
        """Check the transition graph for implausible edges.

        A strong (high-confidence) edge between semantically distant basins
        is suspicious — it may indicate graph corruption.

        Returns:
            List of alert messages.
        """
        alerts: list[str] = []

        if transitions.n_active_edges == 0 or tier2.n_active < 2:
            return alerts

        active_mask = transitions.active_edge_mask
        active_idx = active_mask.nonzero(as_tuple=True)[0]

        suspicious_count = 0

        for edge_idx_tensor in active_idx:
            edge_idx = edge_idx_tensor.item()
            confidence = transitions.edge_confidence[edge_idx].item()

            # Only check high-confidence edges
            if confidence < 0.5:
                continue

            src = transitions.source[edge_idx].item()
            tgt = transitions.target[edge_idx].item()

            # Check if both basins are active
            if not tier2.active_mask[src] or not tier2.active_mask[tgt]:
                continue

            # Compute semantic distance
            src_key = tier2.keys[src]
            tgt_key = tier2.keys[tgt]
            sim = torch.nn.functional.cosine_similarity(
                src_key.unsqueeze(0), tgt_key.unsqueeze(0)
            ).item()

            # High-confidence edge between very dissimilar basins is suspicious
            # Exception: inhibitory relations are expected to connect dissimilar concepts
            rel_type = transitions.relation_type[edge_idx].item()
            is_inhibitory = rel_type == RelationType.INHIBITORY.value

            if sim < 0.1 and not is_inhibitory:
                suspicious_count += 1
                if suspicious_count <= 5:
                    alerts.append(
                        f"Suspicious edge: {src}->{tgt} (confidence={confidence:.2f}, "
                        f"similarity={sim:.4f}, type={RelationType(rel_type).name})"
                    )

        if suspicious_count > 3:
            alerts.append(
                f"ALERT: {suspicious_count} suspicious high-confidence edges detected"
            )

        return alerts

    def detect_reasoning_loops(self, transitions: TransitionStructure) -> list[list[int]]:
        """Detect cycles in the transition graph that could trap reasoning chains.

        Uses DFS-based cycle detection on the directed graph.

        Returns:
            List of detected cycles (each cycle is a list of basin IDs).
        """
        if transitions.n_active_edges == 0:
            return []

        # Build adjacency list from active edges
        adjacency: dict[int, list[int]] = {}
        active_idx = transitions.active_edge_mask.nonzero(as_tuple=True)[0]

        for edge_idx_tensor in active_idx:
            edge_idx = edge_idx_tensor.item()
            src = transitions.source[edge_idx].item()
            tgt = transitions.target[edge_idx].item()
            if src not in adjacency:
                adjacency[src] = []
            adjacency[src].append(tgt)

        # DFS cycle detection
        WHITE, GRAY, BLACK = 0, 1, 2
        color: dict[int, int] = {n: WHITE for n in adjacency}
        # Also add nodes that only appear as targets
        for neighbors in adjacency.values():
            for n in neighbors:
                if n not in color:
                    color[n] = WHITE

        cycles: list[list[int]] = []
        path: list[int] = []

        def dfs(node: int) -> None:
            color[node] = GRAY
            path.append(node)

            for neighbor in adjacency.get(node, []):
                if color.get(neighbor, WHITE) == GRAY:
                    # Found a cycle — extract it from path
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif color.get(neighbor, WHITE) == WHITE:
                    dfs(neighbor)

            path.pop()
            color[node] = BLACK

        for node in list(color.keys()):
            if color[node] == WHITE:
                dfs(node)

        if cycles:
            logger.warning("Detected %d reasoning loops in transition graph", len(cycles))

        return cycles

    def _estimate_components(self, similarity_matrix: torch.Tensor, threshold: float = 0.3) -> int:
        """Estimate the number of connected components via thresholded adjacency.

        Uses a simple union-find on the similarity matrix.
        """
        n = similarity_matrix.shape[0]
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_matrix[i, j].item() > threshold:
                    union(i, j)

        return len(set(find(i) for i in range(n)))
