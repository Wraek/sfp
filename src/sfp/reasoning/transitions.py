"""Transition Structure — sparse directed graph of typed relations between Tier 2 concept basins.

Implements the transition structure from sfp-reasoning-chains.md. Edges connect
concept basins with typed relations (causal, temporal, compositional, analogical,
inhibitory, associative), learned embeddings, and confidence tracking.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sfp.config import TransitionConfig
from sfp.types import RelationType
from sfp.utils.logging import get_logger

logger = get_logger("reasoning.transitions")


class TransitionStructure(nn.Module):
    """Sparse directed graph of typed relations between Tier 2 basins.

    Stores edges in COO format with typed relation embeddings, confidence
    tracking, and learned scoring for multi-hop traversal.

    Args:
        config: TransitionConfig specifying dimensions and capacity.
        d_model: Model dimensionality (for transition scoring).
    """

    def __init__(self, config: TransitionConfig | None = None, d_model: int = 512) -> None:
        super().__init__()
        cfg = config or TransitionConfig()
        self._config = cfg
        self._d_model = d_model

        # Edge storage in COO format
        self.register_buffer(
            "source", torch.zeros(cfg.max_edges, dtype=torch.long)
        )
        self.register_buffer(
            "target", torch.zeros(cfg.max_edges, dtype=torch.long)
        )
        self.register_buffer(
            "weight", torch.zeros(cfg.max_edges)
        )
        self.register_buffer(
            "edge_confidence", torch.zeros(cfg.max_edges)
        )
        self.register_buffer(
            "edge_episode_count", torch.zeros(cfg.max_edges, dtype=torch.long)
        )
        self.register_buffer(
            "relation_type", torch.zeros(cfg.max_edges, dtype=torch.long)
        )
        self.register_buffer(
            "active_edge_mask", torch.zeros(cfg.max_edges, dtype=torch.bool)
        )

        # Learned relation type prototypes (6 types x d_relation)
        self.relation_prototypes = nn.Parameter(
            torch.randn(cfg.n_relation_types, cfg.d_relation) * 0.02
        )

        # Per-edge learned embeddings
        self.relation_embeddings = nn.Parameter(
            torch.zeros(cfg.max_edges, cfg.d_relation)
        )

        # Scoring network: projects query context + relation embedding to a score
        self.transition_query = nn.Linear(d_model + cfg.d_relation, 1, bias=True)

        self._n_active: int = 0

    def add_edge(
        self,
        src: int,
        tgt: int,
        relation: RelationType | int = RelationType.ASSOCIATIVE,
        weight: float = 0.1,
        confidence: float = 0.1,
    ) -> int:
        """Add a directed edge between two basins.

        If an edge from src to tgt already exists, updates it instead of creating
        a duplicate. If at capacity, evicts the edge with the lowest confidence.

        Args:
            src: Source basin index.
            tgt: Target basin index.
            relation: Relation type (enum or int).
            weight: Edge weight.
            confidence: Initial confidence.

        Returns:
            The edge index.
        """
        rel_idx = relation.value if isinstance(relation, RelationType) else relation

        # Check for existing edge
        existing = self._find_edge(src, tgt)
        if existing is not None:
            # Update existing edge
            self.weight[existing] = max(self.weight[existing].item(), weight)
            self.edge_confidence[existing] = min(
                1.0, self.edge_confidence[existing].item() + confidence * 0.1
            )
            self.edge_episode_count[existing] += 1
            return existing

        # Find a free slot
        if self._n_active < self._config.max_edges:
            inactive = (~self.active_edge_mask).nonzero(as_tuple=True)[0]
            slot = inactive[0].item()
        else:
            # Evict lowest-confidence edge
            active_idx = self.active_edge_mask.nonzero(as_tuple=True)[0]
            conf_scores = self.edge_confidence[active_idx]
            worst = conf_scores.argmin()
            slot = active_idx[worst].item()
            self._n_active -= 1

        with torch.no_grad():
            self.source[slot] = src
            self.target[slot] = tgt
            self.weight[slot] = weight
            self.edge_confidence[slot] = confidence
            self.edge_episode_count[slot] = 1
            self.relation_type[slot] = rel_idx
            self.active_edge_mask[slot] = True
            # Initialize edge embedding from the relation prototype
            self.relation_embeddings.data[slot] = self.relation_prototypes[rel_idx].data.clone()

        self._n_active += 1
        return slot

    def get_outgoing(self, basin_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all outgoing edges from a basin.

        Returns:
            Tuple of:
              - target_ids: (E,) target basin indices.
              - weights: (E,) edge weights.
              - edge_indices: (E,) global edge indices (for accessing embeddings).
        """
        if self._n_active == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.source.device)
            empty_f = torch.zeros(0, device=self.weight.device)
            return empty, empty_f, empty

        active_idx = self.active_edge_mask.nonzero(as_tuple=True)[0]
        active_sources = self.source[active_idx]
        match = active_sources == basin_id
        matched_idx = active_idx[match]

        targets = self.target[matched_idx]
        weights = self.weight[matched_idx]
        return targets, weights, matched_idx

    def get_incoming(self, basin_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all incoming edges to a basin.

        Returns:
            Tuple of (source_ids, weights, edge_indices).
        """
        if self._n_active == 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.source.device)
            empty_f = torch.zeros(0, device=self.weight.device)
            return empty, empty_f, empty

        active_idx = self.active_edge_mask.nonzero(as_tuple=True)[0]
        active_targets = self.target[active_idx]
        match = active_targets == basin_id
        matched_idx = active_idx[match]

        sources = self.source[matched_idx]
        weights = self.weight[matched_idx]
        return sources, weights, matched_idx

    def compute_transition_scores(
        self, basin_id: int, query_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score outgoing transitions from a basin based on query context.

        Args:
            basin_id: The current basin to transition from.
            query_context: (d_model,) context vector (e.g., query residual).

        Returns:
            Tuple of:
              - scores: (E,) relevance scores for each outgoing edge.
              - target_ids: (E,) target basin indices.
        """
        targets, weights, edge_indices = self.get_outgoing(basin_id)

        if targets.shape[0] == 0:
            return torch.zeros(0, device=query_context.device), targets

        # Get edge embeddings
        edge_embs = self.relation_embeddings[edge_indices]  # (E, d_relation)
        confidences = self.edge_confidence[edge_indices]  # (E,)

        # Expand query context to match edges
        query_expanded = query_context.unsqueeze(0).expand(edge_embs.shape[0], -1)  # (E, d_model)

        # Concatenate query context with edge embeddings
        combined = torch.cat([query_expanded, edge_embs], dim=-1)  # (E, d_model + d_relation)

        # Score each transition
        raw_scores = self.transition_query(combined).squeeze(-1)  # (E,)

        # Weight by edge confidence and weight
        scores = raw_scores * confidences * weights

        return scores, targets

    def _find_edge(self, src: int, tgt: int) -> int | None:
        """Find an existing edge between src and tgt, or return None."""
        if self._n_active == 0:
            return None

        active_idx = self.active_edge_mask.nonzero(as_tuple=True)[0]
        src_match = self.source[active_idx] == src
        tgt_match = self.target[active_idx] == tgt
        both = src_match & tgt_match

        if both.any():
            return active_idx[both][0].item()
        return None

    @property
    def n_active_edges(self) -> int:
        """Number of active edges in the graph."""
        return self._n_active

    def get_edge_info(self, edge_idx: int) -> dict:
        """Get metadata for a specific edge."""
        return {
            "source": self.source[edge_idx].item(),
            "target": self.target[edge_idx].item(),
            "weight": self.weight[edge_idx].item(),
            "confidence": self.edge_confidence[edge_idx].item(),
            "episode_count": self.edge_episode_count[edge_idx].item(),
            "relation_type": RelationType(self.relation_type[edge_idx].item()).name,
            "active": self.active_edge_mask[edge_idx].item(),
        }
