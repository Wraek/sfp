"""SceneGraph — generic spatial reasoning over positioned entities.

Environment-agnostic: works with any bridge that provides entity embeddings
and 3D positions.  Maintains a lightweight graph of spatial relations between
entities and injects SPATIAL_* edges into the TransitionStructure so the
reasoning chain can traverse spatial links.

No knowledge of Minecraft, zombies, or any specific game — purely geometric
and learned.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sfp.config import TransitionConfig
from sfp.types import RelationType
from sfp.utils.logging import get_logger

logger = get_logger("reasoning.scene_graph")

# Distance thresholds for spatial relation classification (in normalized units)
NEAR_THRESHOLD = 0.25  # entities closer than this are SPATIAL_NEAR
APPROACH_VELOCITY_THRESHOLD = -0.02  # negative = closing distance


class SceneGraph(nn.Module):
    """Generic scene graph for spatial reasoning.

    Maintains a set of entity nodes with positions and embeddings.
    Classifies spatial relations between pairs and injects them as
    edges into the TransitionStructure.

    Args:
        d_model: Embedding dimension (must match SFP's d_model).
        max_entities: Maximum tracked entities per update.
    """

    def __init__(self, d_model: int, max_entities: int = 32) -> None:
        super().__init__()
        self._d_model = d_model
        self._max_entities = max_entities

        # Spatial relation classifier: given concatenated pair features → relation logits
        # Input: [emb_a(d_model) + emb_b(d_model) + spatial_features(8)] → 4 spatial types
        spatial_input_dim = d_model * 2 + 8
        self.relation_classifier = nn.Sequential(
            nn.Linear(spatial_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 4),  # NEAR, APPROACHING, FLEEING, ABOVE
        )

        # Spatial bias projection: query → spatial attention weights
        self.spatial_bias_proj = nn.Linear(d_model, d_model)

        # Previous positions for velocity estimation
        self._prev_positions: torch.Tensor | None = None
        self._prev_embeddings: torch.Tensor | None = None

    def update(
        self,
        entity_embeddings: torch.Tensor,
        positions: torch.Tensor,
    ) -> list[tuple[int, int, RelationType, float]]:
        """Update the scene graph with new entity data.

        Args:
            entity_embeddings: (N, d_model) — embedding per entity.
            positions: (N, 3) — xyz positions per entity (normalized).

        Returns:
            List of (entity_i, entity_j, relation_type, confidence) tuples
            representing classified spatial relations.
        """
        n = entity_embeddings.shape[0]
        if n == 0:
            self._prev_positions = None
            self._prev_embeddings = None
            return []

        # Estimate velocities from previous frame
        velocities = torch.zeros_like(positions)
        if (
            self._prev_positions is not None
            and self._prev_positions.shape[0] == n
        ):
            velocities = positions - self._prev_positions

        self._prev_positions = positions.detach().clone()
        self._prev_embeddings = entity_embeddings.detach().clone()

        # Classify pairwise spatial relations
        relations: list[tuple[int, int, RelationType, float]] = []

        if n < 2:
            return relations

        for i in range(min(n, self._max_entities)):
            for j in range(i + 1, min(n, self._max_entities)):
                rel_type, confidence = self._classify_pair(
                    entity_embeddings[i], entity_embeddings[j],
                    positions[i], positions[j],
                    velocities[i], velocities[j],
                )
                if confidence > 0.3:
                    relations.append((i, j, rel_type, confidence))

        return relations

    def _classify_pair(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
        vel_a: torch.Tensor,
        vel_b: torch.Tensor,
    ) -> tuple[RelationType, float]:
        """Classify the spatial relation between two entities.

        Uses a learned classifier on top of geometric features.
        """
        # Spatial features: relative position, distance, relative velocity, vertical diff
        rel_pos = pos_b - pos_a
        distance = rel_pos.norm().item()
        rel_vel = vel_b - vel_a
        closing_speed = -(rel_pos * rel_vel).sum().item() / max(distance, 1e-6)
        vertical_diff = rel_pos[1].item()  # y-axis

        spatial_feats = torch.tensor([
            rel_pos[0].item(), rel_pos[1].item(), rel_pos[2].item(),
            distance,
            rel_vel[0].item(), rel_vel[1].item(), rel_vel[2].item(),
            vertical_diff,
        ], device=emb_a.device, dtype=emb_a.dtype)

        # Concatenate and classify
        pair_input = torch.cat([emb_a, emb_b, spatial_feats])
        logits = self.relation_classifier(pair_input)
        probs = torch.softmax(logits, dim=0)

        # Map to RelationType
        type_map = [
            RelationType.SPATIAL_NEAR,
            RelationType.SPATIAL_APPROACHING,
            RelationType.SPATIAL_FLEEING,
            RelationType.SPATIAL_ABOVE,
        ]

        best_idx = probs.argmax().item()
        confidence = probs[best_idx].item()

        # Override with geometric heuristics for strong signals
        if distance < NEAR_THRESHOLD:
            return RelationType.SPATIAL_NEAR, max(confidence, 0.8)
        if closing_speed > 0.05:
            return RelationType.SPATIAL_APPROACHING, max(confidence, 0.6)
        if closing_speed < -0.05:
            return RelationType.SPATIAL_FLEEING, max(confidence, 0.6)
        if abs(vertical_diff) > 0.3 and distance < 0.5:
            return RelationType.SPATIAL_ABOVE, max(confidence, 0.5)

        return type_map[best_idx], confidence

    def inject_into_transitions(
        self,
        transitions: object,
        relations: list[tuple[int, int, RelationType, float]],
        entity_basin_map: dict[int, int] | None = None,
    ) -> int:
        """Inject spatial relations as edges in the TransitionStructure.

        Args:
            transitions: TransitionStructure instance.
            relations: Output from update().
            entity_basin_map: Maps entity index → Tier 2 basin ID.
                If None, entity indices are used directly as basin IDs.

        Returns:
            Number of edges injected.
        """
        injected = 0
        for ent_i, ent_j, rel_type, confidence in relations:
            src = entity_basin_map.get(ent_i, ent_i) if entity_basin_map else ent_i
            tgt = entity_basin_map.get(ent_j, ent_j) if entity_basin_map else ent_j

            try:
                transitions.add_edge(
                    src, tgt,
                    relation=rel_type,
                    weight=confidence,
                )
                injected += 1
            except (IndexError, RuntimeError):
                pass  # Basin doesn't exist yet — skip

        return injected

    @torch.no_grad()
    def compute_spatial_bias(
        self,
        query_vec: torch.Tensor,
        entity_embeddings: torch.Tensor,
        positions: torch.Tensor,
    ) -> dict[int, float]:
        """Compute spatial reasoning bias for nearby entities.

        Returns a bias dict mapping entity indices to attention weights,
        suitable for merging with goal/valence bias in the reasoning chain.

        Args:
            query_vec: (d_model,) current query vector.
            entity_embeddings: (N, d_model) entity embeddings.
            positions: (N, 3) entity positions.

        Returns:
            Dict mapping entity index → spatial bias value.
        """
        if entity_embeddings.shape[0] == 0:
            return {}

        # Project query to spatial attention space
        query_spatial = self.spatial_bias_proj(query_vec)  # (d_model,)

        # Dot product attention with entities
        scores = torch.mv(entity_embeddings, query_spatial)  # (N,)
        scores = torch.sigmoid(scores)

        # Weight by inverse distance (closer = more relevant)
        distances = positions.norm(dim=1)  # (N,)
        distance_weight = 1.0 / (1.0 + distances)

        biased_scores = scores * distance_weight

        # Return as dict, filtering low-score entries
        bias: dict[int, float] = {}
        for i in range(biased_scores.shape[0]):
            val = biased_scores[i].item()
            if val > 0.1:
                bias[i] = val

        return bias
