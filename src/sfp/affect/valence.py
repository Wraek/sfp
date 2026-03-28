"""Valence & Affect — affective valence, mood tracking, and safety-biased reasoning.

Annotates episodes, basins, and edges with scalar valence [-1, 1] and vector
valence embeddings.  Maintains 3-timescale mood EMAs (immediate, short-term,
baseline) that influence reasoning mode (approach / avoidance / neutral) with
safety-biased asymmetry (avoidance 2x stronger than approach).

The valence system does not add new data structures — it annotates existing
ones with an affective "color" that enables prioritized memory formation,
directional reasoning, and contextual mood.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sfp.config import ValenceConfig
from sfp.types import ValenceSignal
from sfp.utils.logging import get_logger

logger = get_logger("affect.valence")


class ValenceSystem(nn.Module):
    """Affective valence and mood system.

    Computes scalar valence [-1, 1] and vector valence embeddings from five
    weighted sources: RL reward, user feedback, goal alignment, prediction
    satisfaction, and a learned component.  Maintains 3-timescale mood EMAs
    whose composite value influences risk tolerance and vigilance.

    Basin and edge valence annotations are maintained as registered buffers
    for persistence across checkpoints.

    Architecture:
      - Valence projection: ``Linear(d_model, d_valence, bias=False)``
      - Context projection: ``Linear(d_valence, d_model, bias=False)``
      - Learned valence:    ``Linear(d_model, 128) -> GELU -> Linear(128, 1) -> Tanh``
      - Basin valence:      registered buffers (4096 slots)
      - Edge valence:       registered buffers (40960 slots)

    Args:
        config: ValenceConfig with source weights and mood parameters.
        d_model: Backbone / field dimensionality.
    """

    def __init__(
        self,
        config: ValenceConfig | None = None,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        cfg = config or ValenceConfig()
        self._config = cfg
        self._d_model = d_model
        d_val = cfg.d_valence_embedding

        # --- Valence projection: d_model → d_valence (no bias) ---
        self.valence_proj = nn.Linear(d_model, d_val, bias=False)

        # --- Context projection: d_valence → d_model (no bias) ---
        self.context_proj = nn.Linear(d_val, d_model, bias=False)

        # --- Learned valence predictor ---
        self.learned_valence = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

        # --- Mood state (Python floats, not tensor parameters) ---
        self._mood_immediate: float = 0.0
        self._mood_short_term: float = 0.0
        self._mood_baseline: float = 0.0

        # --- Basin valence annotations (Tier 2 compatible, 4096 slots) ---
        self.register_buffer(
            "basin_valence_scalar", torch.zeros(4096),
        )
        self.register_buffer(
            "basin_valence_embedding", torch.zeros(4096, d_val),
        )
        self.register_buffer(
            "basin_valence_count", torch.zeros(4096, dtype=torch.long),
        )

        # --- Edge valence annotations (TransitionStructure, 40960 edges) ---
        self.register_buffer("edge_valence", torch.zeros(40960))
        self.register_buffer(
            "edge_valence_count", torch.zeros(40960, dtype=torch.long),
        )

        # --- RL value running normalizer ---
        self._rl_mean: float = 0.0
        self._rl_var: float = 1.0
        self._rl_momentum: float = 0.99

        logger.info(
            "ValenceSystem initialized: d_valence=%d, approach=%.2f, "
            "avoidance=%.2f, mood_taus=(%.3f, %.3f, %.4f)",
            d_val, cfg.approach_weight, cfg.avoidance_weight,
            cfg.immediate_tau, cfg.short_term_tau, cfg.baseline_tau,
        )

    # ------------------------------------------------------------------
    # Valence computation
    # ------------------------------------------------------------------

    def compute_valence(
        self,
        embedding: torch.Tensor,
        reward: float = 0.0,
        user_feedback: float = 0.0,
        goal_alignment: float = 0.0,
        prediction_satisfaction: float = 0.0,
    ) -> ValenceSignal:
        """Compute valence for an input embedding.

        Combines 4 explicit sources (RL reward, user feedback, goal alignment,
        prediction satisfaction) with a learned component via ``learned_blend``.
        Updates 3-timescale mood EMAs and derives risk tolerance + vigilance.

        Args:
            embedding: (d_model,) embedding to assign valence to.
            reward: Raw RL reward value (will be normalized).
            user_feedback: Explicit user feedback [-1, 1].
            goal_alignment: Goal alignment score [-1, 1].
            prediction_satisfaction: Prediction confirmation score [-1, 1].

        Returns:
            ValenceSignal with scalar, embedding, mood, and modulators.
        """
        cfg = self._config

        with torch.no_grad():
            inp = embedding.unsqueeze(0)  # (1, d_model)
            # Learned valence from embedding content
            learned_scalar = self.learned_valence(inp).item()
            # Valence embedding
            valence_emb = self.valence_proj(inp).squeeze(0)  # (d_val,)
            # Context projection for backbone injection
            context = self.context_proj(valence_emb.unsqueeze(0)).squeeze(0)

        # Normalize RL reward to ~[-1, 1] via running statistics
        normalized_reward = self._normalize_rl(reward)

        # Weighted combination of explicit sources
        composite = (
            cfg.rl_value_weight * normalized_reward
            + cfg.user_feedback_weight * user_feedback
            + cfg.goal_alignment_weight * goal_alignment
            + cfg.prediction_satisfaction_weight * prediction_satisfaction
        )

        # Blend with learned component
        final_scalar = (
            (1.0 - cfg.learned_blend) * composite
            + cfg.learned_blend * learned_scalar
        )
        final_scalar = max(-1.0, min(1.0, final_scalar))

        # Update 3-timescale mood EMAs
        self._mood_immediate = (
            cfg.immediate_tau * self._mood_immediate
            + (1.0 - cfg.immediate_tau) * final_scalar
        )
        self._mood_short_term = (
            cfg.short_term_tau * self._mood_short_term
            + (1.0 - cfg.short_term_tau) * final_scalar
        )
        self._mood_baseline = (
            cfg.baseline_tau * self._mood_baseline
            + (1.0 - cfg.baseline_tau) * final_scalar
        )

        # Composite mood (weighted blend of 3 timescales)
        w = cfg.mood_weights
        composite_mood = (
            w[0] * self._mood_immediate
            + w[1] * self._mood_short_term
            + w[2] * self._mood_baseline
        )
        composite_mood = max(-1.0, min(1.0, composite_mood))

        # Risk tolerance [0.2, 0.8]: positive mood → more risk-tolerant
        risk_tolerance = max(0.2, min(0.8, 0.5 + 0.3 * composite_mood))

        # Vigilance [0.2, 0.8]: negative mood → more vigilant
        vigilance = max(0.2, min(0.8, 0.5 - 0.3 * composite_mood))

        return ValenceSignal(
            scalar_valence=final_scalar,
            valence_embedding=valence_emb.detach(),
            projected_context=context.detach(),
            mood_immediate=self._mood_immediate,
            mood_short_term=self._mood_short_term,
            mood_baseline=self._mood_baseline,
            composite_mood=composite_mood,
            risk_tolerance=risk_tolerance,
            vigilance=vigilance,
        )

    # ------------------------------------------------------------------
    # Basin and edge annotations
    # ------------------------------------------------------------------

    def annotate_basin(
        self,
        basin_id: int,
        valence_scalar: float,
        valence_embedding: torch.Tensor,
    ) -> None:
        """Update running valence EMA for a Tier 2 basin.

        Uses adaptive momentum that starts low (responsive) and increases
        toward ``basin_valence_ema_decay`` as more observations accumulate.

        Args:
            basin_id: Basin index in Tier 2.
            valence_scalar: Scalar valence [-1, 1] from recent inference.
            valence_embedding: (d_valence,) valence embedding vector.
        """
        decay = self._config.basin_valence_ema_decay
        count = self.basin_valence_count[basin_id].item()
        momentum = min(decay, 1.0 - 1.0 / (count + 1))

        self.basin_valence_scalar[basin_id] = (
            momentum * self.basin_valence_scalar[basin_id]
            + (1.0 - momentum) * valence_scalar
        )

        emb = valence_embedding.detach()
        if emb.dim() > 1:
            emb = emb.squeeze(0)
        self.basin_valence_embedding[basin_id] = (
            momentum * self.basin_valence_embedding[basin_id]
            + (1.0 - momentum) * emb
        )

        self.basin_valence_count[basin_id] += 1

    def annotate_edge(self, edge_idx: int, valence_scalar: float) -> None:
        """Update running valence EMA for a transition edge.

        Args:
            edge_idx: Edge index in the transition structure.
            valence_scalar: Chain-level valence for the chain using this edge.
        """
        decay = self._config.edge_valence_ema_decay
        count = self.edge_valence_count[edge_idx].item()
        momentum = min(decay, 1.0 - 1.0 / (count + 1))

        self.edge_valence[edge_idx] = (
            momentum * self.edge_valence[edge_idx]
            + (1.0 - momentum) * valence_scalar
        )
        self.edge_valence_count[edge_idx] += 1

    # ------------------------------------------------------------------
    # Chain valence
    # ------------------------------------------------------------------

    def compute_chain_valence(
        self, visited_basins: list[int], decay: float = 0.9,
    ) -> float:
        """Compute aggregate valence for a reasoning chain.

        Later basins (closer to the conclusion) are weighted more heavily
        via exponential recency weighting.

        Args:
            visited_basins: List of basin IDs traversed by the chain.
            decay: Recency decay factor (0 < decay < 1).

        Returns:
            Aggregate chain valence [-1, 1].
        """
        if not visited_basins:
            return 0.0

        n = len(visited_basins)
        # Exponential recency: most recent basin gets weight 1.0
        weights = [decay ** (n - 1 - i) for i in range(n)]
        total_weight = sum(weights)

        chain_val = 0.0
        for i, bid in enumerate(visited_basins):
            bv = self.basin_valence_scalar[bid].item()
            chain_val += (weights[i] / total_weight) * bv

        return chain_val

    # ------------------------------------------------------------------
    # Modulators for other modules
    # ------------------------------------------------------------------

    def get_surprise_threshold_modifier(self, signal: ValenceSignal) -> float:
        """Compute surprise threshold modifier from valence intensity.

        High-|valence| events lower the surprise threshold, making it
        easier to store emotionally significant episodes.

        Args:
            signal: ValenceSignal from compute_valence.

        Returns:
            Multiplier for surprise threshold (< 1.0 = lower threshold).
            Range: [0.7, 1.0].
        """
        return 1.0 - 0.3 * abs(signal.scalar_valence)

    def get_retention_priority_modifier(self, signal: ValenceSignal) -> float:
        """Compute retention priority modifier from valence intensity.

        High-|valence| episodes get up to 2x retention priority.

        Args:
            signal: ValenceSignal from compute_valence.

        Returns:
            Multiplier for retention priority (>= 1.0).
            Range: [1.0, 2.0].
        """
        return 1.0 + abs(signal.scalar_valence)

    def get_reasoning_mode(self, signal: ValenceSignal) -> str:
        """Determine reasoning mode from composite mood.

        Args:
            signal: ValenceSignal from compute_valence.

        Returns:
            ``"approach"`` (mood > 0.2), ``"avoidance"`` (mood < -0.2),
            or ``"neutral"``.
        """
        if signal.composite_mood > 0.2:
            return "approach"
        elif signal.composite_mood < -0.2:
            return "avoidance"
        return "neutral"

    def get_reasoning_valence_bias(
        self, mode: str, target_basins: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-basin valence bias for reasoning chain scoring.

        In approach mode, positive-valence basins get a boost.
        In avoidance mode, negative-valence basins get a penalty.
        Safety asymmetry: avoidance weight (0.4) is 2x approach weight (0.2).

        Args:
            mode: Reasoning mode (``"approach"``, ``"avoidance"``, ``"neutral"``).
            target_basins: (N,) int tensor of target basin indices.

        Returns:
            (N,) tensor of additive bias values.
        """
        if mode == "neutral" or len(target_basins) == 0:
            return torch.zeros(len(target_basins), device=target_basins.device)

        cfg = self._config
        target_valences = self.basin_valence_scalar[target_basins]

        if mode == "approach":
            # Boost positive-valence targets only
            bias = cfg.approach_weight * target_valences.clamp(min=0.0)
        elif mode == "avoidance":
            # Penalize negative-valence targets only (2x stronger)
            bias = cfg.avoidance_weight * target_valences.clamp(max=0.0)
        else:
            bias = torch.zeros_like(target_valences)

        return bias

    def get_consolidation_sampling_weights(
        self, basin_ids: list[int],
    ) -> torch.Tensor:
        """Compute valence-based sampling weights for consolidation.

        High-|valence| basins are sampled more often during consolidation
        replay, mirroring biological preferential consolidation of
        emotional memories.

        Args:
            basin_ids: List of basin IDs to compute weights for.

        Returns:
            (N,) tensor of normalized sampling weights.
        """
        if not basin_ids:
            return torch.tensor([], device=self.basin_valence_scalar.device)

        weights = []
        for bid in basin_ids:
            # 1 + |valence| per basin: range [1.0, 2.0]
            w = 1.0 + abs(self.basin_valence_scalar[bid].item())
            weights.append(w)

        w_tensor = torch.tensor(
            weights, device=self.basin_valence_scalar.device,
        )
        return w_tensor / w_tensor.sum()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def mood(self) -> dict[str, float]:
        """Return current mood state as a dict."""
        cfg = self._config
        w = cfg.mood_weights
        composite = (
            w[0] * self._mood_immediate
            + w[1] * self._mood_short_term
            + w[2] * self._mood_baseline
        )
        return {
            "immediate": self._mood_immediate,
            "short_term": self._mood_short_term,
            "baseline": self._mood_baseline,
            "composite": max(-1.0, min(1.0, composite)),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_rl(self, value: float) -> float:
        """Normalize RL reward to ~[-1, 1] via running mean/variance.

        Uses a soft tanh clamp to prevent extreme values.

        Args:
            value: Raw RL reward.

        Returns:
            Normalized reward in approximately [-1, 1].
        """
        mom = self._rl_momentum
        self._rl_mean = mom * self._rl_mean + (1.0 - mom) * value
        self._rl_var = (
            mom * self._rl_var
            + (1.0 - mom) * (value - self._rl_mean) ** 2
        )
        std = max(self._rl_var ** 0.5, 1e-8)
        normalized = (value - self._rl_mean) / std
        return math.tanh(normalized)
