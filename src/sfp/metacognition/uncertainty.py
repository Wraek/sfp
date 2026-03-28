"""Metacognition & Uncertainty — 4-source uncertainty estimation and calibration.

Collects uncertainty signals from retrieval, reasoning chain, world-model
prediction, and knowledge maturity.  Produces a structured UncertaintyEstimate
with scalar confidence [0, 1] and a compositional embedding encoding *what*
the system is uncertain about.  Tracks calibration via Expected Calibration
Error (ECE) and monitors memory health for dormant, declining, and
coverage-gap basins.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import MetacognitionConfig
from sfp.types import ChainTrace, UncertaintyEstimate, WorldModelState
from sfp.utils.logging import get_logger

logger = get_logger("metacognition.uncertainty")

_MAX_HOPS = 7  # normalizer for chain length (ReasoningChainConfig default)


class MetacognitionEngine(nn.Module):
    """4-source uncertainty estimation, calibration, and memory health.

    Estimates uncertainty from four orthogonal sources:

      1. **Retrieval** — entropy of attention weights over Tier 2 keys
      2. **Chain** — cumulative confidence product across reasoning hops
      3. **Prediction** — EMA-normalized world-model error signals
      4. **Knowledge** — basin maturity, modality coverage, confidence

    The four scalar uncertainties are composed with a query context vector
    into a structured uncertainty embedding (``d_uncertainty_embedding``)
    and a scalar confidence score suitable for simple thresholding.

    Also provides active information-seeking suggestions when confidence is
    low and ongoing memory-health monitoring (dormant basins, declining
    confidence, coverage gaps).

    Args:
        config: MetacognitionConfig with embedding sizes and thresholds.
        d_model: Backbone / field dimensionality.
    """

    def __init__(
        self,
        config: MetacognitionConfig | None = None,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        cfg = config or MetacognitionConfig()
        self._config = cfg
        self._d_model = d_model
        d_unc = cfg.d_uncertainty_embedding
        h = cfg.estimator_hidden

        # --- 4 source estimators: 3-input → hidden → 1 → Sigmoid ---
        self.retrieval_estimator = nn.Sequential(
            nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid(),
        )
        self.chain_estimator = nn.Sequential(
            nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid(),
        )
        self.prediction_estimator = nn.Sequential(
            nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid(),
        )
        self.knowledge_estimator = nn.Sequential(
            nn.Linear(3, h), nn.GELU(), nn.Linear(h, 1), nn.Sigmoid(),
        )

        # --- Uncertainty composer: [4 scores + context] → 128 → d_unc ---
        self.uncertainty_composer = nn.Sequential(
            nn.Linear(4 + d_model, 128),
            nn.GELU(),
            nn.Linear(128, d_unc),
        )

        # --- Scalar confidence head: d_unc → 32 → 1 → Sigmoid ---
        self.scalar_head = nn.Sequential(
            nn.Linear(d_unc, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # --- Calibration bins ---
        n_bins = cfg.n_calibration_bins
        self.register_buffer("calibration_counts", torch.zeros(n_bins))
        self.register_buffer("calibration_correct", torch.zeros(n_bins))

        # --- Running EMAs for prediction uncertainty normalization ---
        self._pred_error_ema: float = 1.0
        self._kl_ema: float = 1.0
        self._recon_ema: float = 1.0
        self._ema_momentum: float = 0.99

        # --- Memory health tracking (Python-side, not persisted) ---
        self._activation_counts: dict[int, int] = defaultdict(int)
        self._confidence_history: dict[int, list[float]] = defaultdict(list)

        logger.info(
            "MetacognitionEngine initialized: d_unc=%d, estimator_hidden=%d, "
            "calibration_bins=%d",
            d_unc, h, n_bins,
        )

    # ------------------------------------------------------------------
    # Per-source uncertainty estimators
    # ------------------------------------------------------------------

    def estimate_retrieval_uncertainty(
        self,
        attn_weights: torch.Tensor | None,
        basin_confidence: float,
        n_active: int,
    ) -> float:
        """Estimate retrieval uncertainty from attention distribution.

        Features: [normalized_entropy, 1 - basin_confidence, 1 / (n_active + 1)]

        Args:
            attn_weights: (N,) attention weights over Tier 2 basins,
                or ``None`` if unavailable (defaults to maximum entropy).
            basin_confidence: Confidence of the best-matching basin [0, 1].
            n_active: Number of active basins in Tier 2.

        Returns:
            Scalar uncertainty in [0, 1].
        """
        device = next(self.parameters()).device

        # Normalized entropy of attention distribution
        if attn_weights is not None:
            probs = attn_weights.float().clamp(min=1e-8)
            probs = probs / probs.sum()
            entropy = -(probs * probs.log()).sum().item()
            max_entropy = math.log(max(len(probs), 2))
            norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            # No attention weights available — assume maximum entropy
            norm_entropy = 1.0

        features = torch.tensor(
            [[norm_entropy, 1.0 - basin_confidence, 1.0 / (n_active + 1)]],
            device=device,
        )
        with torch.no_grad():
            return self.retrieval_estimator(features).item()

    def estimate_chain_uncertainty(
        self, chain_trace: list[ChainTrace],
    ) -> float:
        """Estimate reasoning-chain uncertainty from hop metadata.

        Uses a multiplicative confidence model: cumulative confidence is
        the product of per-hop ``sqrt(basin_conf * edge_conf)``.  A single
        weak link dramatically reduces overall confidence.

        Features: [1 - cumulative, n_hops / max_hops, 1 - min_hop_confidence]

        Args:
            chain_trace: List of ChainTrace entries from a reasoning chain.

        Returns:
            Scalar uncertainty in [0, 1].
        """
        device = next(self.parameters()).device

        if not chain_trace:
            features = torch.tensor([[1.0, 0.0, 1.0]], device=device)
            with torch.no_grad():
                return self.chain_estimator(features).item()

        cumulative = 1.0
        min_conf = 1.0
        n_hops = 0

        for i, hop in enumerate(chain_trace):
            if hop.event_type == "start":
                continue
            basin_conf = max(hop.confidence, 1e-8)
            edge_conf = max(hop.score, 1e-8) if i > 0 else 1.0
            hop_conf = math.sqrt(basin_conf * edge_conf)
            cumulative *= hop_conf
            min_conf = min(min_conf, hop_conf)
            n_hops += 1

        features = torch.tensor(
            [[1.0 - cumulative, n_hops / _MAX_HOPS, 1.0 - min_conf]],
            device=device,
        )
        with torch.no_grad():
            return self.chain_estimator(features).item()

    def estimate_prediction_uncertainty(
        self, wm_state: WorldModelState,
    ) -> float:
        """Estimate prediction uncertainty from world-model error signals.

        Each signal is normalized by its running EMA to produce a
        dimensionless ratio (>1 means worse than average).

        Features: [norm_kl, norm_pred_error, norm_recon_error]

        Args:
            wm_state: WorldModelState from the most recent world-model step.

        Returns:
            Scalar uncertainty in [0, 1].
        """
        device = next(self.parameters()).device
        mom = self._ema_momentum

        # Update running EMAs
        pe = max(wm_state.prediction_error, 1e-8)
        kl = max(wm_state.kl_divergence, 1e-8)
        re = max(wm_state.reconstruction_error, 1e-8)

        self._pred_error_ema = mom * self._pred_error_ema + (1 - mom) * pe
        self._kl_ema = mom * self._kl_ema + (1 - mom) * kl
        self._recon_ema = mom * self._recon_ema + (1 - mom) * re

        # Normalize by EMA
        norm_pe = pe / max(self._pred_error_ema, 1e-8)
        norm_kl = kl / max(self._kl_ema, 1e-8)
        norm_re = re / max(self._recon_ema, 1e-8)

        features = torch.tensor(
            [[norm_kl, norm_pe, norm_re]],
            device=device,
        )
        with torch.no_grad():
            return self.prediction_estimator(features).item()

    def estimate_knowledge_uncertainty(
        self,
        confidence: float,
        maturity: float,
        modality_coverage: float,
    ) -> float:
        """Estimate knowledge uncertainty from basin metadata.

        Features: [1 - maturity, 1 - modality_coverage, 1 - confidence]

        Args:
            confidence: Basin confidence [0, 1].
            maturity: Basin maturity [0, 1] (e.g. episode_count / 1000, clamped).
            modality_coverage: Fraction of modalities with episodes [0, 1].

        Returns:
            Scalar uncertainty in [0, 1].
        """
        device = next(self.parameters()).device
        features = torch.tensor(
            [[1.0 - maturity, 1.0 - modality_coverage, 1.0 - confidence]],
            device=device,
        )
        with torch.no_grad():
            return self.knowledge_estimator(features).item()

    # ------------------------------------------------------------------
    # Composition
    # ------------------------------------------------------------------

    def compose_uncertainty(
        self,
        retrieval: float,
        chain: float,
        prediction: float,
        knowledge: float,
        context: torch.Tensor,
    ) -> UncertaintyEstimate:
        """Compose 4 source scores + context into a structured estimate.

        Args:
            retrieval: Retrieval uncertainty [0, 1].
            chain: Chain uncertainty [0, 1].
            prediction: Prediction uncertainty [0, 1].
            knowledge: Knowledge uncertainty [0, 1].
            context: (d_model,) query or state context vector.

        Returns:
            UncertaintyEstimate with embedding, scalar confidence, and
            per-source breakdowns.
        """
        device = context.device

        source_scores = torch.tensor(
            [retrieval, chain, prediction, knowledge],
            device=device,
        )
        composer_input = torch.cat([source_scores, context]).unsqueeze(0)

        with torch.no_grad():
            embedding = self.uncertainty_composer(composer_input).squeeze(0)
            scalar_confidence = self.scalar_head(embedding.unsqueeze(0)).item()

        # scalar_confidence: Sigmoid ∈ [0, 1], higher = more confident
        calibrated = self.calibration_counts.sum().item() >= 100

        return UncertaintyEstimate(
            retrieval_uncertainty=retrieval,
            chain_uncertainty=chain,
            prediction_uncertainty=prediction,
            knowledge_uncertainty=knowledge,
            composite_embedding=embedding.detach(),
            scalar_confidence=scalar_confidence,
            calibrated=calibrated,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def update_calibration(
        self, predicted_confidence: float, was_correct: bool,
    ) -> None:
        """Update calibration tracking with a (confidence, outcome) pair.

        Called after ground truth becomes available (user correction,
        prediction verification, consolidation replay, self-consistency).

        Args:
            predicted_confidence: Scalar confidence [0, 1] from compose.
            was_correct: Whether the system's output was correct.
        """
        n_bins = self._config.n_calibration_bins
        bin_idx = min(int(predicted_confidence * n_bins), n_bins - 1)
        self.calibration_counts[bin_idx] += 1
        if was_correct:
            self.calibration_correct[bin_idx] += 1

    def get_ece(self) -> float:
        """Compute Expected Calibration Error.

        For each bin of predicted confidence values, the average predicted
        confidence should match the average actual accuracy.  ECE is the
        weighted mean of absolute differences.

        Returns:
            ECE value (lower is better calibrated).  0.0 if insufficient data.
        """
        n_bins = self._config.n_calibration_bins
        total = self.calibration_counts.sum().item()
        if total < 10:
            return 0.0

        ece = 0.0
        for i in range(n_bins):
            count = self.calibration_counts[i].item()
            if count == 0:
                continue
            accuracy = self.calibration_correct[i].item() / count
            midpoint = (i + 0.5) / n_bins
            weight = count / total
            ece += weight * abs(accuracy - midpoint)
        return ece

    def get_calibration_report(self) -> dict:
        """Return per-bin calibration data and overall ECE.

        Returns:
            Dict with ``ece``, ``total_samples``, and per-``bins`` breakdown.
        """
        n_bins = self._config.n_calibration_bins
        report: dict = {
            "n_bins": n_bins,
            "ece": self.get_ece(),
            "total_samples": self.calibration_counts.sum().item(),
            "bins": [],
        }
        for i in range(n_bins):
            count = self.calibration_counts[i].item()
            correct = self.calibration_correct[i].item()
            report["bins"].append({
                "range": (i / n_bins, (i + 1) / n_bins),
                "count": count,
                "accuracy": correct / count if count > 0 else 0.0,
                "expected_accuracy": (i + 0.5) / n_bins,
            })
        return report

    # ------------------------------------------------------------------
    # Active information seeking
    # ------------------------------------------------------------------

    def suggest_information_seeking(
        self,
        estimate: UncertaintyEstimate,
        basin_keys: torch.Tensor | None = None,
        basin_confidence: torch.Tensor | None = None,
        n_active: int = 0,
    ) -> list[dict]:
        """Suggest strategies to reduce uncertainty.

        Examines the dominant uncertainty source and proposes up to
        ``seeking_max_alternatives`` information-seeking actions.

        Args:
            estimate: UncertaintyEstimate from compose_uncertainty.
            basin_keys: (n_active, d_model) Tier 2 basin keys (optional).
            basin_confidence: (n_active,) basin confidence scores (optional).
            n_active: Number of active basins.

        Returns:
            List of suggestion dicts with strategy, source, reason, and action.
        """
        cfg = self._config
        if estimate.scalar_confidence >= cfg.confidence_threshold_high:
            return []  # confident enough — no seeking needed

        suggestions: list[dict] = []

        # Rank sources by uncertainty (highest first)
        sources = {
            "retrieval": estimate.retrieval_uncertainty,
            "chain": estimate.chain_uncertainty,
            "prediction": estimate.prediction_uncertainty,
            "knowledge": estimate.knowledge_uncertainty,
        }
        ranked = sorted(sources.items(), key=lambda x: x[1], reverse=True)

        for source_name, source_val in ranked[:cfg.seeking_max_alternatives]:
            if source_val < 0.3:
                continue  # not significant enough

            if source_name == "knowledge":
                suggestion: dict = {
                    "strategy": "alternative_basins",
                    "source": "knowledge",
                    "uncertainty": source_val,
                    "reason": "Low knowledge maturity in relevant basins",
                    "action": "seek_alternative_basins",
                }
                # Count high-confidence alternatives if data available
                if (
                    basin_keys is not None
                    and basin_confidence is not None
                    and n_active > 1
                ):
                    n_alts = int(
                        (basin_confidence[:n_active] > 0.7).sum().item()
                    )
                    suggestion["n_high_confidence_alternatives"] = n_alts
                suggestions.append(suggestion)

            elif source_name == "chain":
                suggestions.append({
                    "strategy": "alternative_chains",
                    "source": "chain",
                    "uncertainty": source_val,
                    "reason": (
                        "Weak reasoning chain "
                        "(low-confidence hops or edges)"
                    ),
                    "action": "retry_with_alternative_path",
                })

            elif source_name == "retrieval":
                suggestions.append({
                    "strategy": "query_refinement",
                    "source": "retrieval",
                    "uncertainty": source_val,
                    "reason": "Poor match between query and known concepts",
                    "action": "decompose_or_refine_query",
                })

            elif source_name == "prediction":
                suggestions.append({
                    "strategy": "temporal_context",
                    "source": "prediction",
                    "uncertainty": source_val,
                    "reason": (
                        "High prediction error suggests "
                        "unexpected situation"
                    ),
                    "action": "increase_temporal_attention",
                })

        return suggestions

    # ------------------------------------------------------------------
    # Memory health monitoring
    # ------------------------------------------------------------------

    def record_activation(
        self,
        basin_ids: list[int],
        confidence_values: list[float] | None = None,
    ) -> None:
        """Record basin activations for health monitoring.

        Should be called after each inference to track which basins are
        being used and how their confidence evolves.

        Args:
            basin_ids: List of basin IDs activated during inference.
            confidence_values: Optional per-basin confidence values.
        """
        for i, bid in enumerate(basin_ids):
            self._activation_counts[bid] += 1
            if confidence_values is not None and i < len(confidence_values):
                hist = self._confidence_history[bid]
                hist.append(confidence_values[i])
                # Trim to avoid unbounded growth
                if len(hist) > 500:
                    self._confidence_history[bid] = hist[-250:]

    def monitor_memory_health(
        self,
        basin_keys: torch.Tensor,
        confidence: torch.Tensor,
        n_active: int,
    ) -> dict:
        """Generate a health report for the memory system.

        Detects dormant basins, declining confidence trends, coverage gaps,
        and overloaded basins.  Intended to run during consolidation (off
        the critical inference path).

        Args:
            basin_keys: (n_slots, d_model) all Tier 2 basin keys.
            confidence: (n_slots,) per-basin confidence.
            n_active: Number of active basins.

        Returns:
            Health report dict with ``dormant_basins``, ``declining_basins``,
            ``coverage_gap_fraction``, and ``overloaded_basins``.
        """
        report: dict = {
            "total_active_basins": n_active,
            "avg_confidence": 0.0,
            "dormant_basins": [],
            "declining_basins": [],
            "coverage_gap_fraction": 0.0,
            "overloaded_basins": [],
        }

        if n_active == 0:
            return report

        report["avg_confidence"] = confidence[:n_active].mean().item()

        # --- Dormant basins: activated fewer than 5 times ---
        threshold_activations = max(
            5, int(self._config.health_dormant_threshold * 100)
        )
        for bid in range(n_active):
            count = self._activation_counts.get(bid, 0)
            if count < threshold_activations:
                report["dormant_basins"].append(bid)

        # --- Declining basins: confidence monotonically decreasing ---
        for bid, hist in self._confidence_history.items():
            if bid >= n_active:
                continue
            if len(hist) < 5:
                continue
            recent = hist[-5:]
            if all(recent[j] >= recent[j + 1] for j in range(len(recent) - 1)):
                report["declining_basins"].append({
                    "basin_id": bid,
                    "current": recent[-1],
                    "decline": recent[0] - recent[-1],
                })

        # --- Coverage gap analysis: random probes in embedding space ---
        if n_active >= 10:
            device = basin_keys.device
            probes = torch.randn(100, basin_keys.shape[1], device=device)
            probes = F.normalize(probes, dim=-1)
            keys_norm = F.normalize(basin_keys[:n_active], dim=-1)
            sims = probes @ keys_norm.T  # (100, n_active)
            max_sims = sims.max(dim=-1).values
            gap_count = (max_sims < 0.3).sum().item()
            report["coverage_gap_fraction"] = gap_count / 100.0

        # --- Overloaded basins: activation count > 10× median ---
        if self._activation_counts:
            counts = [
                self._activation_counts.get(bid, 0) for bid in range(n_active)
            ]
            if counts:
                sorted_counts = sorted(counts)
                median_count = sorted_counts[len(sorted_counts) // 2]
                overload_thresh = max(50, 10 * median_count)
                for bid in range(n_active):
                    if self._activation_counts.get(bid, 0) > overload_thresh:
                        report["overloaded_basins"].append(bid)

        return report
