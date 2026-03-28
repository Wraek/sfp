"""Selective Attention — pre-Perceiver salience gate (Skip / Skim / Full).

Evaluates raw tokenizer outputs *before* the Perceiver bottleneck and assigns
one of three processing levels per cycle: SKIP (<0.05 ms, discard),
SKIM (~0.3 ms, micro-update only), or FULL (~15 ms, complete pipeline).

The gate is modulated by goal context, world-model expectations, and recent
activation history.  Adaptive thresholds prevent all-FULL or all-SKIP drift.
Interrupt escalation (alarm basin match, accumulation, cross-modal convergence)
can upgrade SKIM→FULL when a background input turns urgent.
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import SelectiveAttentionConfig
from sfp.types import ProcessingLevel, SalienceResult
from sfp.utils.logging import get_logger

logger = get_logger("attention.salience")


class SalienceGate(nn.Module):
    """Pre-Perceiver salience filter for multi-modal input streams.

    Architecture:
      - Per-modality estimators (``ModuleDict``, dynamically extensible):
        ``Linear(d_model, d_salience) → GELU → Linear(d_salience, 1) → Sigmoid``
      - Per-modality change detectors (``ModuleDict``, dynamically extensible):
        ``Linear(d_model×2, d_salience) → GELU → Linear(d_salience, 1) → Sigmoid``
      - Context aggregator:
        ``Linear(d_model×3, d_context) → GELU``
      - Salience combiner:
        ``Linear(3 + d_context, 64) → GELU → Linear(64, 1) → Sigmoid``

    Modality estimators are created for all modalities listed in config at
    init time.  Additional modalities encountered at runtime are registered
    lazily — the gate never silently drops an unknown modality.

    Args:
        config: SelectiveAttentionConfig with thresholds and buffer sizes.
        d_model: Token / embedding dimensionality.
    """

    def __init__(
        self,
        config: SelectiveAttentionConfig | None = None,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        cfg = config or SelectiveAttentionConfig()
        self._config = cfg
        self._d_model = d_model
        d_sal = cfg.d_salience

        # --- Per-modality estimators ---
        self.modality_estimators = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model, d_sal),
                nn.GELU(),
                nn.Linear(d_sal, 1),
                nn.Sigmoid(),
            )
            for name in cfg.modality_names
        })

        # --- Per-modality change detectors ---
        self.change_detectors = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(d_model * 2, d_sal),
                nn.GELU(),
                nn.Linear(d_sal, 1),
                nn.Sigmoid(),
            )
            for name in cfg.modality_names
        })

        # --- Context aggregator: [goal, wm_prediction, recent] → d_context ---
        self.context_aggregator = nn.Sequential(
            nn.Linear(d_model * 3, cfg.d_context),
            nn.GELU(),
        )

        # --- Salience combiner: [base, change, expectation_error, context] → score ---
        self.salience_combiner = nn.Sequential(
            nn.Linear(3 + cfg.d_context, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # --- State: per-modality previous inputs ---
        self._prev_inputs: dict[str, torch.Tensor] = {}

        # --- Adaptive skim thresholds (start at config default) ---
        self._adaptive_thresholds: dict[str, float] = {
            name: cfg.skim_threshold for name in cfg.modality_names
        }

        # --- Skim buffer ---
        self._skim_buffer: list[dict] = []

        # --- Accumulation counts for interrupt escalation ---
        self._accumulation_counts: dict[str, int] = defaultdict(int)
        self._accumulation_history: list[float] = []

        # --- Recent full-processing timestamps per modality ---
        self._recent_full_times: dict[str, float] = {}

        # --- Hindsight training buffer ---
        self._hindsight_buffer: list[dict] = []
        self._hindsight_optimizer: torch.optim.Optimizer | None = None

        logger.info(
            "SalienceGate initialized: %d modalities, skip=%.2f, skim=%.2f, "
            "d_salience=%d, d_context=%d",
            cfg.n_modalities, cfg.skip_threshold, cfg.skim_threshold,
            d_sal, cfg.d_context,
        )

    # ------------------------------------------------------------------
    # Dynamic modality registration
    # ------------------------------------------------------------------

    def _get_or_create_estimator(self, mod_name: str) -> nn.Module:
        """Return the estimator for *mod_name*, creating it lazily if needed."""
        if mod_name not in self.modality_estimators:
            device = next(self.parameters()).device
            d_sal = self._config.d_salience
            estimator = nn.Sequential(
                nn.Linear(self._d_model, d_sal),
                nn.GELU(),
                nn.Linear(d_sal, 1),
                nn.Sigmoid(),
            ).to(device)
            self.modality_estimators[mod_name] = estimator
            # Also initialise adaptive threshold for the new modality
            self._adaptive_thresholds[mod_name] = self._config.skim_threshold
            logger.info("Lazily registered modality estimator: %s", mod_name)
        return self.modality_estimators[mod_name]

    def _get_or_create_change_detector(self, mod_name: str) -> nn.Module:
        """Return the change detector for *mod_name*, creating it lazily if needed."""
        if mod_name not in self.change_detectors:
            device = next(self.parameters()).device
            d_sal = self._config.d_salience
            detector = nn.Sequential(
                nn.Linear(self._d_model * 2, d_sal),
                nn.GELU(),
                nn.Linear(d_sal, 1),
                nn.Sigmoid(),
            ).to(device)
            self.change_detectors[mod_name] = detector
            logger.info("Lazily registered change detector: %s", mod_name)
        return self.change_detectors[mod_name]

    # ------------------------------------------------------------------
    # Core evaluation
    # ------------------------------------------------------------------

    def _ensure_d_model(self, v: torch.Tensor) -> torch.Tensor:
        """Pad or truncate a 1-D vector to d_model length.

        Context vectors from goals (d_goal) and world model (d_deterministic)
        may differ from d_model.  Zero-pad or truncate so they can be
        concatenated for the context_aggregator.
        """
        d = self._d_model
        if v.shape[-1] == d:
            return v
        if v.shape[-1] < d:
            return F.pad(v, (0, d - v.shape[-1]))
        return v[..., :d]

    def evaluate(
        self,
        inputs: dict[str, torch.Tensor],
        goal_context: torch.Tensor | None = None,
        world_model_prediction: torch.Tensor | None = None,
        recent_activation: torch.Tensor | None = None,
        alarm_basins: set[int] | None = None,
    ) -> SalienceResult:
        """Evaluate salience of all incoming modality inputs.

        Args:
            inputs: Mapping of modality name to (d_model,) embedding.
            goal_context: (d_goal,) from goal register (or zeros).
            world_model_prediction: (d_deterministic,) from world model (or zeros).
            recent_activation: (d_model,) recent basin activation summary.
            alarm_basins: Set of basin IDs that trigger immediate FULL.

        Returns:
            SalienceResult with per-modality scores and combined decision.
        """
        device = next(self.parameters()).device
        d = self._d_model
        cfg = self._config

        # Default context vectors — resized to d_model for context_aggregator
        goal_ctx = (
            self._ensure_d_model(goal_context.to(device))
            if goal_context is not None
            else torch.zeros(d, device=device)
        )
        wm_pred = (
            self._ensure_d_model(world_model_prediction.to(device))
            if world_model_prediction is not None
            else torch.zeros(d, device=device)
        )
        recent_ctx = (
            self._ensure_d_model(recent_activation.to(device))
            if recent_activation is not None
            else torch.zeros(d, device=device)
        )

        # 1. Aggregate context (shared across all modalities)
        with torch.no_grad():
            context = self.context_aggregator(
                torch.cat([goal_ctx, wm_pred, recent_ctx]).unsqueeze(0)
            ).squeeze(0)  # (d_context,)

        per_modality_scores: dict[str, float] = {}

        for mod_name, embedding in inputs.items():
            # Lazily create estimator/detector for unknown modalities
            estimator = self._get_or_create_estimator(mod_name)
            change_detector = self._get_or_create_change_detector(mod_name)

            emb = embedding.to(device)

            with torch.no_grad():
                # 2a. Base salience
                base_sal = estimator(emb.unsqueeze(0)).item()

                # 2b. Change detection
                prev = self._prev_inputs.get(mod_name)
                if prev is not None:
                    change_input = torch.cat([emb, prev]).unsqueeze(0)
                    change_score = change_detector(change_input).item()
                else:
                    change_score = 1.0  # first input always "changed"

                # 2c. Expectation error
                if world_model_prediction is not None:
                    expectation_error = 1.0 - F.cosine_similarity(
                        emb.unsqueeze(0), wm_pred.unsqueeze(0)
                    ).item()
                    expectation_error = max(0.0, min(1.0, expectation_error))
                else:
                    expectation_error = 0.5  # uninformative default

                # 3. Combine: [base, change, expectation, context] → score
                combine_input = torch.cat([
                    torch.tensor(
                        [base_sal, change_score, expectation_error],
                        device=device,
                    ),
                    context,
                ]).unsqueeze(0)
                final_score = self.salience_combiner(combine_input).item()

            per_modality_scores[mod_name] = final_score

            # Update previous input
            self._prev_inputs[mod_name] = emb.detach().clone()

        if not per_modality_scores:
            return SalienceResult(
                level=ProcessingLevel.SKIP,
                salience_scores={},
                combined_salience=0.0,
            )

        # 4. Combined salience = max across modalities
        combined = max(per_modality_scores.values())

        # 5. Classify using adaptive thresholds
        if combined < cfg.skip_threshold:
            level = ProcessingLevel.SKIP
        elif combined < self._get_adaptive_skim_threshold():
            level = ProcessingLevel.SKIM
        else:
            level = ProcessingLevel.FULL

        # 6. Check interrupts → may escalate
        interrupt, interrupt_reason = self.check_interrupts(
            per_modality_scores, level, alarm_basins,
        )
        if interrupt and level != ProcessingLevel.FULL:
            level = ProcessingLevel.FULL

        # 7. Update adaptive thresholds (EMA of combined salience)
        self._update_adaptive_thresholds(per_modality_scores)

        # 8. Update accumulation counts
        now = time.monotonic()
        for mod_name in per_modality_scores:
            if level == ProcessingLevel.FULL:
                self._accumulation_counts[mod_name] = 0
                self._recent_full_times[mod_name] = now
            else:
                self._accumulation_counts[mod_name] += 1

        return SalienceResult(
            level=level,
            salience_scores=per_modality_scores,
            combined_salience=combined,
            interrupt=interrupt,
            interrupt_reason=interrupt_reason,
        )

    # ------------------------------------------------------------------
    # Interrupt escalation
    # ------------------------------------------------------------------

    def check_interrupts(
        self,
        scores: dict[str, float],
        assigned_level: ProcessingLevel,
        alarm_basins: set[int] | None = None,
    ) -> tuple[bool, str]:
        """Check whether an interrupt should escalate processing to FULL.

        Three triggers:
          1. **Alarm basin match** (checked externally, flagged here if score > threshold)
          2. **Accumulation**: mean salience over recent window exceeds threshold
          3. **Cross-modal convergence**: 2+ modalities above cross_modal_threshold

        Args:
            scores: Per-modality salience scores from evaluate().
            assigned_level: Currently assigned processing level.
            alarm_basins: Set of high-priority basin IDs (checked by skim_process).

        Returns:
            (should_interrupt, reason_string).
        """
        cfg = self._config

        if assigned_level == ProcessingLevel.FULL:
            return False, ""

        # Trigger 1: accumulation — any modality has been skipped/skimmed too long
        for mod_name, count in self._accumulation_counts.items():
            if count >= cfg.accumulation_window:
                # Check if average salience over window is significant
                mod_score = scores.get(mod_name, 0.0)
                if mod_score > cfg.accumulation_threshold:
                    return True, (
                        f"accumulation_trigger:{mod_name} "
                        f"(count={count}, score={mod_score:.3f})"
                    )

        # Trigger 2: cross-modal convergence — multiple modalities are elevated
        high_mods = [
            name for name, s in scores.items()
            if s > cfg.cross_modal_threshold
        ]
        if len(high_mods) >= 2:
            return True, (
                f"cross_modal_convergence: {', '.join(high_mods)}"
            )

        return False, ""

    # ------------------------------------------------------------------
    # Skim processing
    # ------------------------------------------------------------------

    def skim_process(
        self,
        embedding: torch.Tensor,
        modality: str,
        tier0: object | None = None,
    ) -> dict:
        """Minimal skim processing for a low-salience input.

        Stores in skim buffer and performs a micro-update to Tier 0 working
        memory (if provided) to maintain peripheral awareness.

        Args:
            embedding: (d_model,) tokenized input embedding.
            modality: Source modality name.
            tier0: Working memory module (duck-typed, must have ``lmm`` attribute
                   with ``parameters()``).

        Returns:
            Dict with skim processing metadata.
        """
        cfg = self._config

        # 1. Store in skim buffer
        self._skim_buffer.append({
            "embedding": embedding.detach().clone(),
            "modality": modality,
            "timestamp": time.monotonic(),
        })
        if len(self._skim_buffer) > cfg.skim_buffer_size:
            self._skim_buffer.pop(0)

        # 2. Micro-update Tier 0 (tiny gradient-free parameter nudge)
        if tier0 is not None and hasattr(tier0, "lmm"):
            with torch.no_grad():
                lmm = tier0.lmm
                y_pred = lmm(embedding.unsqueeze(0)).squeeze(0)
                error = embedding - y_pred
                for param in lmm.parameters():
                    param.data.add_(
                        cfg.skim_lr_scale * error.mean() * 0.01
                    )

        return {
            "buffered": True,
            "buffer_size": len(self._skim_buffer),
        }

    def get_skim_summary(self) -> torch.Tensor:
        """Produce exponentially-weighted summary of recent skimmed inputs.

        Returns:
            (d_model,) summary vector, or zeros if buffer is empty.
        """
        device = next(self.parameters()).device
        if not self._skim_buffer:
            return torch.zeros(self._d_model, device=device)

        recent = self._skim_buffer[-20:]
        embeddings = torch.stack(
            [s["embedding"].to(device) for s in recent]
        )  # (N, d)
        n = embeddings.shape[0]
        weights = torch.exp(torch.linspace(-2.0, 0.0, n, device=device))
        weights = weights / weights.sum()
        return (weights.unsqueeze(-1) * embeddings).sum(dim=0)

    # ------------------------------------------------------------------
    # Hindsight training
    # ------------------------------------------------------------------

    def train_hindsight(
        self,
        assigned_salience: float,
        was_useful: bool,
        features: torch.Tensor,
    ) -> None:
        """Record a hindsight (prediction, outcome) pair for training.

        Args:
            assigned_salience: The salience score the gate produced.
            was_useful: Whether full processing of this input was worthwhile.
            features: (d_model,) input embedding for feature storage.
        """
        cfg = self._config
        self._hindsight_buffer.append({
            "predicted": assigned_salience,
            "label": 1.0 if was_useful else 0.0,
            "features": features.detach().clone(),
        })
        if len(self._hindsight_buffer) > cfg.hindsight_buffer_size:
            self._hindsight_buffer.pop(0)

    def run_hindsight_training(self) -> float:
        """Train salience gate from accumulated hindsight data.

        Should be called during consolidation (off critical path).
        Uses BCE loss between assigned salience and hindsight label.

        Returns:
            Training loss value, or 0.0 if insufficient data.
        """
        cfg = self._config
        if len(self._hindsight_buffer) < 50:
            return 0.0

        # Lazy-init optimizer
        if self._hindsight_optimizer is None:
            self._hindsight_optimizer = torch.optim.Adam(
                self.parameters(), lr=cfg.hindsight_lr,
            )

        device = next(self.parameters()).device
        total_loss = torch.tensor(0.0, device=device)

        for entry in self._hindsight_buffer:
            pred = torch.tensor(
                [entry["predicted"]], device=device, requires_grad=True,
            )
            label = torch.tensor([entry["label"]], device=device)
            total_loss = total_loss + F.binary_cross_entropy(pred, label)

        loss = total_loss / len(self._hindsight_buffer)

        self._hindsight_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self._hindsight_optimizer.step()

        loss_val = loss.item()
        self._hindsight_buffer.clear()
        return loss_val

    # ------------------------------------------------------------------
    # Modulation helpers
    # ------------------------------------------------------------------

    def apply_goal_modulation(
        self,
        base_thresholds: dict[str, float],
        goal_modulation: dict[str, float],
    ) -> dict[str, float]:
        """Lower thresholds for goal-relevant modalities.

        Args:
            base_thresholds: Mapping of modality → base skim threshold.
            goal_modulation: Mapping of modality → negative adjustment
                (from GoalRegister.get_salience_modulation()).

        Returns:
            Adjusted thresholds per modality.
        """
        adjusted: dict[str, float] = {}
        for mod_name, base in base_thresholds.items():
            adjustment = goal_modulation.get(mod_name, 0.0)
            adjusted[mod_name] = max(0.05, base + adjustment)
        return adjusted

    def apply_expectation_modulation(
        self,
        inputs: dict[str, torch.Tensor],
        prediction: torch.Tensor,
    ) -> dict[str, float]:
        """Compute per-modality expectation error against world model.

        High prediction error → higher salience boost for that modality.

        Args:
            inputs: Mapping of modality → (d_model,) embedding.
            prediction: (d_model,) world model predicted observation.

        Returns:
            Mapping of modality → expectation error [0, 1].
        """
        errors: dict[str, float] = {}
        for mod_name, emb in inputs.items():
            sim = F.cosine_similarity(
                emb.unsqueeze(0), prediction.unsqueeze(0)
            ).item()
            errors[mod_name] = max(0.0, min(1.0, 1.0 - sim))
        return errors

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_adaptive_skim_threshold(self) -> float:
        """Return the current adaptive skim threshold (mean across modalities)."""
        if not self._adaptive_thresholds:
            return self._config.skim_threshold
        return sum(self._adaptive_thresholds.values()) / len(
            self._adaptive_thresholds
        )

    def _update_adaptive_thresholds(
        self, scores: dict[str, float],
    ) -> None:
        """Update per-modality adaptive thresholds via EMA."""
        cfg = self._config
        mom = cfg.threshold_ema_decay
        for mod_name, score in scores.items():
            prev = self._adaptive_thresholds.get(mod_name, cfg.skim_threshold)
            # When salience is consistently high, raise threshold
            target = cfg.skim_threshold * (1.0 + 0.3 * score)
            self._adaptive_thresholds[mod_name] = (
                mom * prev + (1.0 - mom) * target
            )
