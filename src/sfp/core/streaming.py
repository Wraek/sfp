"""StreamingProcessor — Titans-style surprise-gated weight updates during inference."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F

from sfp.config import AttractorConfig, EWCConfig, LoRAConfig, StreamingConfig, Tier0Config
from sfp.core.attractors import AttractorQuery
from sfp.core.field import SemanticFieldProcessor
from sfp.core.forgetting import EWCStrategy
from sfp.core.lora import OnlineLoRAManager
from sfp.defense.surprise_hardening import SurpriseHardener
from sfp.types import AttractorResult, ConsistencyChecker, ForgetStrategy, SurpriseMetric
from sfp.utils.logging import get_logger
from sfp.utils.math import grad_norm

logger = get_logger("core.streaming")


class StreamingProcessor:
    """Processes streaming data via surprise-gated weight updates.

    Data transforms the manifold and moves on — never retained. Composes
    the SemanticFieldProcessor with LoRA adapters, EWC forgetting protection,
    and surprise-gated learning.

    This is the primary usage mode for the SFP framework.
    """

    def __init__(
        self,
        field: SemanticFieldProcessor,
        streaming_config: StreamingConfig | None = None,
        lora_config: LoRAConfig | None = None,
        ewc_config: EWCConfig | None = None,
        tier0_config: Tier0Config | None = None,
        consistency_checker: ConsistencyChecker | None = None,
    ) -> None:
        self.field = field
        self.config = streaming_config or StreamingConfig()

        # Tier 0 hardening
        self._tier0_config = tier0_config
        self._hardener: SurpriseHardener | None = None
        if tier0_config is not None:
            self._hardener = SurpriseHardener(tier0_config)
        self._consistency_checker = consistency_checker

        # Forgetting strategy stack
        self._forget_strategies: list[ForgetStrategy] = []
        self._lora_manager: OnlineLoRAManager | None = None
        self._ewc_strategy: EWCStrategy | None = None

        # Set up LoRA (must happen before optimizer creation)
        if lora_config is not None and lora_config.enabled:
            self._lora_manager = OnlineLoRAManager(field, lora_config)

        # Set up EWC
        if ewc_config is not None and ewc_config.enabled:
            self._ewc_strategy = EWCStrategy(field, ewc_config)
            self._forget_strategies.append(self._ewc_strategy)

        # Build optimizer over the appropriate parameter set
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()

        # State tracking
        self._surprise_ema: float = 0.0
        self._history: list[SurpriseMetric] = []
        self._step_count: int = 0

        # LoRA multi-signal merge context (set externally by orchestrator)
        self._merge_context = None

        # Consolidation notification: basin keys from last consolidation
        self._consolidated_concept_keys: list[torch.Tensor] = []
        self._accumulation_count: int = 0

        # Gradient conflict detection state
        self._grad_loss_ratio_ema: float = 0.0
        self._component_emas: dict[str, float] = {}
        self._component_directions: dict[str, list[float]] = {}
        # Gradient conflict mitigation: per-component opposition score vs primary
        self._opposition_scores: dict[str, float] = {}

        # Curriculum scheduling state
        self._competence_ema: float = 0.0
        self._curriculum_active_step: int = 0

    def process(
        self,
        x: torch.Tensor,
        target: torch.Tensor | None = None,
        latent_distance: float | None = None,
        wm_prediction: torch.Tensor | None = None,
        confidence: float | None = None,
        goal_embeddings: list[torch.Tensor] | None = None,
        tier2_guidance: torch.Tensor | None = None,
        axiom_anchor: torch.Tensor | None = None,
        salience_score: float | None = None,
        external_lr_scale: float | None = None,
    ) -> SurpriseMetric:
        """Process a single input (or batch) through the manifold.

        Args:
            x: Input tensor of shape (dim,) or (batch, dim).
            target: Optional target tensor. If None, uses autoassociative loss.
            latent_distance: Distance to nearest attractor for dual-path verification.
            wm_prediction: Detached world model predicted observation for auxiliary loss.
            confidence: [0, 1] metacognition confidence for gradient scaling.
            goal_embeddings: Goal satisfaction embeddings for regularization.
            tier2_guidance: Tier 2 basin key to guide field output toward.
            axiom_anchor: Tier 3 core knowledge to anchor field output.
            salience_score: [0, 1] salience gate score for gradient scaling.
            external_lr_scale: Combined external LR modifier (valence + urgency).

        Returns:
            SurpriseMetric with gradient norm, loss, and whether weights were updated.
        """
        self.field.train()

        # Forward pass
        y = self.field(x)

        # Loss: autoassociative (reconstruct input) or explicit target
        loss_target = target if target is not None else x
        if self.config.loss_fn == "cosine":
            primary_loss = 1.0 - F.cosine_similarity(y, loss_target, dim=-1).mean()
        else:
            primary_loss = F.mse_loss(y, loss_target)
        loss = primary_loss
        loss_components: dict[str, float] = {"primary": primary_loss.item()}

        # Add forgetting penalties
        for strategy in self._forget_strategies:
            penalty = strategy.penalty(self.field)
            loss = loss + penalty
            loss_components["ewc"] = loss_components.get("ewc", 0.0) + penalty.item()

        # Compute conflict mitigation weight multipliers (from previous step data)
        _conflict_mult = self._compute_effective_weights()

        # Auxiliary loss: world model prediction error (co-adaptation)
        aux_loss_val = 0.0
        if wm_prediction is not None:
            wm_target = wm_prediction.detach()
            wm_weight = self.config.auxiliary_loss_weight * _conflict_mult.get("wm_aux", 1.0)
            aux_loss = wm_weight * F.mse_loss(y, wm_target)
            loss = loss + aux_loss
            aux_loss_val = aux_loss.item()
            loss_components["wm_aux"] = aux_loss_val

        # Goal satisfaction regularization (co-adaptation)
        if goal_embeddings is not None and len(goal_embeddings) > 0:
            y_flat = y.mean(dim=0) if y.dim() > 1 else y
            goal_sim_sum = torch.tensor(0.0, device=y.device)
            n_valid = 0
            for goal_emb in goal_embeddings:
                ge = goal_emb.detach().to(y.device)
                if ge.shape[-1] != y_flat.shape[-1]:
                    continue
                sim = F.cosine_similarity(y_flat.unsqueeze(0), ge.unsqueeze(0))
                goal_sim_sum = goal_sim_sum + sim.squeeze()
                n_valid += 1
            if n_valid > 0:
                goal_weight = self.config.goal_loss_weight * _conflict_mult.get("goal", 1.0)
                goal_loss = -goal_weight * (goal_sim_sum / n_valid)
                loss = loss + goal_loss
                loss_components["goal"] = goal_loss.item()

        # Tier 2 guidance loss: steer field output toward established basin
        if tier2_guidance is not None and self.config.tier2_guidance_weight > 0:
            guidance = tier2_guidance.detach().to(y.device)
            y_flat = y.mean(dim=0) if y.dim() > 1 else y
            if guidance.shape[-1] == y_flat.shape[-1]:
                t2_weight = self.config.tier2_guidance_weight * _conflict_mult.get("tier2", 1.0)
                guidance_loss = t2_weight * F.mse_loss(y_flat, guidance)
                loss = loss + guidance_loss
                loss_components["tier2"] = guidance_loss.item()

        # Tier 3 axiom anchor loss: stabilize field toward core knowledge
        if axiom_anchor is not None and self.config.axiom_anchor_weight > 0:
            anchor = axiom_anchor.detach().to(y.device)
            y_flat = y.mean(dim=0) if y.dim() > 1 else y
            if anchor.shape[-1] == y_flat.shape[-1]:
                ax_weight = self.config.axiom_anchor_weight * _conflict_mult.get("axiom", 1.0)
                anchor_loss = ax_weight * F.mse_loss(y_flat, anchor)
                loss = loss + anchor_loss
                loss_components["axiom"] = anchor_loss.item()

        # Record conflict scale multipliers for observability
        for _cn, _cm in _conflict_mult.items():
            loss_components[f"conflict_scale_{_cn}"] = _cm

        # Loss component monitoring: warn if any component dominates
        primary_val = loss_components["primary"]
        if primary_val > 1e-8:
            for name, val in loss_components.items():
                if name != "primary" and val > primary_val * 10.0:
                    logger.warning(
                        "Loss component '%s' (%.4f) exceeds 10x primary loss (%.4f)",
                        name, val, primary_val,
                    )

        # Pre-backward loss sanity check
        if not torch.isfinite(loss):
            logger.warning("Non-finite loss detected — skipping update")
            self._step_count += 1
            metric = SurpriseMetric(
                grad_norm=0.0, loss=float("nan"), updated=False,
                timestamp=time.monotonic(),
            )
            self._history.append(metric)
            return metric

        # Curriculum scaling (ZPD band-pass)
        curriculum_scale = 1.0
        if self.config.curriculum_enabled:
            curriculum_scale = self._compute_curriculum_scale(loss.item())
            if curriculum_scale != 1.0:
                loss = loss * curriculum_scale
                loss_components["curriculum_scale"] = curriculum_scale

        # Gradient accumulation: scale loss to keep gradient magnitude consistent
        accum_steps = self.config.gradient_accumulation_steps
        if accum_steps > 1:
            loss = loss / accum_steps

        # Backward (accumulate gradients; only zero when stepping)
        if accum_steps <= 1 or self._accumulation_count == 0:
            self._optimizer.zero_grad()
        loss.backward()

        # Compute raw surprise
        current_grad_norm = grad_norm(self.field)

        # Gradient conflict detection (diagnostic only)
        if self.config.gradient_conflict_detection:
            self._check_gradient_conflicts(
                current_grad_norm, loss.item(), loss_components,
            )

        # Apply surprise hardening if Tier 0 config is active
        if self._hardener is not None:
            # Adaptive per-parameter gradient clipping
            self._hardener.clip_gradients(list(self.field.named_parameters()))
            # Compute hardened surprise (clamped, rate-limited, dual-path verified)
            effective_surprise = self._hardener.compute_hardened_surprise(
                current_grad_norm, latent_distance
            )
        else:
            effective_surprise = current_grad_norm
            # Update legacy EMA
            momentum = self.config.momentum
            self._surprise_ema = momentum * self._surprise_ema + (1 - momentum) * current_grad_norm

        # Tier 2 consistency check: scale surprise by consistency score
        consistency_scalar: float | None = None
        if self._consistency_checker is not None:
            with torch.no_grad():
                consistency_score = self._consistency_checker.check_consistency(x, y)
                consistency_scalar = max(consistency_score.mean().item(), 0.0)
                effective_surprise *= consistency_scalar

        # Determine threshold
        threshold = self._compute_threshold()

        # Gate decision: soft sigmoid or legacy binary
        if self.config.soft_gate_enabled:
            importance = self._compute_importance(
                consistency_scalar, confidence, effective_surprise, threshold,
            )
            gate_scale = self._compute_soft_gate(
                effective_surprise, threshold, importance,
            )
            updated = gate_scale >= self.config.soft_gate_floor
            if updated:
                for p in self.field.parameters():
                    if p.grad is not None:
                        p.grad.mul_(gate_scale)
        else:
            updated = effective_surprise > threshold

        if updated:
            # Confidence-based gradient scaling (co-adaptation)
            if (
                self.config.confidence_modulation_enabled
                and confidence is not None
            ):
                scale = self._confidence_to_lr_scale(confidence)
                if scale < 1.0:
                    for p in self.field.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

            # Salience-based gradient scaling
            if self.config.salience_gradient_scaling and salience_score is not None:
                sal_scale = max(0.1, min(2.0, salience_score * 2.0))
                if sal_scale != 1.0:
                    for p in self.field.parameters():
                        if p.grad is not None:
                            p.grad.mul_(sal_scale)

            # External LR scale (valence risk_tolerance + goal urgency)
            if external_lr_scale is not None and external_lr_scale != 1.0:
                lo, hi = self.config.external_lr_scale_range
                clamped = max(lo, min(hi, external_lr_scale))
                for p in self.field.parameters():
                    if p.grad is not None:
                        p.grad.mul_(clamped)

            # Gradient accumulation: only step every N accumulations
            self._accumulation_count += 1
            should_step = (
                accum_steps <= 1
                or self._accumulation_count >= accum_steps
            )

            if should_step:
                self._accumulation_count = 0
                self._optimizer.step()

                # Step LR scheduler (not during replay)
                if self._scheduler is not None:
                    self._scheduler.step()

                # Update importance estimates for EWC
                for strategy in self._forget_strategies:
                    strategy.update_importance(self.field)

                # Check for distribution shift (LoRA merge)
                if self._lora_manager is not None:
                    history_norms = [m.grad_norm for m in self._history]
                    if self._lora_manager.check_and_merge(
                        history_norms, self._merge_context,
                    ):
                        # After merge, EWC anchors should update and optimizer rebuilt
                        if self._ewc_strategy is not None:
                            self._ewc_strategy.update_anchors(self.field)
                        self._optimizer = self._build_optimizer()
                        self._scheduler = self._build_scheduler()

        self._step_count += 1
        metric = SurpriseMetric(
            grad_norm=current_grad_norm,
            loss=loss.item(),
            updated=updated,
            timestamp=time.monotonic(),
            auxiliary_loss=aux_loss_val,
            loss_components=loss_components,
        )
        self._history.append(metric)
        return metric

    def _confidence_to_lr_scale(self, confidence: float) -> float:
        """Map metacognition confidence [0, 1] to gradient scale [0.1, 1.0].

        Low confidence (below low_threshold): scale down to 0.1 (cautious).
        Mid confidence: linear ramp from 0.1 to 1.0.
        High confidence (above high_threshold): full scale 1.0.
        """
        low = self.config.confidence_low_threshold
        high = self.config.confidence_high_threshold
        if confidence >= high:
            return 1.0
        if confidence <= low:
            return 0.1
        return 0.1 + 0.9 * (confidence - low) / (high - low)

    def process_stream(
        self,
        stream: Iterable[torch.Tensor],
        callback: Callable[[SurpriseMetric], None] | None = None,
    ) -> list[SurpriseMetric]:
        """Process an iterable stream of inputs.

        Args:
            stream: Iterable of input tensors.
            callback: Optional function called with each SurpriseMetric.

        Returns:
            List of all SurpriseMetric results.
        """
        results: list[SurpriseMetric] = []
        for x in stream:
            metric = self.process(x)
            results.append(metric)
            if callback is not None:
                callback(metric)
        return results

    def replay_episode(
        self,
        input_embedding: torch.Tensor,
        logit_snapshot: torch.Tensor,
        lr_scale: float = 0.5,
    ) -> float:
        """Replay a stored episode through the field with reduced LR.

        Performs one forward/backward pass using the episode's stored
        input_embedding as input and logit_snapshot as target. Uses a
        scaled learning rate to avoid overwriting recent online learning.
        EWC penalties still apply to prevent forgetting.

        Called during consolidation replay — bypasses the surprise gate
        because we are reinforcing known-important patterns.

        Args:
            input_embedding: (d_model,) stored episode input.
            logit_snapshot: (d_model,) stored field output at storage time.
            lr_scale: Multiplier for the base learning rate.

        Returns:
            Replay loss value (float).
        """
        self.field.train()

        # Temporarily scale LR
        original_lrs = []
        for pg in self._optimizer.param_groups:
            original_lrs.append(pg["lr"])
            pg["lr"] = pg["lr"] * lr_scale

        try:
            y = self.field(input_embedding)
            loss = F.mse_loss(y, logit_snapshot)

            # Add forgetting penalties (EWC)
            for strategy in self._forget_strategies:
                loss = loss + strategy.penalty(self.field)

            self._optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping if hardener available
            if self._hardener is not None:
                self._hardener.clip_gradients(list(self.field.named_parameters()))

            # Unconditional update (no surprise gate for replay)
            self._optimizer.step()

            # Update EWC importance estimates
            for strategy in self._forget_strategies:
                strategy.update_importance(self.field)

            return loss.item()
        finally:
            # Restore original LRs
            for pg, orig_lr in zip(self._optimizer.param_groups, original_lrs):
                pg["lr"] = orig_lr

    def query(self, x: torch.Tensor, config: AttractorConfig | None = None) -> AttractorResult:
        """Query the manifold for the nearest attractor to input x.

        Args:
            x: Input tensor.
            config: Optional AttractorConfig override.

        Returns:
            AttractorResult with converged point.
        """
        return AttractorQuery(self.field, config or AttractorConfig()).query(x)

    @property
    def surprise_history(self) -> list[SurpriseMetric]:
        """Full history of surprise metrics from all processed inputs."""
        return self._history

    @property
    def lora_manager(self) -> OnlineLoRAManager | None:
        """Access the LoRA manager, if LoRA is enabled."""
        return self._lora_manager

    @property
    def ewc_strategy(self) -> EWCStrategy | None:
        """Access the EWC strategy, if EWC is enabled."""
        return self._ewc_strategy

    def reset_working_memory(self) -> None:
        """Reset Tier 0 working memory: reinitialize field weights and all state.

        This implements Tier 0 volatility — complete session reset that erases
        any potentially poisoned state. Call at session boundaries.
        """
        # Reinitialize field weights
        for module in self.field.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(module.weight)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        # Reset optimizer and scheduler
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()

        # Reset surprise tracking
        self._surprise_ema = 0.0
        self._history.clear()
        self._step_count = 0

        # Reset hardener state
        if self._hardener is not None:
            self._hardener.reset()

        # Reset gradient conflict detection state
        self._grad_loss_ratio_ema = 0.0
        self._component_emas.clear()
        self._component_directions.clear()
        self._opposition_scores.clear()

        # Reset curriculum state
        self._competence_ema = 0.0
        self._curriculum_active_step = 0

        logger.info("Working memory (Tier 0) reset")

    def set_consistency_checker(self, checker: ConsistencyChecker | None) -> None:
        """Set or clear the Tier 2 consistency checker.

        Called by the orchestrator when Tier 2 (EssentialMemory) is initialized.
        """
        self._consistency_checker = checker

    def set_merge_context(self, ctx) -> None:
        """Set the LoRA merge context (gathered by orchestrator)."""
        self._merge_context = ctx

    def register_consolidated_concepts(self, keys: list[torch.Tensor]) -> None:
        """Store newly consolidated basin keys for guidance during next process()."""
        self._consolidated_concept_keys = [k.detach().clone() for k in keys]

    def clear_consolidated_concepts(self) -> None:
        """Clear stored consolidated concept keys."""
        self._consolidated_concept_keys = []

    def reset_optimizer(self) -> None:
        """Rebuild the optimizer. Call after LoRA merge changes the parameter set."""
        self._optimizer = self._build_optimizer()
        self._scheduler = self._build_scheduler()

    def _build_optimizer(self) -> torch.optim.AdamW:
        """Build an AdamW optimizer over the appropriate parameters."""
        if self._lora_manager is not None:
            params = list(self._lora_manager.trainable_parameters())
        else:
            params = list(self.field.parameters())
        use_fused = torch.cuda.is_available() and all(
            p.is_cuda for p in params
        )
        return torch.optim.AdamW(
            params,
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            fused=use_fused,
        )

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Build an LR scheduler: optional warmup followed by optional cosine decay."""
        schedulers: list[torch.optim.lr_scheduler.LRScheduler] = []
        milestones: list[int] = []

        if self.config.warmup_steps > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                self._optimizer,
                start_factor=self.config.warmup_start_factor,
                total_iters=self.config.warmup_steps,
            )
            schedulers.append(warmup)
            milestones.append(self.config.warmup_steps)

        if self.config.lr_decay_enabled:
            decay = torch.optim.lr_scheduler.CosineAnnealingLR(
                self._optimizer,
                T_max=self.config.lr_decay_steps,
                eta_min=self.config.lr * self.config.lr_decay_min_factor,
            )
            schedulers.append(decay)

        if len(schedulers) > 1:
            return torch.optim.lr_scheduler.SequentialLR(
                self._optimizer, schedulers, milestones,
            )
        if schedulers:
            return schedulers[0]
        return None

    def _compute_effective_weights(self) -> dict[str, float]:
        """Compute dynamic weight multipliers based on opposition scores.

        Returns a dict mapping component name -> multiplier in [weight_floor, 1.0].
        Components not tracked or below threshold get no entry (use base weight).
        Primary loss is never scaled.
        """
        if (
            not self.config.gradient_conflict_mitigation
            or not self.config.gradient_conflict_detection
        ):
            return {}

        cfg = self.config
        multipliers: dict[str, float] = {}

        for name, score in self._opposition_scores.items():
            if name == "primary" or score < cfg.gradient_conflict_score_threshold:
                continue
            raw = 1.0 - cfg.gradient_conflict_damping * score
            multipliers[name] = max(raw, cfg.gradient_conflict_weight_floor)

        return multipliers

    def _check_gradient_conflicts(
        self,
        current_grad_norm: float,
        total_loss: float,
        loss_components: dict[str, float],
    ) -> None:
        """Detect gradient conflicts via ratio tracking and component analysis.

        Three diagnostic signals (logging only, no behavioral change):
        1. Grad-loss ratio drop: grad_norm/loss drops below EMA → cancellation.
        2. Component oscillation: high sign-reversal rate in loss deltas.
        3. Component opposition: pairs consistently moving opposite directions.
        """
        cfg = self.config
        decay = cfg.gradient_conflict_ema_decay

        # Signal 1: grad_norm / loss ratio
        if total_loss > 1e-8:
            ratio = current_grad_norm / total_loss
            if self._grad_loss_ratio_ema < 1e-8:
                self._grad_loss_ratio_ema = ratio
            else:
                self._grad_loss_ratio_ema = (
                    decay * self._grad_loss_ratio_ema + (1 - decay) * ratio
                )
                if ratio < cfg.gradient_conflict_warn_ratio * self._grad_loss_ratio_ema:
                    logger.warning(
                        "Gradient conflict: grad/loss ratio %.4f is %.0f%% "
                        "below EMA %.4f — loss components may be cancelling",
                        ratio,
                        (1 - ratio / self._grad_loss_ratio_ema) * 100,
                        self._grad_loss_ratio_ema,
                    )

        # Signals 2 & 3: per-component EMA deltas
        for name, val in loss_components.items():
            if name == "curriculum_scale" or name.startswith("conflict_scale_"):
                continue
            prev_ema = self._component_emas.get(name, val)
            new_ema = decay * prev_ema + (1 - decay) * val
            delta = val - prev_ema

            dirs = self._component_directions.setdefault(name, [])
            dirs.append(delta)
            if len(dirs) > 50:
                dirs.pop(0)

            self._component_emas[name] = new_ema

        # Signal 2: oscillation — count sign changes in recent deltas
        for name, dirs in self._component_directions.items():
            if len(dirs) >= 10:
                sign_changes = sum(
                    1 for i in range(1, len(dirs))
                    if dirs[i] * dirs[i - 1] < 0
                    and abs(dirs[i]) > 1e-6
                )
                oscillation_rate = sign_changes / (len(dirs) - 1)
                if oscillation_rate > 0.7:
                    logger.warning(
                        "Loss component '%s' oscillating: %.0f%% sign reversals "
                        "in last %d steps",
                        name, oscillation_rate * 100, len(dirs),
                    )

        # Signal 3: opposition — pairs moving in opposite directions
        names = [
            n for n in loss_components
            if n != "curriculum_scale" and not n.startswith("conflict_scale_")
        ]
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                dirs_i = self._component_directions.get(names[i], [])
                dirs_j = self._component_directions.get(names[j], [])
                if len(dirs_i) >= 5 and len(dirs_j) >= 5:
                    recent = min(len(dirs_i), len(dirs_j), 10)
                    opposing = sum(
                        1 for k in range(-recent, 0)
                        if dirs_i[k] * dirs_j[k] < 0
                        and abs(dirs_i[k]) > 1e-6
                        and abs(dirs_j[k]) > 1e-6
                    )
                    if opposing > recent * 0.7:
                        logger.warning(
                            "Gradient conflict: '%s' and '%s' opposing in "
                            "%d/%d recent steps",
                            names[i], names[j], opposing, recent,
                        )

        # Update opposition scores for mitigation (primary as reference)
        if self.config.gradient_conflict_mitigation:
            dirs_primary = self._component_directions.get("primary", [])
            present_names = set(names)
            if len(dirs_primary) >= 5:
                for name in names:
                    if name == "primary":
                        continue
                    dirs_other = self._component_directions.get(name, [])
                    if len(dirs_other) >= 5:
                        recent = min(len(dirs_primary), len(dirs_other), 10)
                        opposing = sum(
                            1 for k in range(-recent, 0)
                            if dirs_primary[k] * dirs_other[k] < 0
                            and abs(dirs_primary[k]) > 1e-6
                            and abs(dirs_other[k]) > 1e-6
                        )
                        raw_score = opposing / recent
                        prev = self._opposition_scores.get(name, 0.0)
                        decay = self.config.gradient_conflict_score_ema_decay
                        self._opposition_scores[name] = (
                            decay * prev + (1 - decay) * raw_score
                        )

            # Decay scores for components absent this step
            for name in list(self._opposition_scores):
                if name not in present_names:
                    self._opposition_scores[name] *= (
                        self.config.gradient_conflict_score_ema_decay
                    )

    def _compute_curriculum_scale(self, current_loss: float) -> float:
        """Compute loss scaling based on zone of proximal development.

        Returns 1.0 when the input is within the ZPD (normal learning).
        Returns a reduced scale for too-easy or too-hard inputs.
        """
        cfg = self.config

        # Update competence EMA
        if self._competence_ema < 1e-8:
            self._competence_ema = current_loss
        else:
            decay = cfg.curriculum_competence_ema_decay
            self._competence_ema = (
                decay * self._competence_ema + (1 - decay) * current_loss
            )

        self._curriculum_active_step += 1

        # Wait for warmup before applying curriculum
        if self._curriculum_active_step < cfg.curriculum_warmup_steps:
            return 1.0

        competence = self._competence_ema
        if competence < 1e-8:
            return 1.0

        # Too easy: loss far below competence
        if current_loss < competence * cfg.curriculum_too_easy_threshold:
            return cfg.curriculum_too_easy_scale

        # Too hard: loss far above competence
        if current_loss > competence * cfg.curriculum_too_hard_ratio:
            return cfg.curriculum_too_hard_scale

        return 1.0

    def _compute_threshold(self) -> float:
        """Compute the surprise threshold (static or adaptive)."""
        if not self.config.adaptive_surprise:
            return self.config.surprise_threshold

        # Adaptive: use percentile of recent history
        if len(self._history) < 10:
            return self.config.surprise_threshold

        recent_norms = [m.grad_norm for m in self._history[-100:]]
        recent_norms.sort()
        idx = int(self.config.surprise_percentile * len(recent_norms))
        idx = min(idx, len(recent_norms) - 1)
        return recent_norms[idx]

    def _compute_importance(
        self,
        consistency_scalar: float | None,
        confidence: float | None,
        effective_surprise: float,
        threshold: float,
    ) -> float:
        """Compute composite importance score in [0.5, 1.5].

        Shifts the soft gate center:
          importance > 1.0 → center < threshold → easier to update
          importance < 1.0 → center > threshold → harder to update
        """
        cfg = self.config
        score = 0.0
        weight_sum = 0.0

        if consistency_scalar is not None:
            score += cfg.importance_consistency_weight * consistency_scalar
            weight_sum += cfg.importance_consistency_weight

        if confidence is not None:
            score += cfg.importance_confidence_weight * confidence
            weight_sum += cfg.importance_confidence_weight

        if threshold > 0:
            surprise_ratio = min(effective_surprise / threshold, 2.0) / 2.0
            score += cfg.importance_surprise_weight * surprise_ratio
            weight_sum += cfg.importance_surprise_weight

        if weight_sum > 0:
            normalized = score / weight_sum  # ~[0, 1]
            return 0.5 + normalized
        return 1.0

    def _compute_soft_gate(
        self,
        effective_surprise: float,
        threshold: float,
        importance: float,
    ) -> float:
        """Compute continuous gate scale via importance-shifted sigmoid.

        Returns a value in [0, 1] replacing the binary threshold check.
        """
        center = threshold / max(importance, 0.1)
        k = self.config.soft_gate_steepness
        z = k * (effective_surprise - center)
        # Numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z))
        ez = math.exp(z)
        return ez / (1.0 + ez)
