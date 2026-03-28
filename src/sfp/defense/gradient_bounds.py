"""Defense Layer 3: Adaptive gradient clipping and update magnitude constraints.

Implements per-parameter ARC-style adaptive gradient clipping and per-step
weight change L2 budget enforcement from the poisoning defense document.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sfp.utils.logging import get_logger

logger = get_logger("defense.gradient")


class AdaptiveGradientClipper:
    """Per-parameter adaptive gradient clipping (ARC-style).

    Tracks a running EMA of gradient magnitudes per parameter and clips any
    gradient exceeding clip_multiplier * EMA. Adapts to the natural variance
    of honest updates while bounding adversarial outliers.

    Unlike the simpler version in SurpriseHardener, this operates at the
    per-element level and uses a two-pass approach for accurate threshold
    estimation.

    Args:
        model: The model whose gradients to clip.
        clip_multiplier: How many EMA-widths above the mean to clip.
        ema_decay: Decay factor for the gradient magnitude EMA.
    """

    def __init__(
        self,
        model: nn.Module,
        clip_multiplier: float = 2.0,
        ema_decay: float = 0.99,
    ) -> None:
        self._clip_multiplier = clip_multiplier
        self._ema_decay = ema_decay
        self._grad_ema: dict[str, torch.Tensor] = {}
        self._grad_var_ema: dict[str, torch.Tensor] = {}
        self._initialized: bool = False

    def clip(self, model: nn.Module) -> float:
        """Apply adaptive gradient clipping to all parameters.

        Returns:
            Fraction of gradient elements that were clipped (0.0 = none, 1.0 = all).
        """
        total_elements = 0
        clipped_elements = 0

        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            grad = param.grad.data
            grad_abs = grad.abs()
            n_elem = grad.numel()
            total_elements += n_elem

            if name not in self._grad_ema:
                self._grad_ema[name] = grad_abs.clone()
                self._grad_var_ema[name] = torch.zeros_like(grad_abs)
                continue

            # Update EMA of gradient magnitude
            ema = self._grad_ema[name]
            ema.mul_(self._ema_decay).add_(grad_abs, alpha=1.0 - self._ema_decay)

            # Update EMA of gradient variance (for adaptive threshold)
            diff = (grad_abs - ema).pow(2)
            var_ema = self._grad_var_ema[name]
            var_ema.mul_(self._ema_decay).add_(diff, alpha=1.0 - self._ema_decay)

            # Adaptive threshold: mean + clip_multiplier * std
            std = var_ema.sqrt()
            threshold = ema + self._clip_multiplier * std
            threshold = torch.clamp(threshold, min=1e-7)

            # Clip
            exceeds = grad_abs > threshold
            n_clipped = exceeds.sum().item()
            clipped_elements += n_clipped

            if n_clipped > 0:
                param.grad.data = torch.clamp(grad, min=-threshold, max=threshold)

        if total_elements == 0:
            return 0.0

        clip_frac = clipped_elements / total_elements
        if clip_frac > 0.1:
            logger.debug("Gradient clipping: %.1f%% of elements clipped", clip_frac * 100)

        return clip_frac


class UpdateBudget:
    """Per-step weight change L2 budget enforcement.

    Enforces a hard bound on the total L2 norm of weight changes per step.
    If the update would exceed the budget, all gradients are scaled down
    proportionally to stay within bounds.

    Args:
        model: The model to monitor.
        budget_fraction: Maximum weight change as a fraction of total weight norm.
    """

    def __init__(self, model: nn.Module, budget_fraction: float = 0.005) -> None:
        self._budget_fraction = budget_fraction
        self._weight_snapshot: dict[str, torch.Tensor] = {}
        self._budget_exceed_count: int = 0
        self._snapshot(model)

    def _snapshot(self, model: nn.Module) -> None:
        """Take a snapshot of current weights."""
        for name, param in model.named_parameters():
            self._weight_snapshot[name] = param.data.clone()

    def enforce(self, model: nn.Module) -> bool:
        """Check and enforce the weight change budget after an optimizer step.

        If the budget is exceeded, rolls back the weights to the snapshot
        plus the budget-limited delta.

        Args:
            model: The model to check.

        Returns:
            True if the budget was exceeded and enforcement was applied.
        """
        total_weight_norm_sq = 0.0
        total_delta_norm_sq = 0.0

        for name, param in model.named_parameters():
            if name not in self._weight_snapshot:
                continue
            total_weight_norm_sq += param.data.pow(2).sum().item()
            delta = param.data - self._weight_snapshot[name]
            total_delta_norm_sq += delta.pow(2).sum().item()

        total_weight_norm = total_weight_norm_sq**0.5
        total_delta_norm = total_delta_norm_sq**0.5

        budget = total_weight_norm * self._budget_fraction

        exceeded = total_delta_norm > budget
        if exceeded:
            # Scale back all changes proportionally
            scale = budget / (total_delta_norm + 1e-12)
            for name, param in model.named_parameters():
                if name not in self._weight_snapshot:
                    continue
                delta = param.data - self._weight_snapshot[name]
                param.data = self._weight_snapshot[name] + delta * scale

            self._budget_exceed_count += 1
            if self._budget_exceed_count <= 3 or self._budget_exceed_count % 100 == 0:
                logger.debug(
                    "Update budget exceeded (#%d): delta=%.6f, budget=%.6f (%.3f%%), scaled to %.3f%%",
                    self._budget_exceed_count,
                    total_delta_norm,
                    budget,
                    self._budget_fraction * 100,
                    scale * self._budget_fraction * 100,
                )

        # Update snapshot for next step
        self._snapshot(model)
        return exceeded

    @property
    def budget_fraction(self) -> float:
        return self._budget_fraction
