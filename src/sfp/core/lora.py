"""Online-LoRA: low-rank adaptation for continual learning without catastrophic forgetting."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from sfp.config import LoRAConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.utils.logging import get_logger

logger = get_logger("core.lora")


@dataclass
class LoRAMergeContext:
    """Additional signals beyond surprise history for LoRA merge decisions."""

    prediction_uncertainty_history: list[float] | None = None
    mood_history: list[float] | None = None
    goal_progress_history: dict[int, list[float]] | None = field(default=None)


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a low-rank bypass for parameter-efficient adaptation.

    output = base(x) + (x @ A @ B) * (alpha / rank)

    Base weights are frozen; only A and B are trainable.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        d_in = base.in_features
        d_out = base.out_features
        device = base.weight.device  # Match device of base layer
        self.A = nn.Parameter(torch.randn(d_in, config.rank, device=device) * 0.01)
        self.B = nn.Parameter(torch.zeros(config.rank, d_out, device=device))
        self.scaling = config.alpha / config.rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = (x @ self.A @ self.B) * self.scaling
        return base_out + lora_out

    def merge_and_reinit(self) -> None:
        """Merge LoRA weights into base and reinitialize adapters."""
        with torch.no_grad():
            self.base.weight.data += (self.B.T @ self.A.T) * self.scaling
            nn.init.normal_(self.A, std=0.01)
            nn.init.zeros_(self.B)

    @property
    def lora_param_count(self) -> int:
        return self.A.numel() + self.B.numel()


def _replace_module_recursive(
    parent: nn.Module, target: nn.Module, replacement: nn.Module
) -> bool:
    """Recursively search parent's children and replace target with replacement."""
    for name, child in parent._modules.items():
        if child is target:
            parent._modules[name] = replacement
            return True
        if child is not None and _replace_module_recursive(child, target, replacement):
            return True
    return False


class OnlineLoRAManager:
    """Manages LoRA adapters across all Linear layers of a SemanticFieldProcessor.

    Handles wrapping, training parameter extraction, merge/reinit on
    distribution shift, and forced merges.
    """

    def __init__(self, field: SemanticFieldProcessor, config: LoRAConfig) -> None:
        self.config = config
        self.lora_layers: list[LoRALinear] = []
        self._wrap_field(field)

    def _wrap_field(self, field: SemanticFieldProcessor) -> None:
        """Replace each nn.Linear in field.net with a LoRALinear wrapper."""
        linears = field.linear_layers()
        for linear in linears:
            lora_linear = LoRALinear(linear, self.config)
            replaced = _replace_module_recursive(field.net, linear, lora_linear)
            if replaced:
                self.lora_layers.append(lora_linear)
            else:
                logger.warning("Failed to replace Linear layer in field.net")

        logger.info(
            "Wrapped %d Linear layers with LoRA (rank=%d, total_lora_params=%d)",
            len(self.lora_layers),
            self.config.rank,
            self.total_lora_params,
        )

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only the LoRA A and B parameters (not base weights)."""
        for layer in self.lora_layers:
            yield layer.A
            yield layer.B

    @property
    def total_lora_params(self) -> int:
        return sum(layer.lora_param_count for layer in self.lora_layers)

    def check_and_merge(
        self,
        surprise_history: list[float],
        merge_context: LoRAMergeContext | None = None,
    ) -> bool:
        """Detect distribution shift and merge LoRA if triggered.

        Checks multiple signals: surprise ratio (original), sustained prediction
        uncertainty, sustained negative mood, and goal progress stalls.

        Returns:
            True if a merge was performed.
        """
        # Signal 1: Surprise ratio (original check)
        if len(surprise_history) >= 100:
            recent = surprise_history[-50:]
            previous = surprise_history[-100:-50]
            recent_mean = sum(recent) / len(recent)
            previous_mean = sum(previous) / max(len(previous), 1)

            if previous_mean > 0:
                ratio = recent_mean / previous_mean
                if ratio > self.config.merge_threshold:
                    self.merge_all()
                    logger.info(
                        "Distribution shift detected (ratio=%.3f > %.3f). Merged LoRA weights.",
                        ratio,
                        self.config.merge_threshold,
                    )
                    return True

        # Multi-signal merge triggers
        if merge_context is not None:
            cfg = self.config

            # Signal 2: Sustained high prediction uncertainty
            if merge_context.prediction_uncertainty_history is not None:
                window = merge_context.prediction_uncertainty_history[-cfg.uncertainty_merge_window:]
                if (
                    len(window) >= cfg.uncertainty_merge_window
                    and all(u > cfg.uncertainty_merge_threshold for u in window)
                ):
                    self.merge_all()
                    logger.info(
                        "Uncertainty-triggered LoRA merge (sustained >%.2f for %d steps)",
                        cfg.uncertainty_merge_threshold, cfg.uncertainty_merge_window,
                    )
                    return True

            # Signal 3: Sustained negative mood
            if merge_context.mood_history is not None:
                window = merge_context.mood_history[-cfg.mood_merge_window:]
                if (
                    len(window) >= cfg.mood_merge_window
                    and all(m < cfg.mood_merge_threshold for m in window)
                ):
                    self.merge_all()
                    logger.info(
                        "Mood-triggered LoRA merge (sustained <%.2f for %d steps)",
                        cfg.mood_merge_threshold, cfg.mood_merge_window,
                    )
                    return True

            # Signal 4: Goal progress stall
            if merge_context.goal_progress_history is not None:
                for goal_id, hist in merge_context.goal_progress_history.items():
                    if len(hist) >= cfg.goal_stall_merge_steps:
                        recent = hist[-cfg.goal_stall_merge_steps:]
                        if max(recent) - min(recent) < 0.01:
                            self.merge_all()
                            logger.info(
                                "Goal-stall-triggered LoRA merge (goal %d stalled for %d steps)",
                                goal_id, cfg.goal_stall_merge_steps,
                            )
                            return True

        return False

    def merge_all(self) -> None:
        """Force merge all LoRA adapters into base weights and reinitialize."""
        for layer in self.lora_layers:
            layer.merge_and_reinit()
