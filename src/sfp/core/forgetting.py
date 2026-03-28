"""Forgetting mitigation strategies: EWC and weight decay."""

from __future__ import annotations

import torch
import torch.nn as nn

from sfp.config import EWCConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.utils.logging import get_logger

logger = get_logger("core.forgetting")


class WeightDecayStrategy:
    """Primary forgetting strategy: AdamW weight decay (handled by the optimizer).

    This is a no-op penalty — weight decay is applied by the AdamW optimizer,
    not as a loss term. Exists to make the strategy stack uniform.
    """

    def penalty(self, model: nn.Module) -> torch.Tensor:
        device = next(model.parameters()).device
        return torch.tensor(0.0, device=device)

    def update_importance(self, model: nn.Module) -> None:
        pass


class EWCStrategy:
    """Online Elastic Weight Consolidation.

    Maintains a running diagonal Fisher information estimate as a proxy for
    parameter importance. Penalizes deviations from anchor parameters
    proportional to their estimated importance.

    Penalty: lambda * sum_i F_i * (theta_i - theta*_i)^2
    """

    def __init__(self, field: SemanticFieldProcessor, config: EWCConfig) -> None:
        self.config = config
        self._fisher: dict[str, torch.Tensor] = {}
        self._anchors: dict[str, torch.Tensor] = {}
        self._initialize(field)

    def _initialize(self, field: SemanticFieldProcessor) -> None:
        """Initialize Fisher diagonal to zeros and anchors to current params."""
        for name, param in field.named_parameters():
            if param.requires_grad:
                self._fisher[name] = torch.zeros_like(param.data)
                self._anchors[name] = param.data.clone()

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Compute the EWC penalty: lambda * mean_i F_i * (theta_i - theta*_i)^2.

        Normalizes by total parameter count so the penalty magnitude stays
        proportional to the primary loss regardless of model size.
        """
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device)
        n_params = 0
        for name, param in model.named_parameters():
            if name in self._fisher:
                fisher = self._fisher[name]
                anchor = self._anchors[name]
                loss = loss + (fisher * (param - anchor) ** 2).sum()
                n_params += param.numel()
        return self.config.lambda_ * (loss / max(n_params, 1))

    def update_importance(self, model: nn.Module) -> None:
        """Update the running Fisher diagonal estimate from current gradients.

        Uses exponential moving average: F = decay * F + (1 - decay) * grad^2
        """
        for name, param in model.named_parameters():
            if param.grad is not None and name in self._fisher:
                self._fisher[name] = (
                    self.config.decay * self._fisher[name]
                    + (1 - self.config.decay) * param.grad.data ** 2
                )

    def update_anchors(self, model: nn.Module) -> None:
        """Snapshot current parameters as new anchors.

        Call after LoRA merges or explicit consolidation to reset
        the reference point for EWC penalties.
        """
        for name, param in model.named_parameters():
            if name in self._anchors:
                self._anchors[name] = param.data.clone()
        logger.info("EWC anchors updated to current parameters")
