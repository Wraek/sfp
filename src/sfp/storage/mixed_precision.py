"""Mixed-precision management for memory-efficient inference and training."""

from __future__ import annotations

import torch
import torch.nn as nn

from sfp.core.field import SemanticFieldProcessor
from sfp.utils.logging import get_logger

logger = get_logger("storage.mixed_precision")


class MixedPrecisionManager:
    """Manages mixed-precision storage: FP16 base weights, FP32 optimizer states.

    Converts base Linear weights to FP16 for memory savings while keeping
    LoRA adapters and optimizer states in FP32 for training stability.
    """

    def __init__(self, field: SemanticFieldProcessor) -> None:
        self.field = field

    def apply(self, lora_manager: object | None = None) -> None:
        """Convert base weights to FP16.

        Args:
            lora_manager: Optional OnlineLoRAManager. If provided, only
                base Linear weights are converted; LoRA A/B stay FP32.
        """
        converted = 0
        for module in self.field.modules():
            if isinstance(module, nn.Linear):
                module.weight.data = module.weight.data.half()
                if module.bias is not None:
                    module.bias.data = module.bias.data.half()
                converted += 1

        # If LoRA is active, ensure LoRA params stay FP32
        if lora_manager is not None and hasattr(lora_manager, "lora_layers"):
            for layer in lora_manager.lora_layers:
                layer.A.data = layer.A.data.float()
                layer.B.data = layer.B.data.float()

        logger.info("Converted %d Linear layers to FP16", converted)

    def memory_footprint(
        self,
        lora_manager: object | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        batch_size: int = 1,
    ) -> dict[str, int]:
        """Compute memory footprint breakdown.

        Args:
            lora_manager: Optional LoRA manager for separate accounting.
            optimizer: Optional optimizer for state memory estimation.
            batch_size: Batch size for activation memory estimate.

        Returns:
            Dict with byte counts per category.
        """
        base_bytes = 0
        lora_bytes = 0

        lora_params: set[int] = set()
        if lora_manager is not None and hasattr(lora_manager, "lora_layers"):
            for layer in lora_manager.lora_layers:
                lora_bytes += layer.A.numel() * layer.A.element_size()
                lora_bytes += layer.B.numel() * layer.B.element_size()
                lora_params.add(id(layer.A))
                lora_params.add(id(layer.B))

        for param in self.field.parameters():
            if id(param) not in lora_params:
                base_bytes += param.numel() * param.element_size()

        # Optimizer state estimate: Adam stores 2 running averages per param
        optimizer_bytes = 0
        if optimizer is not None:
            for group in optimizer.param_groups:
                for p in group["params"]:
                    state = optimizer.state.get(p, {})
                    for v in state.values():
                        if isinstance(v, torch.Tensor):
                            optimizer_bytes += v.numel() * v.element_size()

        # Activation estimate: each layer stores activations for backward
        config = self.field.config
        activation_bytes = (
            batch_size * config.n_layers * config.dim * 2  # FP16
        )

        total = base_bytes + lora_bytes + optimizer_bytes + activation_bytes

        report = {
            "base_weights": base_bytes,
            "lora_params": lora_bytes,
            "optimizer_states": optimizer_bytes,
            "activations_estimate": activation_bytes,
            "total": total,
        }

        logger.info(
            "Memory footprint: base=%dKB, lora=%dKB, opt=%dKB, act=%dKB, total=%dKB",
            base_bytes // 1024,
            lora_bytes // 1024,
            optimizer_bytes // 1024,
            activation_bytes // 1024,
            total // 1024,
        )
        return report
