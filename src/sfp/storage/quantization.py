"""INT8 quantization and information content estimation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from sfp.config import FieldConfig, QuantizationConfig
from sfp.utils.logging import get_logger

logger = get_logger("storage.quantization")


def quantize_tensor_int8(
    tensor: torch.Tensor, per_channel: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-channel symmetric INT8 quantization.

    Args:
        tensor: Float tensor to quantize.
        per_channel: If True, compute scale per output channel (dim 0).

    Returns:
        (quantized_int8, scale, zero_point) tuple.
    """
    if per_channel and tensor.dim() >= 2:
        # Scale per output channel (row)
        scale = tensor.abs().amax(dim=-1, keepdim=True) / 127.0
    else:
        scale = tensor.abs().amax() / 127.0
        scale = scale.unsqueeze(0)

    # Avoid division by zero
    scale = torch.clamp(scale, min=1e-8)

    quantized = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    zero_point = torch.zeros_like(scale)
    return quantized, scale, zero_point


def dequantize_tensor_int8(
    quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    """Dequantize INT8 tensor back to float32.

    Args:
        quantized: INT8 tensor.
        scale: Per-channel or per-tensor scale.
        zero_point: Zero point (typically zeros for symmetric).

    Returns:
        Float32 reconstructed tensor.
    """
    return quantized.float() * scale + zero_point


class ManifoldQuantizer:
    """Quantize and dequantize entire SemanticFieldProcessor models."""

    @staticmethod
    def quantize(
        field: nn.Module, config: QuantizationConfig | None = None
    ) -> dict:
        """Quantize all parameters to INT8.

        Args:
            field: The SemanticFieldProcessor to quantize.
            config: Quantization configuration.

        Returns:
            Dict mapping param name -> (quantized, scale, zero_point),
            plus 'field_config' for reconstruction.
        """
        config = config or QuantizationConfig()
        state: dict = {"params": {}, "field_config": None}

        # Store field config if available
        if hasattr(field, "config"):
            from dataclasses import asdict

            state["field_config"] = asdict(field.config)

        for name, param in field.named_parameters():
            quantized, scale, zero_point = quantize_tensor_int8(
                param.data, per_channel=config.per_channel
            )
            state["params"][name] = {
                "quantized": quantized,
                "scale": scale,
                "zero_point": zero_point,
                "shape": param.shape,
            }

        total_original = sum(
            p.numel() * p.element_size() for p in field.parameters()
        )
        total_quantized = sum(
            v["quantized"].numel() + v["scale"].numel() * 4
            for v in state["params"].values()
        )
        logger.info(
            "Quantized %d params: %d bytes -> %d bytes (%.1fx compression)",
            len(state["params"]),
            total_original,
            total_quantized,
            total_original / max(1, total_quantized),
        )
        return state

    @staticmethod
    def dequantize(
        quantized_state: dict,
        field_config: FieldConfig | None = None,
        device: str | torch.device = "cpu",
    ) -> nn.Module:
        """Reconstruct a SemanticFieldProcessor from quantized state.

        Args:
            quantized_state: Output of quantize().
            field_config: FieldConfig to build the model. If None, uses
                the config stored in quantized_state.
            device: Target device.

        Returns:
            Reconstructed SemanticFieldProcessor with dequantized weights.
        """
        from sfp.core.field import SemanticFieldProcessor

        if field_config is None:
            cfg_dict = quantized_state.get("field_config")
            if cfg_dict is None:
                raise ValueError(
                    "No field_config in quantized state and none provided"
                )
            field_config = FieldConfig(**cfg_dict)

        field = SemanticFieldProcessor(field_config)
        state_dict = field.state_dict()

        for name, qdata in quantized_state["params"].items():
            if name in state_dict:
                restored = dequantize_tensor_int8(
                    qdata["quantized"], qdata["scale"], qdata["zero_point"]
                )
                state_dict[name] = restored.reshape(qdata["shape"])

        field.load_state_dict(state_dict)
        field.to(device)
        return field

    @staticmethod
    def estimate_information_content(field: nn.Module) -> float:
        """Estimate usable bits per parameter via weight histogram entropy.

        For each parameter tensor, bins values into 256 buckets and computes
        Shannon entropy. Typical MLP manifolds use ~3-6 bits/param, consistent
        with the ~5 bit ceiling from information-theoretic analysis.

        Args:
            field: The model to analyze.

        Returns:
            Average entropy in bits per parameter.
        """
        entropies: list[float] = []
        for param in field.parameters():
            data = param.data.detach().float().flatten()
            if data.numel() < 2:
                continue

            # Histogram with 256 bins
            hist = torch.histc(data, bins=256)
            # Normalize to probability distribution
            probs = hist / hist.sum()
            # Remove zeros to avoid log(0)
            probs = probs[probs > 0]
            # Shannon entropy in bits
            entropy = -(probs * probs.log2()).sum().item()
            entropies.append(entropy)

        if not entropies:
            return 0.0

        avg = sum(entropies) / len(entropies)
        logger.info(
            "Information content: %.2f bits/param (across %d tensors)",
            avg,
            len(entropies),
        )
        return avg
