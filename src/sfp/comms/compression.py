"""Gradient compression: TopK sparsification and SignSGD with error accumulation."""

from __future__ import annotations

import math

import torch

from sfp.types import CompressedDeltas
from sfp.utils.logging import get_logger

logger = get_logger("comms.compression")


class GradientCompressor:
    """Compresses weight deltas for efficient L3 communication.

    Supports TopK sparsification and SignSGD, both with error feedback
    (accumulated residuals from previous compressions are added to the
    next round, preventing information loss over time).
    """

    def __init__(self, method: str = "topk", density: float = 0.01) -> None:
        self.method = method
        self.density = density
        self._error_buffer: dict[str, torch.Tensor] = {}

    def compress(self, deltas: dict[str, torch.Tensor]) -> CompressedDeltas:
        """Compress weight deltas.

        Args:
            deltas: Dict mapping param name -> delta tensor.

        Returns:
            CompressedDeltas with sparse representation.
        """
        if self.method == "topk":
            return self._compress_topk(deltas)
        elif self.method == "signsgd":
            return self._compress_signsgd(deltas)
        else:
            raise ValueError(f"Unknown compression method: {self.method}")

    def decompress(self, compressed: CompressedDeltas) -> dict[str, torch.Tensor]:
        """Decompress back to full-size tensors.

        Args:
            compressed: CompressedDeltas from compress().

        Returns:
            Dict mapping param name -> reconstructed delta tensor.
        """
        if compressed.method == "topk":
            return self._decompress_topk(compressed)
        elif compressed.method == "signsgd":
            return self._decompress_signsgd(compressed)
        else:
            raise ValueError(f"Unknown method: {compressed.method}")

    def reset_error_buffers(self) -> None:
        """Clear accumulated error buffers."""
        self._error_buffer.clear()

    def _compress_topk(self, deltas: dict[str, torch.Tensor]) -> CompressedDeltas:
        """TopK sparsification: keep only the k largest-magnitude values."""
        indices: dict[str, torch.Tensor] = {}
        values: dict[str, torch.Tensor] = {}
        shapes: dict[str, tuple[int, ...]] = {}

        for name, delta in deltas.items():
            shapes[name] = tuple(delta.shape)
            flat = delta.flatten()

            # Add accumulated error
            if name in self._error_buffer:
                flat = flat + self._error_buffer[name]

            k = max(1, math.ceil(self.density * flat.numel()))
            top_vals, top_idx = torch.topk(flat.abs(), k)

            # Store actual values (with sign) at those indices
            actual_vals = flat[top_idx]
            indices[name] = top_idx
            values[name] = actual_vals

            # Compute residual error and accumulate
            reconstructed = torch.zeros_like(flat)
            reconstructed[top_idx] = actual_vals
            self._error_buffer[name] = flat - reconstructed

        return CompressedDeltas(
            indices=indices,
            values=values,
            shapes=shapes,
            method="topk",
            metadata={"density": self.density},
        )

    def _decompress_topk(self, compressed: CompressedDeltas) -> dict[str, torch.Tensor]:
        """Reconstruct full tensors from TopK sparse representation."""
        result: dict[str, torch.Tensor] = {}
        for name in compressed.shapes:
            shape = compressed.shapes[name]
            numel = 1
            for s in shape:
                numel *= s

            flat = torch.zeros(numel, dtype=compressed.values[name].dtype,
                               device=compressed.values[name].device)
            flat[compressed.indices[name]] = compressed.values[name]
            result[name] = flat.reshape(shape)
        return result

    def _compress_signsgd(self, deltas: dict[str, torch.Tensor]) -> CompressedDeltas:
        """SignSGD: store only signs and per-tensor scale."""
        indices: dict[str, torch.Tensor] = {}
        values: dict[str, torch.Tensor] = {}
        shapes: dict[str, tuple[int, ...]] = {}

        for name, delta in deltas.items():
            shapes[name] = tuple(delta.shape)
            flat = delta.flatten()

            # Add accumulated error
            if name in self._error_buffer:
                flat = flat + self._error_buffer[name]

            scale = flat.abs().mean()
            signs = flat.sign().to(torch.int8)

            # Store signs as "indices" and scale as single "value"
            indices[name] = signs
            values[name] = scale.unsqueeze(0)

            # Accumulate error
            reconstructed = signs.float() * scale
            self._error_buffer[name] = flat - reconstructed

        return CompressedDeltas(
            indices=indices,
            values=values,
            shapes=shapes,
            method="signsgd",
        )

    def _decompress_signsgd(self, compressed: CompressedDeltas) -> dict[str, torch.Tensor]:
        """Reconstruct from sign + scale representation."""
        result: dict[str, torch.Tensor] = {}
        for name in compressed.shapes:
            shape = compressed.shapes[name]
            signs = compressed.indices[name].float()
            scale = compressed.values[name].item()
            result[name] = (signs * scale).reshape(shape)
        return result
