"""L0-L4 communication layer implementations."""

from __future__ import annotations

import struct
from typing import Any

import torch
import torch.nn.functional as F

from sfp.comms.compression import GradientCompressor
from sfp.core.attractors import AttractorQuery
from sfp.core.field import SemanticFieldProcessor
from sfp.config import AttractorConfig
from sfp.storage.quantization import dequantize_tensor_int8, quantize_tensor_int8
from sfp.types import CompressedDeltas
from sfp.utils.logging import get_logger

logger = get_logger("comms.layers")


class L0RawText:
    """L0: Raw text encoding — no compression, baseline."""

    def encode(self, text: str) -> bytes:
        return text.encode("utf-8")

    def decode(self, data: bytes) -> str:
        return data.decode("utf-8")


class L1Embedding:
    """L1: Embedding vector encoding with optional INT8 quantization.

    Encodes embedding vectors into compact byte representations.
    Header: 2 bytes (dim as uint16) + 1 byte (dtype flag).
    """

    DTYPE_FP16 = 0
    DTYPE_INT8 = 1

    def __init__(self, quantize_to: str = "int8") -> None:
        self.quantize_to = quantize_to

    def encode(self, embedding: torch.Tensor) -> bytes:
        """Encode an embedding vector to bytes.

        Args:
            embedding: 1D tensor of shape (dim,).

        Returns:
            Packed bytes with header + data.
        """
        if embedding.dim() > 1:
            embedding = embedding.squeeze(0)

        dim = embedding.shape[0]

        if self.quantize_to == "int8":
            quantized, scale, zero_point = quantize_tensor_int8(
                embedding.unsqueeze(0), per_channel=False
            )
            quantized = quantized.squeeze(0)
            scale_f16 = scale.half().item()

            header = struct.pack("<HBe", dim, self.DTYPE_INT8, scale_f16)
            data = quantized.numpy().tobytes()
            return header + data
        else:
            # FP16
            header = struct.pack("<HB", dim, self.DTYPE_FP16)
            data = embedding.half().numpy().tobytes()
            return header + data

    def decode(self, data: bytes) -> torch.Tensor:
        """Decode bytes back to an embedding vector.

        Args:
            data: Packed bytes from encode().

        Returns:
            Float32 tensor of shape (dim,).
        """
        # Read first 3 bytes to determine format
        dim, dtype_flag = struct.unpack_from("<HB", data, 0)

        if dtype_flag == self.DTYPE_INT8:
            scale_f16 = struct.unpack_from("<e", data, 3)[0]
            offset = 5  # 2 (dim) + 1 (flag) + 2 (scale)
            import numpy as np

            quantized = torch.tensor(
                np.frombuffer(data[offset : offset + dim], dtype=np.int8).copy()
            )
            scale = torch.tensor(float(scale_f16))
            zero_point = torch.tensor(0.0)
            return dequantize_tensor_int8(quantized, scale, zero_point)
        else:
            offset = 3
            import numpy as np

            fp16_arr = np.frombuffer(data[offset : offset + dim * 2], dtype=np.float16).copy()
            return torch.tensor(fp16_arr, dtype=torch.float32)


class L2ManifoldCoord:
    """L2: Manifold coordinate encoding via attractor index + residual.

    Encodes a concept as its nearest attractor index (2 bytes) plus an
    optional INT8-quantized residual. Total: 2-34 bytes for 512d field
    (vs 2048 bytes for raw FP32).
    """

    RESIDUAL_THRESHOLD = 0.01

    def __init__(self, field: SemanticFieldProcessor) -> None:
        self._query = AttractorQuery(field, AttractorConfig())
        self._codebook: torch.Tensor | None = None

    def set_codebook(self, codebook: torch.Tensor) -> None:
        """Set the attractor codebook for encoding.

        Args:
            codebook: Tensor of shape (n_attractors, dim).
        """
        self._codebook = codebook

    def encode(self, concept: torch.Tensor) -> bytes:
        """Encode a concept vector as attractor index + optional residual.

        Args:
            concept: Tensor of shape (dim,) or (1, dim).

        Returns:
            Packed bytes: index (2 bytes) + optional residual.
        """
        if self._codebook is None:
            raise RuntimeError("Codebook not set. Call set_codebook() first.")

        if concept.dim() > 1:
            concept = concept.squeeze(0)

        # Find nearest attractor
        dists = (self._codebook - concept.unsqueeze(0)).norm(dim=-1)
        min_idx = dists.argmin().item()

        residual = concept - self._codebook[min_idx]
        residual_norm = residual.norm().item()

        # Flag bit in high bit of index
        if residual_norm < self.RESIDUAL_THRESHOLD:
            # No residual needed — set high bit
            packed_idx = min_idx | 0x8000
            return struct.pack("<H", packed_idx)
        else:
            # Include quantized residual
            header = struct.pack("<H", min_idx)
            quantized, scale, _ = quantize_tensor_int8(
                residual.unsqueeze(0), per_channel=False
            )
            scale_bytes = struct.pack("<e", scale.half().item())
            residual_bytes = quantized.squeeze(0).numpy().tobytes()
            return header + scale_bytes + residual_bytes

    def decode(self, data: bytes) -> torch.Tensor:
        """Decode bytes back to a concept vector.

        Args:
            data: Packed bytes from encode().

        Returns:
            Float32 tensor of shape (dim,).
        """
        if self._codebook is None:
            raise RuntimeError("Codebook not set. Call set_codebook() first.")

        packed_idx = struct.unpack_from("<H", data, 0)[0]

        # Check high bit for residual flag
        has_no_residual = bool(packed_idx & 0x8000)
        idx = packed_idx & 0x7FFF

        if idx >= len(self._codebook):
            raise ValueError(f"Attractor index {idx} out of codebook range")

        point = self._codebook[idx].clone()

        if not has_no_residual and len(data) > 2:
            scale_f16 = struct.unpack_from("<e", data, 2)[0]
            import numpy as np

            dim = self._codebook.shape[1]
            residual_data = np.frombuffer(data[4 : 4 + dim], dtype=np.int8).copy()
            quantized = torch.tensor(residual_data)
            scale = torch.tensor(float(scale_f16))
            zero_point = torch.tensor(0.0)
            residual = dequantize_tensor_int8(quantized, scale, zero_point)
            point = point + residual

        return point


class L3Deformation:
    """L3: Weight delta encoding via compressed gradients.

    Serializes compressed weight deltas for transmission. The receiver
    applies these deltas to their local field to absorb new knowledge.
    """

    def __init__(self, compressor: GradientCompressor) -> None:
        self.compressor = compressor

    def encode(self, weight_delta: dict[str, torch.Tensor]) -> bytes:
        """Compress and serialize weight deltas.

        Args:
            weight_delta: Dict mapping param name -> delta tensor.

        Returns:
            Serialized bytes.
        """
        compressed = self.compressor.compress(weight_delta)
        return self._serialize(compressed)

    def decode(self, data: bytes) -> dict[str, torch.Tensor]:
        """Deserialize and decompress weight deltas.

        Args:
            data: Serialized bytes from encode().

        Returns:
            Dict mapping param name -> reconstructed delta tensor.
        """
        compressed = self._deserialize(data)
        return self.compressor.decompress(compressed)

    def apply_to_field(
        self, field: SemanticFieldProcessor, data: bytes
    ) -> None:
        """Decode deltas and apply to a field's parameters.

        Args:
            field: Target field to update.
            data: Serialized compressed deltas.
        """
        deltas = self.decode(data)
        with torch.no_grad():
            for name, param in field.named_parameters():
                if name in deltas:
                    param.data += deltas[name].to(param.device)

    def _serialize(self, compressed: CompressedDeltas) -> bytes:
        """Serialize CompressedDeltas to bytes."""
        import io
        import pickle

        buf = io.BytesIO()
        # Simple pickle serialization (production would use custom format)
        pickle.dump(
            {
                "method": compressed.method,
                "indices": {k: v.cpu() for k, v in compressed.indices.items()},
                "values": {k: v.cpu() for k, v in compressed.values.items()},
                "shapes": compressed.shapes,
                "metadata": compressed.metadata,
            },
            buf,
        )
        return buf.getvalue()

    def _deserialize(self, data: bytes) -> CompressedDeltas:
        """Deserialize bytes to CompressedDeltas."""
        import io
        import pickle

        buf = io.BytesIO(data)
        d = pickle.load(buf)  # noqa: S301
        return CompressedDeltas(
            indices=d["indices"],
            values=d["values"],
            shapes=d["shapes"],
            method=d["method"],
            metadata=d.get("metadata", {}),
        )


class L4SurpriseGated:
    """L4: Surprise-gated communication — only transmit what the peer doesn't know.

    Evaluates whether a concept is surprising to the peer's estimated
    manifold. If the peer already "knows" the concept (low reconstruction
    error), suppress the transmission.
    """

    def __init__(self, field: SemanticFieldProcessor) -> None:
        self.field = field

    def should_transmit(
        self,
        concept: torch.Tensor,
        peer_field: SemanticFieldProcessor,
        threshold: float = 0.1,
    ) -> bool:
        """Check if a concept is surprising to the peer.

        Args:
            concept: The concept tensor.
            peer_field: Estimated state of the peer's field.
            threshold: MSE threshold above which the concept is "surprising".

        Returns:
            True if the concept should be transmitted.
        """
        peer_field.eval()
        with torch.no_grad():
            output = peer_field(concept.unsqueeze(0) if concept.dim() == 1 else concept)
            target = concept.unsqueeze(0) if concept.dim() == 1 else concept
            loss = F.mse_loss(output, target).item()
        return loss > threshold

    def encode(
        self,
        concept: torch.Tensor,
        peer_field: SemanticFieldProcessor,
        fallback_layer: Any,
        threshold: float = 0.1,
    ) -> bytes | None:
        """Encode concept only if surprising to peer.

        Args:
            concept: The concept tensor.
            peer_field: Estimated state of the peer's field.
            fallback_layer: Layer to use for actual encoding (e.g., L2 or L3).
            threshold: Surprise threshold.

        Returns:
            Encoded bytes, or None if not surprising.
        """
        if not self.should_transmit(concept, peer_field, threshold):
            return None
        return fallback_layer.encode(concept)
