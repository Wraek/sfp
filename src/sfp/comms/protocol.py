"""Communication protocol: Message format and CommEndpoint."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sfp.comms.compression import GradientCompressor
from sfp.comms.layers import L0RawText, L1Embedding, L2ManifoldCoord, L3Deformation, L4SurpriseGated
from sfp.config import CommConfig
from sfp.core.attractors import AttractorQuery
from sfp.core.field import SemanticFieldProcessor
from sfp.types import CommLayer
from sfp.utils.logging import get_logger

logger = get_logger("comms.protocol")


@dataclass
class Message:
    """A communication message between endpoints."""

    layer: CommLayer
    payload: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    sender_id: str = ""
    sequence_num: int = 0
    timestamp: float = 0.0

    def size_bytes(self) -> int:
        """Total payload size in bytes."""
        return len(self.payload)

    def compression_ratio(self, original_size: int) -> float:
        """Compression ratio relative to an original size."""
        return original_size / max(1, len(self.payload))


class CommEndpoint:
    """Communication endpoint for a semantic field.

    Manages encoding/decoding at all protocol layers, peer tracking,
    and attractor codebook maintenance.
    """

    def __init__(
        self,
        field: SemanticFieldProcessor,
        endpoint_id: str,
        config: CommConfig | None = None,
    ) -> None:
        self.field = field
        self.endpoint_id = endpoint_id
        self.config = config or CommConfig()

        # Layer handlers
        self._l0 = L0RawText()
        self._l1 = L1Embedding()
        self._l2 = L2ManifoldCoord(field)
        self._l3 = L3Deformation(
            GradientCompressor(
                method=self.config.compression_method,
                density=self.config.compression_density,
            )
        )
        self._l4 = L4SurpriseGated(field)

        # State
        self._attractor_codebook: torch.Tensor | None = None
        self._peer_fields: dict[str, SemanticFieldProcessor] = {}
        self._sequence_counter: int = 0

    def build_codebook(
        self, n_probes: int = 1000, merge_radius: float = 0.1
    ) -> int:
        """Discover attractors and build the L2 codebook.

        Args:
            n_probes: Number of random probes for attractor discovery.
            merge_radius: Distance threshold for merging nearby attractors.

        Returns:
            Number of discovered attractors.
        """
        query = AttractorQuery(self.field)
        self._attractor_codebook = query.discover_attractors(
            n_probes=n_probes, merge_radius=merge_radius
        )
        self._l2.set_codebook(self._attractor_codebook)
        count = len(self._attractor_codebook)
        logger.info("Built codebook with %d attractors", count)
        return count

    def encode(
        self,
        concept: torch.Tensor,
        layer: CommLayer | None = None,
        peer_id: str | None = None,
    ) -> Message:
        """Encode a concept for transmission.

        Args:
            concept: Concept tensor of shape (dim,) or (1, dim).
            layer: Protocol layer to use. Defaults to config.preferred_layer.
            peer_id: Optional peer ID for L4 surprise gating.

        Returns:
            Encoded Message.
        """
        layer = layer or self.config.preferred_layer
        self._sequence_counter += 1

        if layer == CommLayer.L0_RAW_TEXT:
            raise ValueError("L0 requires text input, not tensor. Use encode_text().")

        elif layer == CommLayer.L1_EMBEDDING:
            payload = self._l1.encode(concept)

        elif layer == CommLayer.L2_MANIFOLD_COORD:
            if self._attractor_codebook is None:
                self.build_codebook()
            payload = self._l2.encode(concept)

        elif layer == CommLayer.L3_DEFORMATION:
            raise ValueError(
                "L3 encodes weight deltas, not concepts. Use encode_deformation()."
            )

        elif layer == CommLayer.L4_SURPRISE_GATED:
            if peer_id is None or peer_id not in self._peer_fields:
                # Fall back to L2
                return self.encode(concept, layer=CommLayer.L2_MANIFOLD_COORD)

            peer_field = self._peer_fields[peer_id]
            payload = self._l4.encode(
                concept,
                peer_field,
                self._l2 if self._attractor_codebook is not None else self._l1,
                threshold=self.config.surprise_threshold,
            )
            if payload is None:
                # Not surprising — empty payload signals "already known"
                payload = b""

        else:
            raise ValueError(f"Unknown layer: {layer}")

        return Message(
            layer=layer,
            payload=payload,
            sender_id=self.endpoint_id,
            sequence_num=self._sequence_counter,
            timestamp=time.monotonic(),
        )

    def encode_text(self, text: str) -> Message:
        """Encode raw text at L0.

        Args:
            text: Text string to encode.

        Returns:
            L0 Message.
        """
        self._sequence_counter += 1
        return Message(
            layer=CommLayer.L0_RAW_TEXT,
            payload=self._l0.encode(text),
            sender_id=self.endpoint_id,
            sequence_num=self._sequence_counter,
            timestamp=time.monotonic(),
        )

    def encode_deformation(
        self, weight_delta: dict[str, torch.Tensor]
    ) -> Message:
        """Encode weight deltas at L3.

        Args:
            weight_delta: Dict mapping param name -> delta tensor.

        Returns:
            L3 Message.
        """
        self._sequence_counter += 1
        payload = self._l3.encode(weight_delta)
        return Message(
            layer=CommLayer.L3_DEFORMATION,
            payload=payload,
            sender_id=self.endpoint_id,
            sequence_num=self._sequence_counter,
            timestamp=time.monotonic(),
        )

    def decode(self, message: Message) -> torch.Tensor | str | None:
        """Decode a received message.

        Args:
            message: Incoming Message.

        Returns:
            Decoded content: str for L0, Tensor for L1/L2, None for empty L4.
        """
        if message.layer == CommLayer.L0_RAW_TEXT:
            return self._l0.decode(message.payload)

        elif message.layer == CommLayer.L1_EMBEDDING:
            return self._l1.decode(message.payload)

        elif message.layer == CommLayer.L2_MANIFOLD_COORD:
            return self._l2.decode(message.payload)

        elif message.layer == CommLayer.L3_DEFORMATION:
            # Apply deltas to local field
            self._l3.apply_to_field(self.field, message.payload)
            return None

        elif message.layer == CommLayer.L4_SURPRISE_GATED:
            if len(message.payload) == 0:
                return None  # Not surprising — nothing new
            # Decode with fallback layer (L2 or L1)
            if self._attractor_codebook is not None:
                return self._l2.decode(message.payload)
            return self._l1.decode(message.payload)

        raise ValueError(f"Unknown layer: {message.layer}")

    def apply_deformation(self, message: Message) -> None:
        """Apply an L3 deformation message to the local field.

        Args:
            message: L3 Message containing weight deltas.
        """
        if message.layer != CommLayer.L3_DEFORMATION:
            raise ValueError("Expected L3 message")
        self._l3.apply_to_field(self.field, message.payload)

    def register_peer(self, peer_id: str, peer_field: SemanticFieldProcessor) -> None:
        """Register a peer's field estimate for L4 surprise gating.

        Args:
            peer_id: Unique peer identifier.
            peer_field: Clone or estimate of the peer's current field.
        """
        self._peer_fields[peer_id] = peer_field
        logger.info("Registered peer '%s'", peer_id)

    @property
    def codebook(self) -> torch.Tensor | None:
        """Current attractor codebook, if built."""
        return self._attractor_codebook

    @property
    def codebook_size(self) -> int:
        """Number of attractors in the codebook."""
        return len(self._attractor_codebook) if self._attractor_codebook is not None else 0
