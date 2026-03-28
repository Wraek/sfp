"""Protocol negotiation: capability exchange and layer selection."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from sfp.comms.sync import ManifoldSynchronizer
from sfp.core.field import SemanticFieldProcessor
from sfp.types import CommLayer
from sfp.utils.logging import get_logger

logger = get_logger("comms.negotiation")


@dataclass
class PeerState:
    """Tracked state for a communication peer."""

    peer_id: str
    field_estimate: SemanticFieldProcessor
    last_fingerprint: torch.Tensor | None = None
    last_sync_time: float = 0.0
    messages_sent: int = 0
    messages_received: int = 0


class ProtocolNegotiator:
    """Negotiates the optimal communication layer with a peer.

    Considers manifold drift, peer capabilities, and bandwidth
    to select the most efficient protocol layer.
    """

    def __init__(
        self,
        field: SemanticFieldProcessor,
        endpoint_id: str,
        supported_layers: set[CommLayer] | None = None,
    ) -> None:
        self.field = field
        self.endpoint_id = endpoint_id
        self.supported_layers = supported_layers or {
            CommLayer.L0_RAW_TEXT,
            CommLayer.L1_EMBEDDING,
            CommLayer.L2_MANIFOLD_COORD,
            CommLayer.L3_DEFORMATION,
            CommLayer.L4_SURPRISE_GATED,
        }
        self._synchronizer = ManifoldSynchronizer(field)

    def handshake(
        self,
        remote_fingerprint: torch.Tensor,
        remote_capabilities: set[CommLayer],
    ) -> CommLayer:
        """Negotiate the best communication layer with a peer.

        Considers drift magnitude and shared capabilities to select
        the most efficient layer both sides support.

        Args:
            remote_fingerprint: Peer's manifold fingerprint.
            remote_capabilities: Set of CommLayers the peer supports.

        Returns:
            The agreed-upon CommLayer.
        """
        local_fp = self._synchronizer.compute_fingerprint()
        drift = self._synchronizer.detect_drift(local_fp, remote_fingerprint)

        shared = self.supported_layers & remote_capabilities

        if not shared:
            raise ValueError("No shared communication layers")

        # Low drift: manifolds are aligned, can use high-efficiency layers
        if drift < 0.01:
            if CommLayer.L4_SURPRISE_GATED in shared:
                logger.info(
                    "Handshake: L4 (drift=%.4f, manifolds aligned)", drift
                )
                return CommLayer.L4_SURPRISE_GATED
            if CommLayer.L2_MANIFOLD_COORD in shared:
                logger.info(
                    "Handshake: L2 (drift=%.4f, manifolds aligned)", drift
                )
                return CommLayer.L2_MANIFOLD_COORD

        # Moderate drift: need more information per message
        if drift < 0.1:
            if CommLayer.L2_MANIFOLD_COORD in shared:
                logger.info(
                    "Handshake: L2 (drift=%.4f, moderate drift)", drift
                )
                return CommLayer.L2_MANIFOLD_COORD
            if CommLayer.L1_EMBEDDING in shared:
                logger.info(
                    "Handshake: L1 (drift=%.4f, moderate drift)", drift
                )
                return CommLayer.L1_EMBEDDING

        # High drift: fall back to low-compression layers
        if CommLayer.L1_EMBEDDING in shared:
            logger.info("Handshake: L1 (drift=%.4f, high drift)", drift)
            return CommLayer.L1_EMBEDDING

        logger.info("Handshake: L0 fallback (drift=%.4f)", drift)
        return CommLayer.L0_RAW_TEXT

    def create_capability_message(self) -> dict[str, Any]:
        """Create a capability announcement for peer discovery.

        Returns:
            Dict with endpoint info, supported layers, and fingerprint.
        """
        from dataclasses import asdict

        return {
            "endpoint_id": self.endpoint_id,
            "supported_layers": [layer.value for layer in self.supported_layers],
            "fingerprint": self._synchronizer.compute_fingerprint().tolist(),
            "field_config": asdict(self.field.config),
        }
