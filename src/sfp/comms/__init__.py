"""Communication protocol: L0-L4 layers, compression, sync, negotiation."""

from sfp.comms.compression import GradientCompressor
from sfp.comms.layers import L0RawText, L1Embedding, L2ManifoldCoord, L3Deformation, L4SurpriseGated
from sfp.comms.negotiation import PeerState, ProtocolNegotiator
from sfp.comms.protocol import CommEndpoint, Message
from sfp.comms.sync import ManifoldSynchronizer

__all__ = [
    "GradientCompressor",
    "L0RawText",
    "L1Embedding",
    "L2ManifoldCoord",
    "L3Deformation",
    "L4SurpriseGated",
    "CommEndpoint",
    "Message",
    "ManifoldSynchronizer",
    "PeerState",
    "ProtocolNegotiator",
]
