"""Core semantic field processing: manifold, streaming, attractors, forgetting, neural arch."""

from sfp.core.attractors import AttractorQuery
from sfp.core.backbone import BackboneTransformer
from sfp.core.field import SemanticFieldProcessor
from sfp.core.forgetting import EWCStrategy, WeightDecayStrategy
from sfp.core.lora import LoRALinear, OnlineLoRAManager
from sfp.core.perceiver import PerceiverIO
from sfp.core.streaming import StreamingProcessor

__all__ = [
    "SemanticFieldProcessor",
    "StreamingProcessor",
    "AttractorQuery",
    "LoRALinear",
    "OnlineLoRAManager",
    "EWCStrategy",
    "WeightDecayStrategy",
    "PerceiverIO",
    "BackboneTransformer",
]
