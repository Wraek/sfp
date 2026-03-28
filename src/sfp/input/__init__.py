"""Pluggable input layer: encoders, embedding adapters, byte-level encoder, projection."""

from sfp.input.adapters import CLIPAdapter, PrecomputedAdapter, SentenceTransformerAdapter
from sfp.input.bytelevel import ByteLevelConfig, ByteLevelEncoder, BytePatcher, EntropyEstimator
from sfp.input.encoder import BaseInputEncoder
from sfp.input.projection import DimensionalityProjection
from sfp.input.registry import EncoderRegistry
from sfp.input.token_types import (
    DEPTH_PATCH,
    ENTITY,
    STANDARD_TOKEN_TYPES,
    STATE,
    TEMPORAL_DIFF,
    VISUAL_PATCH,
    TokenTypeSpec,
)

__all__ = [
    "BaseInputEncoder",
    "SentenceTransformerAdapter",
    "CLIPAdapter",
    "PrecomputedAdapter",
    "ByteLevelConfig",
    "ByteLevelEncoder",
    "BytePatcher",
    "EntropyEstimator",
    "DimensionalityProjection",
    "EncoderRegistry",
    "TokenTypeSpec",
    "VISUAL_PATCH",
    "ENTITY",
    "STATE",
    "DEPTH_PATCH",
    "TEMPORAL_DIFF",
    "STANDARD_TOKEN_TYPES",
]
