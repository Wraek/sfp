"""Defense framework: 7-layer protection against poisoning attacks."""

from sfp.defense.anchor_verification import AnchorVerifier
from sfp.defense.gradient_bounds import AdaptiveGradientClipper, UpdateBudget
from sfp.defense.input_validation import EmbeddingAnomalyDetector, InputSanitizer
from sfp.defense.surprise_hardening import SurpriseHardener
from sfp.defense.topology_monitor import ManifoldIntegrityMonitor

__all__ = [
    "SurpriseHardener",
    "InputSanitizer",
    "EmbeddingAnomalyDetector",
    "AdaptiveGradientClipper",
    "UpdateBudget",
    "ManifoldIntegrityMonitor",
    "AnchorVerifier",
]
