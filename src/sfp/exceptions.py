"""SFP exception hierarchy."""

from __future__ import annotations


class SFPError(Exception):
    """Base exception for all SFP errors."""


class ConvergenceError(SFPError):
    """Attractor query did not converge within max_iterations."""


class ManifoldDriftError(SFPError):
    """Manifolds have diverged beyond the synchronization threshold."""


class QuantizationError(SFPError):
    """Quantization caused unacceptable accuracy loss."""


class AdapterNotFoundError(SFPError):
    """Requested embedding adapter is not registered."""


class MemoryTierError(SFPError):
    """Error in memory tier operations (admission, eviction, promotion)."""


class ConsolidationError(SFPError):
    """Error during knowledge consolidation between tiers."""


class IntegrityError(SFPError):
    """Integrity check failure (hash mismatch, corruption detected)."""


class PoisoningDetectedError(SFPError):
    """Potential poisoning attack detected by the defense framework."""


class ReasoningChainError(SFPError):
    """Error during reasoning chain traversal (cycle, dead-end, etc.)."""


class WorldModelError(SFPError):
    """Error in world model prediction or training."""


class GoalError(SFPError):
    """Error in goal management (creation, decomposition, serialization)."""


class MetacognitionError(SFPError):
    """Error in uncertainty estimation or calibration."""


class SalienceError(SFPError):
    """Error in salience gate processing."""


class GenerativeReplayError(SFPError):
    """Error in synthetic episode generation or validation."""
