"""Defense Layer 1-2: Input sanitization and embedding-space anomaly detection.

Implements input validation, provenance tracking, and Mahalanobis-distance
anomaly detection from the poisoning defense document (Layers 1-2).
"""

from __future__ import annotations

import hashlib
import time
from collections import deque

import torch

from sfp.config import DefenseConfig
from sfp.utils.logging import get_logger

logger = get_logger("defense.input")


class InputSanitizer:
    """Defense Layer 1: Input validation, smoothing, and provenance tracking.

    Applies lightweight preprocessing to reduce adversarial signal strength:
      - L2 norm clamping (prevents outsized activations)
      - Gaussian smoothing (degrades high-frequency adversarial features)
      - Provenance tracking (records input hash, modality, timestamp)
    """

    def __init__(
        self,
        max_norm: float = 10.0,
        smoothing_sigma: float = 0.01,
        provenance_log_size: int = 10000,
    ) -> None:
        self._max_norm = max_norm
        self._smoothing_sigma = smoothing_sigma
        self._provenance_log: deque[tuple[bytes, str, float]] = deque(maxlen=provenance_log_size)

    def sanitize(self, input_tensor: torch.Tensor, modality: str = "tensor") -> torch.Tensor:
        """Sanitize an input tensor before it enters the processing pipeline.

        Args:
            input_tensor: Raw input tensor.
            modality: Input modality identifier.

        Returns:
            Sanitized tensor with norm clamping and optional smoothing applied.
        """
        with torch.no_grad():
            # L2 norm clamping: prevent outsized activations
            norm = input_tensor.norm(dim=-1, keepdim=True)
            scale = torch.clamp(self._max_norm / (norm + 1e-8), max=1.0)
            clamped = input_tensor * scale

            # Gaussian smoothing: add small noise and average
            # This degrades high-frequency adversarial perturbations
            if self._smoothing_sigma > 0:
                noise = torch.randn_like(clamped) * self._smoothing_sigma
                smoothed = clamped + noise
            else:
                smoothed = clamped

        return smoothed

    def record_provenance(self, raw_bytes: bytes, modality: str, source: str = "") -> bytes:
        """Record the provenance of an input for retrospective analysis.

        Args:
            raw_bytes: Raw bytes of the input data.
            modality: Input modality (e.g., "text", "image", "audio").
            source: Source identifier.

        Returns:
            SHA-256 hash of the input.
        """
        hasher = hashlib.sha256()
        hasher.update(raw_bytes)
        hasher.update(modality.encode())
        hasher.update(source.encode())
        input_hash = hasher.digest()

        self._provenance_log.append((input_hash, modality, time.time()))
        return input_hash

    @property
    def provenance_log_size(self) -> int:
        return len(self._provenance_log)


class EmbeddingAnomalyDetector:
    """Defense Layer 2: Embedding-space anomaly detection via Mahalanobis distance.

    Maintains running statistics (mean, covariance) of embeddings per modality
    and flags inputs whose Mahalanobis distance exceeds a threshold.

    This catches both out-of-distribution inputs and adversarially crafted samples
    that produce anomalous activations.

    Uses diagonal covariance (per-dimension variance) rather than full
    covariance to avoid rank-deficiency issues in high dimensions where
    d_model >> number of warmup samples.

    Args:
        d_model: Embedding dimensionality.
        threshold: RMS z-score threshold for anomaly detection (dimension-independent).
        warmup_samples: Minimum samples before detection activates.
    """

    def __init__(
        self,
        d_model: int = 512,
        threshold: float = 3.0,
        warmup_samples: int = 100,
    ) -> None:
        self._d_model = d_model
        self._threshold = threshold
        self._warmup = warmup_samples

        # Per-modality statistics (diagonal: O(d) instead of O(d²))
        self._counts: dict[str, int] = {}
        self._means: dict[str, torch.Tensor] = {}
        self._m2: dict[str, torch.Tensor] = {}  # Welford M2 for variance

    def update_statistics(self, embedding: torch.Tensor, modality: str) -> None:
        """Update running statistics with a new embedding.

        Uses Welford's online algorithm for numerically stable mean and variance.

        Args:
            embedding: (d_model,) embedding vector.
            modality: Modality identifier.
        """
        emb = embedding.detach().float()
        if emb.dim() > 1:
            emb = emb.mean(dim=0)

        device = emb.device

        if modality not in self._counts:
            self._counts[modality] = 0
            self._means[modality] = torch.zeros(self._d_model, device=device)
            self._m2[modality] = torch.zeros(self._d_model, device=device)

        self._counts[modality] += 1
        n = self._counts[modality]

        # Welford's online mean + variance update
        delta = emb - self._means[modality]
        self._means[modality] = self._means[modality] + delta / n
        delta2 = emb - self._means[modality]
        self._m2[modality] = self._m2[modality] + delta * delta2

    def is_anomalous(self, embedding: torch.Tensor, modality: str) -> bool:
        """Test whether an embedding is anomalous relative to the learned distribution.

        Computes the RMS z-score (root-mean-square of per-dimension z-scores),
        which has expected value ~1.0 for in-distribution data regardless of
        dimensionality, making the threshold dimension-independent.

        Args:
            embedding: (d_model,) embedding vector.
            modality: Modality identifier.

        Returns:
            True if the RMS z-score exceeds the threshold.
        """
        if modality not in self._counts or self._counts[modality] < self._warmup:
            return False  # Not enough data to judge

        emb = embedding.detach().float()
        if emb.dim() > 1:
            emb = emb.mean(dim=0)

        mean = self._means[modality]
        n = self._counts[modality]

        # Per-dimension variance with regularization
        var = self._m2[modality] / max(n - 1, 1)
        var_reg = var + 1e-4

        # RMS z-score: sqrt(mean_i((x_i - mu_i)^2 / var_i))
        z_sq = ((emb - mean) ** 2) / var_reg
        mahal_dist = z_sq.mean().item() ** 0.5

        if mahal_dist > self._threshold:
            logger.warning(
                "Anomalous embedding detected: modality=%s, mahal_dist=%.2f (threshold=%.2f)",
                modality, mahal_dist, self._threshold,
            )
            return True

        return False

    def get_statistics(self, modality: str) -> dict:
        """Get current statistics for a modality."""
        if modality not in self._counts:
            return {"count": 0}
        return {
            "count": self._counts[modality],
            "mean_norm": self._means[modality].norm().item(),
            "warmup_complete": self._counts[modality] >= self._warmup,
        }
