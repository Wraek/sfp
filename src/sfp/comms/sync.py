"""Manifold synchronization: drift detection via functional fingerprinting."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from sfp.core.field import SemanticFieldProcessor
from sfp.utils.logging import get_logger

logger = get_logger("comms.sync")


class ManifoldSynchronizer:
    """Detects manifold drift between endpoints via functional fingerprinting.

    Uses a fixed set of anchor points. Same inputs through same manifold
    produce same outputs — so comparing outputs reveals drift magnitude.
    """

    def __init__(
        self,
        field: SemanticFieldProcessor,
        anchor_points: torch.Tensor | None = None,
        drift_threshold: float = 0.05,
    ) -> None:
        self.field = field
        self.drift_threshold = drift_threshold

        if anchor_points is not None:
            self._anchors = anchor_points
        else:
            device = next(field.parameters()).device
            dim = field.config.dim
            # Deterministic anchors for reproducibility
            gen = torch.Generator(device="cpu").manual_seed(42)
            self._anchors = torch.randn(100, dim, generator=gen).to(device)

    @torch.no_grad()
    def compute_fingerprint(self) -> torch.Tensor:
        """Compute a functional fingerprint of the current manifold.

        Returns:
            Flattened output tensor from passing anchors through the field.
        """
        self.field.eval()
        output = self.field(self._anchors)
        return output.flatten()

    def detect_drift(
        self, local_fp: torch.Tensor, remote_fp: torch.Tensor
    ) -> float:
        """Compute drift magnitude between two fingerprints.

        Args:
            local_fp: Local manifold fingerprint.
            remote_fp: Remote manifold fingerprint.

        Returns:
            MSE between fingerprints (lower = more aligned).
        """
        return F.mse_loss(local_fp, remote_fp).item()

    def needs_sync(self, remote_fingerprint: torch.Tensor) -> bool:
        """Check if synchronization is needed.

        Args:
            remote_fingerprint: The peer's fingerprint.

        Returns:
            True if drift exceeds threshold.
        """
        local_fp = self.compute_fingerprint()
        drift = self.detect_drift(local_fp, remote_fingerprint)
        logger.debug("Drift magnitude: %.6f (threshold: %.6f)", drift, self.drift_threshold)
        return drift > self.drift_threshold

    def create_sync_payload(
        self, remote_field: SemanticFieldProcessor
    ) -> dict[str, torch.Tensor]:
        """Compute full weight delta to bring remote into exact sync.

        Args:
            remote_field: The peer's field (or estimate thereof).

        Returns:
            Dict mapping param name -> (local_param - remote_param).
        """
        deltas: dict[str, torch.Tensor] = {}
        local_params = dict(self.field.named_parameters())
        remote_params = dict(remote_field.named_parameters())

        for name in local_params:
            if name in remote_params:
                deltas[name] = local_params[name].data - remote_params[name].data

        return deltas
