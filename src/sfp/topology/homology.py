"""Persistent homology tracking for manifold topology analysis."""

from __future__ import annotations

import time

import numpy as np
import torch

from sfp.core.field import SemanticFieldProcessor
from sfp.types import TopologicalEvent, TopologySnapshot
from sfp.utils.logging import get_logger

logger = get_logger("topology.homology")


class PersistentHomologyTracker:
    """Tracks topological features of the manifold over time.

    Uses Vietoris-Rips persistent homology (via giotto-tda) to compute
    Betti numbers and persistence diagrams from field activations.
    Detects topological events (births/deaths of features) by comparing
    successive snapshots.

    Requires giotto-tda: pip install sfp[topology]
    """

    def __init__(
        self, max_dimension: int = 2, n_sample_points: int = 500
    ) -> None:
        self.max_dimension = max_dimension
        self.n_sample_points = n_sample_points
        self._history: list[TopologySnapshot] = []

    @property
    def history(self) -> list[TopologySnapshot]:
        """All recorded topology snapshots."""
        return self._history

    def snapshot(
        self,
        field: SemanticFieldProcessor,
        sample_points: torch.Tensor | None = None,
    ) -> TopologySnapshot:
        """Compute a topological snapshot of the manifold.

        Args:
            field: The SemanticFieldProcessor to analyze.
            sample_points: Optional sample points. If None, generates
                random points from N(0,1).

        Returns:
            TopologySnapshot with Betti numbers and persistence diagram.
        """
        try:
            from gtda.homology import VietorisRipsPersistence
        except ImportError as e:
            raise ImportError(
                "giotto-tda is required for topology features. "
                "Install with: pip install sfp[topology]"
            ) from e

        device = next(field.parameters()).device
        dim = field.config.dim

        if sample_points is None:
            sample_points = torch.randn(
                self.n_sample_points, dim, device=device
            )

        # Forward pass through field
        field.eval()
        with torch.no_grad():
            activations = field(sample_points).cpu().numpy()

        # Compute persistent homology
        vr = VietorisRipsPersistence(
            homology_dimensions=list(range(self.max_dimension + 1)),
            max_edge_length=np.inf,
        )

        # VR expects 3D array: (n_samples, n_points, n_features)
        diagrams = vr.fit_transform(activations[np.newaxis, :, :])[0]

        # Compute Betti numbers: count features alive at median death scale
        betti = self._compute_betti(diagrams)

        # Total persistence: sum of (death - birth) for all finite features
        finite_mask = np.isfinite(diagrams[:, 1])
        total_persistence = float(
            np.sum(diagrams[finite_mask, 1] - diagrams[finite_mask, 0])
        )

        snap = TopologySnapshot(
            timestamp=time.monotonic(),
            betti_numbers=betti,
            persistence_diagram=diagrams,
            total_persistence=total_persistence,
        )
        self._history.append(snap)
        logger.info(
            "Topology snapshot: betti=%s, total_persistence=%.4f",
            betti,
            total_persistence,
        )
        return snap

    def detect_changes(self, threshold: float = 0.1) -> list[TopologicalEvent]:
        """Detect topological changes between the last two snapshots.

        Args:
            threshold: Minimum significance for reporting an event.

        Returns:
            List of TopologicalEvent objects.
        """
        if len(self._history) < 2:
            return []

        prev = self._history[-2]
        curr = self._history[-1]
        events: list[TopologicalEvent] = []

        max_dim = min(len(prev.betti_numbers), len(curr.betti_numbers))
        for dim in range(max_dim):
            diff = curr.betti_numbers[dim] - prev.betti_numbers[dim]
            if diff == 0:
                continue

            prev_val = max(1, prev.betti_numbers[dim])
            significance = abs(diff) / prev_val

            if significance >= threshold:
                event_type = "birth" if diff > 0 else "death"
                events.append(
                    TopologicalEvent(
                        event_type=event_type,
                        dimension=dim,
                        significance=significance,
                    )
                )

        if events:
            logger.info("Detected %d topological events", len(events))
        return events

    def _compute_betti(self, diagrams: np.ndarray) -> tuple[int, ...]:
        """Compute Betti numbers from persistence diagrams.

        Uses the median death time as the threshold for counting
        features that are "alive" at that scale.
        """
        betti: list[int] = []
        for dim in range(self.max_dimension + 1):
            # Filter to this dimension
            dim_mask = diagrams[:, 2] == dim
            dim_diag = diagrams[dim_mask]

            if len(dim_diag) == 0:
                betti.append(0)
                continue

            # Count features alive at median death scale
            finite_deaths = dim_diag[np.isfinite(dim_diag[:, 1]), 1]
            if len(finite_deaths) == 0:
                # All infinite features (e.g., H0 connected components)
                betti.append(len(dim_diag))
                continue

            threshold = np.median(finite_deaths)
            alive = np.sum(
                (dim_diag[:, 0] <= threshold)
                & ((dim_diag[:, 1] > threshold) | np.isinf(dim_diag[:, 1]))
            )
            betti.append(int(alive))

        return tuple(betti)
