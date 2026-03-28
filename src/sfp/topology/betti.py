"""Betti number monitoring for tracking manifold topology evolution."""

from __future__ import annotations

from sfp.topology.homology import PersistentHomologyTracker
from sfp.utils.logging import get_logger

logger = get_logger("topology.betti")


class BettiNumberMonitor:
    """Monitors Betti number evolution over time.

    Wraps a PersistentHomologyTracker to provide convenient access to
    Betti number time series and stability detection.
    """

    def __init__(self, tracker: PersistentHomologyTracker) -> None:
        self.tracker = tracker

    def current_betti(self) -> tuple[int, ...]:
        """Return Betti numbers from the latest snapshot.

        Returns:
            Tuple of Betti numbers (beta_0, beta_1, ...), or zeros if
            no snapshots exist.
        """
        if not self.tracker.history:
            return tuple(0 for _ in range(self.tracker.max_dimension + 1))
        return self.tracker.history[-1].betti_numbers

    def betti_series(self) -> dict[int, list[int]]:
        """Return time series of Betti numbers per dimension.

        Returns:
            Dict mapping dimension -> list of Betti numbers across snapshots.
        """
        series: dict[int, list[int]] = {
            dim: [] for dim in range(self.tracker.max_dimension + 1)
        }
        for snap in self.tracker.history:
            for dim in range(min(len(snap.betti_numbers), self.tracker.max_dimension + 1)):
                series[dim].append(snap.betti_numbers[dim])
        return series

    def is_stable(self, window: int = 5, tolerance: int = 0) -> bool:
        """Check if Betti numbers have been constant over recent snapshots.

        Args:
            window: Number of recent snapshots to check.
            tolerance: Maximum allowed Betti number deviation.

        Returns:
            True if topology has been stable.
        """
        history = self.tracker.history
        if len(history) < window:
            return False

        recent = history[-window:]
        reference = recent[0].betti_numbers

        for snap in recent[1:]:
            for dim in range(min(len(reference), len(snap.betti_numbers))):
                if abs(snap.betti_numbers[dim] - reference[dim]) > tolerance:
                    return False

        return True
