"""Tests for topology.betti — BettiNumberMonitor.

These tests use a mock PersistentHomologyTracker since giotto-tda may not be
available.
"""

import pytest

from sfp.types import TopologySnapshot


class _MockSnapshot:
    def __init__(self, betti: tuple[int, ...]):
        self.betti_numbers = betti


class _MockTracker:
    def __init__(self, max_dimension: int = 2):
        self.max_dimension = max_dimension
        self.history: list[_MockSnapshot] = []

    def add_snapshot(self, betti: tuple[int, ...]):
        self.history.append(_MockSnapshot(betti))


# Import after defining mock to avoid giotto-tda dependency
try:
    from sfp.topology.betti import BettiNumberMonitor
    _HAS_BETTI = True
except ImportError:
    _HAS_BETTI = False


@pytest.mark.skipif(not _HAS_BETTI, reason="BettiNumberMonitor requires giotto-tda transitive dep")
class TestBettiNumberMonitor:
    def test_current_betti_empty(self):
        tracker = _MockTracker()
        monitor = BettiNumberMonitor(tracker)
        betti = monitor.current_betti()
        assert betti == (0, 0, 0)

    def test_current_betti_with_history(self):
        tracker = _MockTracker()
        tracker.add_snapshot((3, 1, 0))
        tracker.add_snapshot((5, 2, 1))
        monitor = BettiNumberMonitor(tracker)
        assert monitor.current_betti() == (5, 2, 1)

    def test_betti_series(self):
        tracker = _MockTracker()
        tracker.add_snapshot((1, 0, 0))
        tracker.add_snapshot((2, 1, 0))
        tracker.add_snapshot((3, 1, 1))
        monitor = BettiNumberMonitor(tracker)
        series = monitor.betti_series()
        assert series[0] == [1, 2, 3]
        assert series[1] == [0, 1, 1]

    def test_is_stable_true(self):
        tracker = _MockTracker()
        for _ in range(5):
            tracker.add_snapshot((3, 1, 0))
        monitor = BettiNumberMonitor(tracker)
        assert monitor.is_stable(window=5)

    def test_is_stable_false(self):
        tracker = _MockTracker()
        tracker.add_snapshot((1, 0, 0))
        tracker.add_snapshot((2, 0, 0))
        tracker.add_snapshot((3, 0, 0))
        tracker.add_snapshot((4, 0, 0))
        tracker.add_snapshot((5, 0, 0))
        monitor = BettiNumberMonitor(tracker)
        assert not monitor.is_stable(window=5, tolerance=0)

    def test_is_stable_insufficient_data(self):
        tracker = _MockTracker()
        tracker.add_snapshot((1, 0, 0))
        monitor = BettiNumberMonitor(tracker)
        assert not monitor.is_stable(window=5)
