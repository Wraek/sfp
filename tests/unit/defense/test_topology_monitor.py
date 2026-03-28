"""Tests for defense.topology_monitor — ManifoldIntegrityMonitor."""

import torch
import pytest

from sfp.config import DefenseConfig, Tier2Config, TransitionConfig
from sfp.defense.topology_monitor import ManifoldIntegrityMonitor
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import RelationType


def _make_tier2(d: int = 32, n_slots: int = 16) -> EssentialMemory:
    tier2 = EssentialMemory(Tier2Config(n_slots=n_slots, d_value=d), d_model=d)
    with torch.no_grad():
        tier2.query_proj.weight.copy_(torch.eye(d))
        tier2.key_proj.weight.copy_(torch.eye(d))
    return tier2


class TestManifoldIntegrityMonitor:
    def test_no_alerts_with_distinct_basins(self):
        d = 32
        tier2 = _make_tier2(d)
        # Create well-separated basins
        for i in range(4):
            k = torch.zeros(d)
            k[i] = 5.0
            tier2.allocate_slot(k)

        monitor = ManifoldIntegrityMonitor()
        alerts = monitor.check_basin_integrity(tier2)
        merge_alerts = [a for a in alerts if "merge" in a.lower()]
        assert len(merge_alerts) == 0

    def test_merge_alert_with_similar_basins(self):
        d = 32
        tier2 = _make_tier2(d)
        # Create nearly identical basins
        k = torch.randn(d)
        k = k / k.norm()
        tier2.allocate_slot(k * 5.0)
        tier2.allocate_slot(k * 5.0 + torch.randn(d) * 0.001)

        monitor = ManifoldIntegrityMonitor()
        alerts = monitor.check_basin_integrity(tier2)
        assert any("merge" in a.lower() for a in alerts)

    def test_component_change_alert(self):
        d = 32
        tier2 = _make_tier2(d)
        monitor = ManifoldIntegrityMonitor(
            DefenseConfig(topology_change_threshold=0)
        )
        # First check with 2 distinct basins
        for i in range(2):
            k = torch.zeros(d)
            k[i] = 5.0
            tier2.allocate_slot(k)
        monitor.check_basin_integrity(tier2)

        # Add 3 more very different basins to change component count
        for i in range(2, 5):
            k = torch.zeros(d)
            k[i] = 5.0
            tier2.allocate_slot(k)
        alerts = monitor.check_basin_integrity(tier2)
        component_alerts = [a for a in alerts if "Component count" in a]
        assert len(component_alerts) > 0

    def test_single_basin_no_alert(self):
        d = 32
        tier2 = _make_tier2(d)
        k = torch.randn(d)
        tier2.allocate_slot(k)
        monitor = ManifoldIntegrityMonitor()
        alerts = monitor.check_basin_integrity(tier2)
        assert len(alerts) == 0

    def test_detect_reasoning_loops_empty_graph(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        monitor = ManifoldIntegrityMonitor()
        cycles = monitor.detect_reasoning_loops(transitions)
        assert cycles == []

    def test_detect_reasoning_loops_with_cycle(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        # Create cycle: 0 -> 1 -> 2 -> 0
        transitions.add_edge(0, 1, RelationType.ASSOCIATIVE, weight=0.5, confidence=0.5)
        transitions.add_edge(1, 2, RelationType.ASSOCIATIVE, weight=0.5, confidence=0.5)
        transitions.add_edge(2, 0, RelationType.ASSOCIATIVE, weight=0.5, confidence=0.5)

        monitor = ManifoldIntegrityMonitor()
        cycles = monitor.detect_reasoning_loops(transitions)
        assert len(cycles) >= 1

    def test_detect_reasoning_loops_no_cycle(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        # Acyclic: 0 -> 1 -> 2
        transitions.add_edge(0, 1, RelationType.CAUSAL)
        transitions.add_edge(1, 2, RelationType.CAUSAL)

        monitor = ManifoldIntegrityMonitor()
        cycles = monitor.detect_reasoning_loops(transitions)
        assert len(cycles) == 0
