"""Tests for reasoning.transitions — TransitionStructure."""

import torch
import pytest

from sfp.config import TransitionConfig
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import RelationType


class TestTransitionStructure:
    def test_add_edge(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        idx = ts.add_edge(0, 1, RelationType.CAUSAL, weight=0.5, confidence=0.3)
        assert ts.n_active_edges == 1
        assert ts.source[idx].item() == 0
        assert ts.target[idx].item() == 1

    def test_duplicate_edge_updates(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        idx1 = ts.add_edge(0, 1, RelationType.CAUSAL, weight=0.5, confidence=0.3)
        idx2 = ts.add_edge(0, 1, RelationType.CAUSAL, weight=0.8, confidence=0.1)
        assert idx1 == idx2  # Same edge updated
        assert ts.n_active_edges == 1
        # Weight should be max of old and new
        assert ts.weight[idx1].item() == pytest.approx(0.8)

    def test_get_outgoing(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        ts.add_edge(0, 1)
        ts.add_edge(0, 2)
        ts.add_edge(1, 2)
        targets, weights, indices = ts.get_outgoing(0)
        assert targets.shape[0] == 2
        assert set(targets.tolist()) == {1, 2}

    def test_get_outgoing_empty(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        targets, weights, indices = ts.get_outgoing(5)
        assert targets.shape[0] == 0

    def test_get_incoming(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        ts.add_edge(0, 2)
        ts.add_edge(1, 2)
        sources, weights, indices = ts.get_incoming(2)
        assert sources.shape[0] == 2
        assert set(sources.tolist()) == {0, 1}

    def test_compute_transition_scores(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16, d_relation=16), d_model=32)
        ts.add_edge(0, 1, RelationType.ASSOCIATIVE, weight=0.5, confidence=0.5)
        ts.add_edge(0, 2, RelationType.CAUSAL, weight=0.8, confidence=0.8)
        query = torch.randn(32)
        scores, targets = ts.compute_transition_scores(0, query)
        assert scores.shape[0] == 2
        assert targets.shape[0] == 2

    def test_compute_transition_scores_empty(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        scores, targets = ts.compute_transition_scores(0, torch.randn(32))
        assert scores.shape[0] == 0

    def test_get_edge_info(self):
        ts = TransitionStructure(TransitionConfig(max_edges=16), d_model=32)
        idx = ts.add_edge(3, 5, RelationType.TEMPORAL, weight=0.7, confidence=0.4)
        info = ts.get_edge_info(idx)
        assert info["source"] == 3
        assert info["target"] == 5
        assert info["relation_type"] == "TEMPORAL"
        assert info["active"] is True

    def test_eviction_at_capacity(self):
        ts = TransitionStructure(TransitionConfig(max_edges=3), d_model=32)
        ts.add_edge(0, 1, confidence=0.1)
        ts.add_edge(1, 2, confidence=0.9)
        ts.add_edge(2, 3, confidence=0.5)
        assert ts.n_active_edges == 3
        # Adding a 4th should evict lowest confidence (0.1)
        ts.add_edge(3, 4, confidence=0.8)
        assert ts.n_active_edges == 3
        # Basin 0->1 (conf=0.1) should be gone
        targets, _, _ = ts.get_outgoing(0)
        assert 1 not in targets.tolist()
