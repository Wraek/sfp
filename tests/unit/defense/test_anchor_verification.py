"""Tests for defense.anchor_verification — AnchorVerifier."""

import torch
import pytest

from sfp.config import Tier2Config
from sfp.defense.anchor_verification import AnchorVerifier
from sfp.memory.essential import EssentialMemory


def _make_tier2(d: int = 32, n_slots: int = 8) -> EssentialMemory:
    """Create a small EssentialMemory with identity projections."""
    tier2 = EssentialMemory(Tier2Config(n_slots=n_slots, d_value=d), d_model=d)
    with torch.no_grad():
        tier2.query_proj.weight.copy_(torch.eye(d))
        tier2.key_proj.weight.copy_(torch.eye(d))
        tier2.value_proj.weight.copy_(torch.eye(d))
        tier2.output_proj.weight.copy_(torch.eye(d))
    return tier2


class TestAnchorVerifier:
    def test_no_violations_with_correct_basins(self):
        d = 32
        tier2 = _make_tier2(d)
        # Allocate 2 well-separated orthogonal basins
        k0 = torch.zeros(d)
        k0[0] = 5.0
        k1 = torch.zeros(d)
        k1[1] = 5.0
        tier2.allocate_slot(k0, value=k0)
        tier2.allocate_slot(k1, value=k1)
        # Set confidence so attention scores are meaningful (allocate_slot sets 0.0)
        with torch.no_grad():
            tier2.confidence[0] = 0.9
            tier2.confidence[1] = 0.9

        anchors = torch.stack([k0, k1])
        expected = torch.tensor([0, 1])
        verifier = AnchorVerifier(anchors, expected, drift_threshold=0.5)
        violations = verifier.verify(tier2)
        assert len(violations) == 0

    def test_basin_shift_detected(self):
        d = 32
        tier2 = _make_tier2(d)
        k0 = torch.zeros(d)
        k0[0] = 5.0
        k1 = torch.zeros(d)
        k1[1] = 5.0
        tier2.allocate_slot(k0, value=k0)
        tier2.allocate_slot(k1, value=k1)
        with torch.no_grad():
            tier2.confidence[0] = 0.9
            tier2.confidence[1] = 0.9

        # Anchor expects basin 1, but it's actually basin 0
        anchors = torch.stack([k0])
        expected = torch.tensor([1])  # wrong expected basin
        verifier = AnchorVerifier(anchors, expected, drift_threshold=0.5)
        violations = verifier.verify(tier2)
        assert any("basin shifted" in v for v in violations)

    def test_no_active_basins_violation(self):
        d = 32
        tier2 = _make_tier2(d)
        anchors = torch.randn(2, d)
        expected = torch.tensor([0, 1])
        verifier = AnchorVerifier(anchors, expected)
        violations = verifier.verify(tier2)
        assert any("No active basins" in v for v in violations)

    def test_n_anchors_property(self):
        anchors = torch.randn(5, 16)
        expected = torch.arange(5)
        verifier = AnchorVerifier(anchors, expected)
        assert verifier.n_anchors == 5

    def test_pairwise_distance_stability(self):
        d = 32
        tier2 = _make_tier2(d)
        # Two well-separated anchors
        k0 = torch.zeros(d)
        k0[0] = 5.0
        k1 = torch.zeros(d)
        k1[1] = 5.0
        tier2.allocate_slot(k0, value=k0)
        tier2.allocate_slot(k1, value=k1)
        with torch.no_grad():
            tier2.confidence[0] = 0.9
            tier2.confidence[1] = 0.9

        anchors = torch.stack([k0, k1])
        expected = torch.tensor([0, 1])
        # Pairwise distances are based on anchor embeddings alone,
        # so they shouldn't change (anchors are cloned at init)
        verifier = AnchorVerifier(anchors, expected, pairwise_tolerance=0.001)
        violations = verifier.verify(tier2)
        # No pairwise violations since anchors haven't changed
        pairwise_violations = [v for v in violations if "pairwise" in v]
        assert len(pairwise_violations) == 0
