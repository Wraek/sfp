"""Tests for manifold synchronization."""

import torch
import pytest

from sfp.comms.sync import ManifoldSynchronizer
from sfp.core.field import SemanticFieldProcessor
from sfp.config import FieldConfig


@pytest.fixture
def field_pair():
    config = FieldConfig(dim=64, n_layers=2)
    a = SemanticFieldProcessor(config)
    b = SemanticFieldProcessor(config)
    return a, b


class TestManifoldSynchronizer:
    def test_fingerprint_shape(self, tiny_field):
        sync = ManifoldSynchronizer(tiny_field)
        fp = sync.compute_fingerprint()
        assert fp.dim() == 1

    def test_identical_fields_no_drift(self):
        config = FieldConfig(dim=64, n_layers=2)
        field = SemanticFieldProcessor(config)
        sync = ManifoldSynchronizer(field)
        fp = sync.compute_fingerprint()
        drift = sync.detect_drift(fp, fp)
        assert drift < 1e-6

    def test_different_fields_have_drift(self, field_pair):
        a, b = field_pair
        sync_a = ManifoldSynchronizer(a)
        sync_b = ManifoldSynchronizer(b, anchor_points=sync_a._anchors)
        fp_a = sync_a.compute_fingerprint()
        fp_b = sync_b.compute_fingerprint()
        drift = sync_a.detect_drift(fp_a, fp_b)
        assert drift > 0

    def test_needs_sync(self, field_pair):
        a, b = field_pair
        sync_a = ManifoldSynchronizer(a, drift_threshold=0.001)
        sync_b = ManifoldSynchronizer(b, anchor_points=sync_a._anchors)
        fp_b = sync_b.compute_fingerprint()
        assert sync_a.needs_sync(fp_b)

    def test_create_sync_payload(self, field_pair):
        a, b = field_pair
        sync = ManifoldSynchronizer(a)
        deltas = sync.create_sync_payload(b)
        assert len(deltas) > 0
        for name, delta in deltas.items():
            assert isinstance(delta, torch.Tensor)
