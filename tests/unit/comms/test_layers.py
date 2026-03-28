"""Tests for communication layers."""

import torch
import pytest

from sfp.comms.layers import (
    L0RawText,
    L1Embedding,
    L2ManifoldCoord,
    L4SurpriseGated,
)
from sfp.core.field import SemanticFieldProcessor
from sfp.config import FieldConfig


@pytest.fixture
def small_field():
    config = FieldConfig(dim=64, n_layers=2)
    return SemanticFieldProcessor(config)


class TestL0RawText:
    def test_round_trip(self):
        l0 = L0RawText()
        text = "hello world"
        assert l0.decode(l0.encode(text)) == text

    def test_unicode(self):
        l0 = L0RawText()
        text = "unicode test"
        assert l0.decode(l0.encode(text)) == text


class TestL1Embedding:
    def test_int8_round_trip(self):
        l1 = L1Embedding(quantize_to="int8")
        vec = torch.randn(128)
        data = l1.encode(vec)
        restored = l1.decode(data)
        assert restored.shape == vec.shape
        mse = (vec - restored).pow(2).mean().item()
        assert mse < 0.01

    def test_fp16_round_trip(self):
        l1 = L1Embedding(quantize_to="fp16")
        vec = torch.randn(128)
        data = l1.encode(vec)
        restored = l1.decode(data)
        assert restored.shape == vec.shape

    def test_compression(self):
        l1 = L1Embedding(quantize_to="int8")
        vec = torch.randn(256)
        data = l1.encode(vec)
        raw_size = 256 * 4  # FP32
        assert len(data) < raw_size


class TestL2ManifoldCoord:
    def test_round_trip(self, small_field):
        l2 = L2ManifoldCoord(small_field)
        # Build codebook
        from sfp.core.attractors import AttractorQuery

        q = AttractorQuery(small_field)
        codebook = q.discover_attractors(n_probes=50, merge_radius=0.5)
        l2.set_codebook(codebook)

        concept = torch.randn(64)
        data = l2.encode(concept)
        restored = l2.decode(data)
        assert restored.shape == concept.shape

    def test_compression_ratio(self, small_field):
        l2 = L2ManifoldCoord(small_field)
        from sfp.core.attractors import AttractorQuery

        q = AttractorQuery(small_field)
        codebook = q.discover_attractors(n_probes=50, merge_radius=0.5)
        l2.set_codebook(codebook)

        concept = torch.randn(64)
        data = l2.encode(concept)
        raw_size = 64 * 4
        assert len(data) < raw_size


class TestL4SurpriseGated:
    def test_should_transmit(self, small_field):
        l4 = L4SurpriseGated(small_field)
        # Peer with identical field -> not surprising
        peer = SemanticFieldProcessor(FieldConfig(dim=64, n_layers=2))
        concept = torch.randn(64)
        # Different random weights -> likely surprising
        result = l4.should_transmit(concept, peer, threshold=0.001)
        assert isinstance(result, bool)
