"""Tests for core.backbone — BackboneTransformer."""

import torch
import pytest

from sfp.config import BackboneConfig
from sfp.core.backbone import BackboneTransformer


@pytest.fixture
def backbone():
    return BackboneTransformer(BackboneConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))


class TestBackboneTransformer:
    def test_forward_shape(self, backbone):
        x = torch.randn(2, 8, 64)
        out = backbone(x)
        assert out.shape == (2, 8, 64)

    def test_single_batch(self, backbone):
        x = torch.randn(1, 4, 64)
        out = backbone(x)
        assert out.shape == (1, 4, 64)

    def test_single_token(self, backbone):
        x = torch.randn(1, 1, 64)
        out = backbone(x)
        assert out.shape == (1, 1, 64)

    def test_output_differs_from_input(self, backbone):
        x = torch.randn(1, 4, 64)
        out = backbone(x)
        assert not torch.allclose(x, out, atol=1e-3)

    def test_param_count(self, backbone):
        assert backbone.param_count > 0

    def test_memory_bytes(self, backbone):
        mb = backbone.memory_bytes()
        assert mb == backbone.param_count * 4  # float32

    def test_gradient_flow(self, backbone):
        x = torch.randn(1, 4, 64, requires_grad=True)
        out = backbone(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.norm().item() > 0

    def test_different_inputs_different_outputs(self, backbone):
        x1 = torch.randn(1, 4, 64)
        x2 = torch.randn(1, 4, 64)
        out1 = backbone(x1)
        out2 = backbone(x2)
        assert not torch.allclose(out1, out2)
