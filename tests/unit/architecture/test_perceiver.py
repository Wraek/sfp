"""Tests for core.perceiver — PerceiverIO."""

import torch
import pytest

from sfp.config import PerceiverConfig
from sfp.core.perceiver import PerceiverIO


@pytest.fixture
def perceiver():
    return PerceiverIO(PerceiverConfig(
        d_input=32,
        d_latent=64,
        n_latents=8,
        n_heads=4,
        n_cross_attn_layers=1,
        n_self_attn_layers=1,
    ))


class TestPerceiverIO:
    def test_forward_shape(self, perceiver):
        x = torch.randn(2, 16, 32)  # batch=2, 16 tokens, d_input=32
        out = perceiver(x)
        assert out.shape == (2, 8, 64)  # batch=2, n_latents=8, d_latent=64

    def test_variable_input_length(self, perceiver):
        x_short = torch.randn(1, 4, 32)
        x_long = torch.randn(1, 32, 32)
        out_short = perceiver(x_short)
        out_long = perceiver(x_long)
        # Both should produce same latent shape
        assert out_short.shape == (1, 8, 64)
        assert out_long.shape == (1, 8, 64)

    def test_decode(self, perceiver):
        x = torch.randn(1, 8, 32)
        latents = perceiver(x)
        query = torch.randn(1, 4, 64)
        decoded = perceiver.decode(latents, query)
        assert decoded.shape == (1, 4, 64)

    def test_param_count(self, perceiver):
        assert perceiver.param_count > 0

    def test_memory_bytes(self, perceiver):
        assert perceiver.memory_bytes() == perceiver.param_count * 4

    def test_gradient_flow(self, perceiver):
        x = torch.randn(1, 8, 32, requires_grad=True)
        out = perceiver(x)
        out.sum().backward()
        assert x.grad is not None

    def test_different_inputs_different_latents(self, perceiver):
        x1 = torch.randn(1, 8, 32)
        x2 = torch.randn(1, 8, 32)
        out1 = perceiver(x1)
        out2 = perceiver(x2)
        assert not torch.allclose(out1, out2)
