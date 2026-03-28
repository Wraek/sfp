"""Tests for DimensionalityProjection."""

import torch
import pytest

from sfp.input.projection import DimensionalityProjection


class TestDimensionalityProjection:
    def test_forward(self):
        proj = DimensionalityProjection(128, 256)
        x = torch.randn(4, 128)
        out = proj(x)
        assert out.shape == (4, 256)

    def test_no_normalize(self):
        proj = DimensionalityProjection(128, 256, normalize=False)
        x = torch.randn(4, 128)
        out = proj(x)
        assert out.shape == (4, 256)

    def test_fit(self):
        proj = DimensionalityProjection(64, 128)
        src = torch.randn(50, 64)
        tgt = torch.randn(50, 128)
        loss = proj.fit(src, tgt, epochs=20)
        assert loss > 0
        # After training, should be better than random
        with torch.no_grad():
            pred = proj(src)
            final_mse = (pred - tgt).pow(2).mean().item()
        assert final_mse < 2.0  # Should be somewhat close
