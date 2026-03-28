"""Tests for gradient compression."""

import torch
import pytest

from sfp.comms.compression import GradientCompressor


class TestGradientCompressor:
    def test_topk_compress_decompress(self):
        comp = GradientCompressor(method="topk", density=0.1)
        deltas = {"w": torch.randn(50, 50)}
        compressed = comp.compress(deltas)
        assert compressed.method == "topk"
        restored = comp.decompress(compressed)
        assert "w" in restored
        assert restored["w"].shape == (50, 50)

    def test_topk_sparsity(self):
        comp = GradientCompressor(method="topk", density=0.01)
        deltas = {"w": torch.randn(100, 100)}
        compressed = comp.compress(deltas)
        n_nonzero = compressed.indices["w"].shape[0]
        assert n_nonzero == 100  # 0.01 * 10000

    def test_signsgd_compress_decompress(self):
        comp = GradientCompressor(method="signsgd")
        deltas = {"w": torch.randn(50, 50)}
        compressed = comp.compress(deltas)
        assert compressed.method == "signsgd"
        restored = comp.decompress(compressed)
        assert restored["w"].shape == (50, 50)

    def test_error_accumulation(self):
        comp = GradientCompressor(method="topk", density=0.01)
        deltas = {"w": torch.randn(100, 100)}

        # First compression
        comp.compress(deltas)
        assert "w" in comp._error_buffer

        # Error buffer should be non-zero
        assert comp._error_buffer["w"].abs().sum().item() > 0

    def test_reset_error_buffers(self):
        comp = GradientCompressor(method="topk", density=0.01)
        comp.compress({"w": torch.randn(50, 50)})
        assert len(comp._error_buffer) > 0
        comp.reset_error_buffers()
        assert len(comp._error_buffer) == 0

    def test_multiple_params(self):
        comp = GradientCompressor(method="topk", density=0.1)
        deltas = {
            "layer1.weight": torch.randn(32, 32),
            "layer2.weight": torch.randn(64, 64),
        }
        compressed = comp.compress(deltas)
        restored = comp.decompress(compressed)
        assert restored["layer1.weight"].shape == (32, 32)
        assert restored["layer2.weight"].shape == (64, 64)
