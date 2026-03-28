"""Tests for quantization module."""

import torch
import pytest

from sfp.storage.quantization import (
    ManifoldQuantizer,
    dequantize_tensor_int8,
    quantize_tensor_int8,
)


class TestQuantizeTensor:
    def test_round_trip(self):
        t = torch.randn(64, 64)
        q, scale, zp = quantize_tensor_int8(t, per_channel=True)
        restored = dequantize_tensor_int8(q, scale, zp)
        assert restored.shape == t.shape
        mse = (t - restored).pow(2).mean().item()
        assert mse < 0.01

    def test_int8_dtype(self):
        t = torch.randn(32, 32)
        q, _, _ = quantize_tensor_int8(t)
        assert q.dtype == torch.int8

    def test_per_tensor(self):
        t = torch.randn(32, 32)
        q, scale, zp = quantize_tensor_int8(t, per_channel=False)
        restored = dequantize_tensor_int8(q, scale, zp)
        assert restored.shape == t.shape


class TestManifoldQuantizer:
    def test_quantize_dequantize(self, tiny_field):
        x = torch.randn(256)
        orig = tiny_field(x)

        qstate = ManifoldQuantizer.quantize(tiny_field)
        restored = ManifoldQuantizer.dequantize(qstate)
        rest_out = restored(x)

        mse = (orig - rest_out).pow(2).mean().item()
        assert mse < 0.01

    def test_estimate_information_content(self, tiny_field):
        info = ManifoldQuantizer.estimate_information_content(tiny_field)
        assert 0 < info <= 8.0  # Max 8 bits for float

    def test_quantize_stores_config(self, tiny_field):
        qstate = ManifoldQuantizer.quantize(tiny_field)
        assert "field_config" in qstate
        assert qstate["field_config"]["dim"] == tiny_field.config.dim
