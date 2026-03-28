"""Tests for SemanticFieldProcessor."""

import torch
import pytest

import sfp
from sfp.config import FieldConfig
from sfp.core.field import SemanticFieldProcessor


class TestFieldConfig:
    def test_from_preset_tiny(self):
        config = FieldConfig.from_preset(sfp.FieldSize.TINY)
        assert config.dim == 256
        assert config.n_layers == 4

    def test_from_preset_small(self):
        config = FieldConfig.from_preset(sfp.FieldSize.SMALL)
        assert config.dim == 512
        assert config.n_layers == 6

    def test_from_preset_medium(self):
        config = FieldConfig.from_preset(sfp.FieldSize.MEDIUM)
        assert config.dim == 1024
        assert config.n_layers == 8

    def test_from_preset_large(self):
        config = FieldConfig.from_preset(sfp.FieldSize.LARGE)
        assert config.dim == 2048
        assert config.n_layers == 8


class TestSemanticFieldProcessor:
    def test_forward_single(self, tiny_field, random_input):
        out = tiny_field(random_input)
        assert out.shape == random_input.shape

    def test_forward_batch(self, tiny_field, random_batch):
        out = tiny_field(random_batch)
        assert out.shape == random_batch.shape

    def test_param_count(self, tiny_field):
        assert tiny_field.param_count > 0
        manual = sum(p.numel() for p in tiny_field.parameters())
        assert tiny_field.param_count == manual

    def test_memory_bytes(self, tiny_field):
        mem = tiny_field.memory_bytes()
        assert mem == tiny_field.param_count * 4  # FP32

    def test_memory_bytes_fp16(self, tiny_field):
        mem = tiny_field.memory_bytes(torch.float16)
        assert mem == tiny_field.param_count * 2

    def test_linear_layers(self, tiny_field):
        linears = tiny_field.linear_layers()
        assert len(linears) == tiny_field.config.n_layers
        for layer in linears:
            assert isinstance(layer, torch.nn.Linear)

    def test_jacobian(self, tiny_field):
        x = torch.randn(tiny_field.config.dim)
        jac = tiny_field.jacobian(x)
        assert jac.shape == (tiny_field.config.dim, tiny_field.config.dim)

    def test_residual_field(self):
        config = FieldConfig(dim=64, n_layers=4, residual=True)
        field = SemanticFieldProcessor(config)
        x = torch.randn(64)
        out = field(x)
        assert out.shape == x.shape

    def test_relu_activation(self):
        config = FieldConfig(dim=64, n_layers=2, activation="relu")
        field = SemanticFieldProcessor(config)
        x = torch.randn(64)
        out = field(x)
        assert out.shape == x.shape

    def test_no_layernorm(self):
        config = FieldConfig(dim=64, n_layers=2, use_layernorm=False)
        field = SemanticFieldProcessor(config)
        x = torch.randn(64)
        out = field(x)
        assert out.shape == x.shape
