"""Shared fixtures for all SFP tests."""

import pytest
import torch

import sfp
from sfp.config import FieldConfig


@pytest.fixture
def tiny_config():
    """Tiny field config for fast tests."""
    return FieldConfig.from_preset(sfp.FieldSize.TINY)


@pytest.fixture
def tiny_field(tiny_config):
    """Tiny SemanticFieldProcessor on CPU."""
    return sfp.SemanticFieldProcessor(tiny_config)


@pytest.fixture
def tiny_processor(tiny_field):
    """Tiny StreamingProcessor with LoRA and EWC."""
    return sfp.StreamingProcessor(
        field=tiny_field,
        streaming_config=sfp.StreamingConfig(),
        lora_config=sfp.LoRAConfig(enabled=True),
        ewc_config=sfp.EWCConfig(enabled=True),
    )


@pytest.fixture
def random_input(tiny_config):
    """Random input tensor matching tiny field dim."""
    return torch.randn(tiny_config.dim)


@pytest.fixture
def random_batch(tiny_config):
    """Random batch of 8 inputs."""
    return torch.randn(8, tiny_config.dim)
