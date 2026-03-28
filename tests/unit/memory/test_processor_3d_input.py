"""Tests for HierarchicalMemoryProcessor 3D (multi-token) input path.

Verifies that (1, N, d_model) tensors flow correctly through Perceiver IO →
Backbone → downstream pipeline, as required for multi-token observations
from bridges.
"""

import torch
import pytest

from sfp.config import (
    FieldConfig,
    SelectiveAttentionConfig,
    StreamingConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
)
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import SurpriseMetric

D = 64  # tiny dimension for fast tests


@pytest.fixture
def processor():
    """Minimal processor with salience gate enabled."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=Tier1Config(
            hot_capacity=20, cold_capacity=40, surprise_threshold=0.0,
        ),
        tier2_config=Tier2Config(n_slots=16, d_value=D),
        tier3_config=Tier3Config(n_slots=8, d_value=D),
        streaming_config=StreamingConfig(surprise_threshold=0.0),
        attention_config=SelectiveAttentionConfig(
            n_modalities=2,
            modality_names=("text", "vision"),
            d_salience=16,
            d_context=16,
        ),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_no_salience():
    """Processor without salience gate (simpler path)."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=Tier1Config(
            hot_capacity=20, cold_capacity=40, surprise_threshold=0.0,
        ),
        tier2_config=Tier2Config(n_slots=16, d_value=D),
        tier3_config=Tier3Config(n_slots=8, d_value=D),
        streaming_config=StreamingConfig(surprise_threshold=0.0),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


class TestMultiTokenInput:
    """Test that (1, N, d_model) inputs flow through Perceiver correctly."""

    def test_3d_input_returns_surprise_metric(self, processor_no_salience):
        """3D tensor should be processed without error and return SurpriseMetric."""
        x = torch.randn(1, 96, D)
        result = processor_no_salience.process(x, modality="environment")
        assert isinstance(result, SurpriseMetric)

    def test_variable_token_counts(self, processor_no_salience):
        """Different N values should all work (Perceiver handles variable length)."""
        for n_tokens in [50, 96, 112, 200]:
            x = torch.randn(1, n_tokens, D)
            result = processor_no_salience.process(x, modality="environment")
            assert isinstance(result, SurpriseMetric)

    def test_3d_input_with_salience_gate(self, processor):
        """3D input should work when salience gate is enabled.

        The gate pools to a vector internally (_pool_to_vector) for scoring,
        then the full 3D tensor flows through Perceiver if processing level is FULL.
        """
        x = torch.randn(1, 96, D)
        result = processor.process(x, modality="environment")
        assert isinstance(result, SurpriseMetric)

    def test_3d_vs_1d_both_work(self, processor_no_salience):
        """Both 1D and 3D inputs should produce valid results."""
        x_1d = torch.randn(D)
        x_3d = torch.randn(1, 96, D)

        result_1d = processor_no_salience.process(x_1d, modality="text")
        result_3d = processor_no_salience.process(x_3d, modality="environment")

        assert isinstance(result_1d, SurpriseMetric)
        assert isinstance(result_3d, SurpriseMetric)

    def test_query_with_3d_input(self, processor_no_salience):
        """query() should also accept 3D inputs (read-only path)."""
        x = torch.randn(1, 96, D)
        result = processor_no_salience.query(x)
        assert hasattr(result, "knowledge")
        assert result.knowledge.shape == (D,)


class TestSalienceGateWith3DInput:
    """Test that the salience gate correctly handles multi-token observations."""

    def test_unknown_modality_not_dropped(self, processor):
        """Salience gate should not drop unknown modalities for 3D input."""
        x = torch.randn(1, 96, D)
        # "environment" is not in the gate's initial modality list
        result = processor.process(x, modality="environment")
        assert isinstance(result, SurpriseMetric)

    def test_repeated_3d_processing(self, processor_no_salience):
        """Multiple 3D inputs in sequence should work (no state corruption)."""
        for i in range(10):
            x = torch.randn(1, 96, D)
            result = processor_no_salience.process(x, modality="environment")
            assert isinstance(result, SurpriseMetric)
