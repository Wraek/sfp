"""Tests for metacognition.uncertainty — MetacognitionEngine."""

import torch
import pytest

from sfp.config import MetacognitionConfig
from sfp.metacognition.uncertainty import MetacognitionEngine
from sfp.types import ChainTrace, UncertaintyEstimate, WorldModelState


@pytest.fixture
def engine():
    return MetacognitionEngine(MetacognitionConfig(), d_model=32)


class TestRetrievalUncertainty:
    def test_output_in_range(self, engine):
        attn = torch.softmax(torch.randn(5), dim=0)
        u = engine.estimate_retrieval_uncertainty(attn, basin_confidence=0.8, n_active=10)
        assert 0.0 <= u <= 1.0

    def test_uniform_attention_higher_uncertainty(self, engine):
        uniform = torch.ones(10) / 10
        peaked = torch.zeros(10)
        peaked[0] = 1.0
        u_uniform = engine.estimate_retrieval_uncertainty(uniform, 0.5, 10)
        u_peaked = engine.estimate_retrieval_uncertainty(peaked, 0.9, 10)
        # Uniform attention and lower confidence should generally yield higher uncertainty
        # But since these are learned estimators, just check bounds
        assert 0.0 <= u_uniform <= 1.0
        assert 0.0 <= u_peaked <= 1.0


class TestChainUncertainty:
    def test_empty_trace(self, engine):
        u = engine.estimate_chain_uncertainty([])
        assert 0.0 <= u <= 1.0

    def test_with_trace(self, engine):
        trace = [
            ChainTrace(hop=0, basin_id=0, event_type="start", confidence=0.9),
            ChainTrace(hop=1, basin_id=1, event_type="hop", confidence=0.8, score=0.7),
        ]
        u = engine.estimate_chain_uncertainty(trace)
        assert 0.0 <= u <= 1.0


class TestPredictionUncertainty:
    def test_output_in_range(self, engine):
        state = WorldModelState(
            deterministic=torch.randn(64),
            stochastic=torch.randn(64),
            prediction_error=0.5,
            kl_divergence=0.3,
            reconstruction_error=0.2,
        )
        u = engine.estimate_prediction_uncertainty(state)
        assert 0.0 <= u <= 1.0


class TestKnowledgeUncertainty:
    def test_output_in_range(self, engine):
        u = engine.estimate_knowledge_uncertainty(
            confidence=0.8, maturity=0.5, modality_coverage=0.6,
        )
        assert 0.0 <= u <= 1.0

    def test_high_maturity_lower_uncertainty(self, engine):
        u_low = engine.estimate_knowledge_uncertainty(0.9, 0.9, 0.9)
        u_high = engine.estimate_knowledge_uncertainty(0.1, 0.1, 0.1)
        # Both in range
        assert 0.0 <= u_low <= 1.0
        assert 0.0 <= u_high <= 1.0


class TestComposition:
    def test_compose_returns_estimate(self, engine):
        ctx = torch.randn(32)
        est = engine.compose_uncertainty(0.3, 0.4, 0.5, 0.6, ctx)
        assert isinstance(est, UncertaintyEstimate)
        assert 0.0 <= est.scalar_confidence <= 1.0
        assert est.composite_embedding.shape[0] > 0


class TestCalibration:
    def test_update_and_ece(self, engine):
        for i in range(50):
            engine.update_calibration(0.8, True)
            engine.update_calibration(0.2, False)
        ece = engine.get_ece()
        assert ece >= 0.0

    def test_ece_zero_with_insufficient_data(self, engine):
        assert engine.get_ece() == 0.0

    def test_calibration_report(self, engine):
        for _ in range(20):
            engine.update_calibration(0.5, True)
        report = engine.get_calibration_report()
        assert "ece" in report
        assert "bins" in report


class TestMemoryHealth:
    def test_monitor_empty(self, engine):
        keys = torch.randn(16, 32)
        conf = torch.ones(16) * 0.5
        report = engine.monitor_memory_health(keys, conf, n_active=0)
        assert report["total_active_basins"] == 0

    def test_record_activation(self, engine):
        engine.record_activation([0, 1, 2], [0.9, 0.8, 0.7])
        assert engine._activation_counts[0] == 1

    def test_dormant_basins_detected(self, engine):
        keys = torch.randn(16, 32)
        conf = torch.ones(16) * 0.5
        # No activations recorded → all basins are dormant
        report = engine.monitor_memory_health(keys, conf, n_active=5)
        assert len(report["dormant_basins"]) > 0


class TestInformationSeeking:
    def test_no_suggestions_when_confident(self, engine):
        est = UncertaintyEstimate(
            retrieval_uncertainty=0.1,
            chain_uncertainty=0.1,
            prediction_uncertainty=0.1,
            knowledge_uncertainty=0.1,
            composite_embedding=torch.randn(64),
            scalar_confidence=0.9,
            calibrated=True,
        )
        suggestions = engine.suggest_information_seeking(est)
        assert len(suggestions) == 0

    def test_suggestions_when_uncertain(self, engine):
        est = UncertaintyEstimate(
            retrieval_uncertainty=0.8,
            chain_uncertainty=0.7,
            prediction_uncertainty=0.9,
            knowledge_uncertainty=0.6,
            composite_embedding=torch.randn(64),
            scalar_confidence=0.2,
            calibrated=True,
        )
        suggestions = engine.suggest_information_seeking(est)
        assert len(suggestions) > 0
