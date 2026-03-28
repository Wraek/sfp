"""Tests for affect.valence — ValenceSystem."""

import torch
import pytest

from sfp.affect.valence import ValenceSystem
from sfp.config import ValenceConfig
from sfp.types import ValenceSignal


@pytest.fixture
def vs():
    return ValenceSystem(ValenceConfig(), d_model=32)


class TestComputeValence:
    def test_returns_valence_signal(self, vs):
        emb = torch.randn(32)
        signal = vs.compute_valence(emb)
        assert isinstance(signal, ValenceSignal)
        assert -1.0 <= signal.scalar_valence <= 1.0

    def test_positive_reward_increases_mood(self, vs):
        emb = torch.randn(32)
        for _ in range(20):
            signal = vs.compute_valence(emb, reward=1.0)
        assert signal.mood_immediate > -1.0  # Moved from 0

    def test_negative_feedback_decreases_mood(self, vs):
        emb = torch.randn(32)
        for _ in range(20):
            signal = vs.compute_valence(emb, user_feedback=-1.0)
        assert signal.mood_immediate < 0.5

    def test_valence_embedding_shape(self, vs):
        signal = vs.compute_valence(torch.randn(32))
        d_val = vs._config.d_valence_embedding
        assert signal.valence_embedding.shape == (d_val,)
        assert signal.projected_context.shape == (32,)

    def test_risk_tolerance_range(self, vs):
        signal = vs.compute_valence(torch.randn(32))
        assert 0.2 <= signal.risk_tolerance <= 0.8

    def test_vigilance_range(self, vs):
        signal = vs.compute_valence(torch.randn(32))
        assert 0.2 <= signal.vigilance <= 0.8


class TestBasinAnnotation:
    def test_annotate_basin(self, vs):
        vs.annotate_basin(0, 0.5, torch.randn(vs._config.d_valence_embedding))
        assert vs.basin_valence_count[0].item() == 1
        assert vs.basin_valence_scalar[0].item() != 0

    def test_annotate_edge(self, vs):
        vs.annotate_edge(0, -0.3)
        assert vs.edge_valence_count[0].item() == 1
        assert abs(vs.edge_valence[0].item() + 0.3) < 0.5  # EMA starts at 0


class TestChainValence:
    def test_empty_chain(self, vs):
        assert vs.compute_chain_valence([]) == 0.0

    def test_chain_valence_reflects_basin_valence(self, vs):
        vs.basin_valence_scalar[0] = 0.8
        vs.basin_valence_scalar[1] = -0.5
        val = vs.compute_chain_valence([0, 1])
        # Weighted average — closer to basin 1 (more recent)
        assert -1.0 <= val <= 1.0


class TestModulators:
    def test_surprise_threshold_modifier(self, vs):
        signal = vs.compute_valence(torch.randn(32))
        mod = vs.get_surprise_threshold_modifier(signal)
        assert 0.7 <= mod <= 1.0

    def test_retention_priority_modifier(self, vs):
        signal = vs.compute_valence(torch.randn(32))
        mod = vs.get_retention_priority_modifier(signal)
        assert 1.0 <= mod <= 2.0

    def test_reasoning_mode(self, vs):
        emb = torch.randn(32)
        # After many positive inputs
        for _ in range(50):
            signal = vs.compute_valence(emb, reward=1.0, user_feedback=1.0)
        mode = vs.get_reasoning_mode(signal)
        assert mode in ("approach", "avoidance", "neutral")

    def test_reasoning_valence_bias_neutral(self, vs):
        targets = torch.tensor([0, 1, 2])
        bias = vs.get_reasoning_valence_bias("neutral", targets)
        assert torch.allclose(bias, torch.zeros(3))

    def test_consolidation_sampling_weights(self, vs):
        vs.basin_valence_scalar[0] = 0.9
        vs.basin_valence_scalar[1] = 0.0
        vs.basin_valence_scalar[2] = -0.5
        weights = vs.get_consolidation_sampling_weights([0, 1, 2])
        assert weights.shape == (3,)
        assert abs(weights.sum().item() - 1.0) < 1e-5
        # Basin 0 (high |valence|) should have highest weight
        assert weights[0].item() > weights[1].item()


class TestMoodProperty:
    def test_mood_dict(self, vs):
        vs.compute_valence(torch.randn(32))
        mood = vs.mood
        assert "immediate" in mood
        assert "short_term" in mood
        assert "baseline" in mood
        assert "composite" in mood
        assert -1.0 <= mood["composite"] <= 1.0
