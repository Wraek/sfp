"""Tests for attention.salience — SalienceGate."""

import torch
import pytest

from sfp.attention.salience import SalienceGate
from sfp.config import SelectiveAttentionConfig
from sfp.types import ProcessingLevel


@pytest.fixture
def gate():
    cfg = SelectiveAttentionConfig(
        n_modalities=3,
        modality_names=["text", "vision", "audio"],
        skip_threshold=0.1,
        skim_threshold=0.5,
        d_salience=16,
        d_context=16,
    )
    return SalienceGate(cfg, d_model=32)


class TestSalienceEvaluation:
    def test_evaluate_returns_result(self, gate):
        inputs = {"text": torch.randn(32), "vision": torch.randn(32)}
        result = gate.evaluate(inputs)
        assert result.level in (ProcessingLevel.SKIP, ProcessingLevel.SKIM, ProcessingLevel.FULL)
        assert isinstance(result.combined_salience, float)

    def test_empty_inputs_skip(self, gate):
        result = gate.evaluate({})
        assert result.level == ProcessingLevel.SKIP

    def test_with_context_vectors(self, gate):
        inputs = {"text": torch.randn(32)}
        result = gate.evaluate(
            inputs,
            goal_context=torch.randn(32),
            world_model_prediction=torch.randn(32),
            recent_activation=torch.randn(32),
        )
        assert result.level in (ProcessingLevel.SKIP, ProcessingLevel.SKIM, ProcessingLevel.FULL)

    def test_per_modality_scores(self, gate):
        inputs = {"text": torch.randn(32), "vision": torch.randn(32)}
        result = gate.evaluate(inputs)
        assert "text" in result.salience_scores
        assert "vision" in result.salience_scores


class TestInterrupts:
    def test_no_interrupt_at_full(self, gate):
        scores = {"text": 0.9}
        interrupt, reason = gate.check_interrupts(scores, ProcessingLevel.FULL)
        assert not interrupt

    def test_cross_modal_convergence(self, gate):
        cfg = SelectiveAttentionConfig(
            n_modalities=3,
            modality_names=["text", "vision", "audio"],
            cross_modal_threshold=0.3,
        )
        g = SalienceGate(cfg, d_model=32)
        scores = {"text": 0.5, "vision": 0.5, "audio": 0.1}
        interrupt, reason = g.check_interrupts(scores, ProcessingLevel.SKIM)
        assert interrupt
        assert "cross_modal" in reason


class TestSkimProcessing:
    def test_skim_buffers_input(self, gate):
        emb = torch.randn(32)
        result = gate.skim_process(emb, "text")
        assert result["buffered"]
        assert result["buffer_size"] == 1

    def test_skim_summary_empty(self, gate):
        summary = gate.get_skim_summary()
        assert summary.shape == (32,)
        assert summary.norm().item() == 0.0

    def test_skim_summary_after_inputs(self, gate):
        for _ in range(5):
            gate.skim_process(torch.randn(32), "text")
        summary = gate.get_skim_summary()
        assert summary.norm().item() > 0


class TestHindsightTraining:
    def test_train_hindsight_insufficient_data(self, gate):
        loss = gate.run_hindsight_training()
        assert loss == 0.0

    def test_record_hindsight(self, gate):
        gate.train_hindsight(0.5, True, torch.randn(32))
        assert len(gate._hindsight_buffer) == 1


class TestGoalModulation:
    def test_apply_goal_modulation(self, gate):
        base = {"text": 0.5, "vision": 0.5}
        mods = {"text": -0.1, "vision": -0.2}
        adjusted = gate.apply_goal_modulation(base, mods)
        assert adjusted["text"] < 0.5
        assert adjusted["vision"] < 0.5
        assert adjusted["text"] >= 0.05  # floor

    def test_apply_expectation_modulation(self, gate):
        inputs = {"text": torch.randn(32), "vision": torch.randn(32)}
        pred = torch.randn(32)
        errors = gate.apply_expectation_modulation(inputs, pred)
        for v in errors.values():
            assert 0.0 <= v <= 1.0
