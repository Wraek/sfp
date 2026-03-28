"""Tests for dynamic modality registration in SalienceGate.

Verifies that unknown modalities are lazily registered rather than silently
dropped, making the SFP open to any bridge-provided modality at runtime.
"""

import torch
import pytest

from sfp.attention.salience import SalienceGate
from sfp.config import SelectiveAttentionConfig
from sfp.types import ProcessingLevel


@pytest.fixture
def gate():
    """Gate with only 2 known modalities."""
    cfg = SelectiveAttentionConfig(
        n_modalities=2,
        modality_names=("text", "vision"),
        skip_threshold=0.1,
        skim_threshold=0.5,
        d_salience=16,
        d_context=16,
    )
    return SalienceGate(cfg, d_model=32)


class TestLazyModalityRegistration:
    def test_known_modality_works(self, gate):
        """Baseline: known modalities produce scores as before."""
        inputs = {"text": torch.randn(32)}
        result = gate.evaluate(inputs)
        assert "text" in result.salience_scores

    def test_unknown_modality_not_dropped(self, gate):
        """Unknown modality should produce a score, not be silently skipped."""
        inputs = {"environment": torch.randn(32)}
        result = gate.evaluate(inputs)
        assert "environment" in result.salience_scores
        assert result.level in (
            ProcessingLevel.SKIP,
            ProcessingLevel.SKIM,
            ProcessingLevel.FULL,
        )

    def test_unknown_modality_creates_estimator(self, gate):
        """Lazy registration should add to ModuleDict."""
        assert "environment" not in gate.modality_estimators
        assert "environment" not in gate.change_detectors

        inputs = {"environment": torch.randn(32)}
        gate.evaluate(inputs)

        assert "environment" in gate.modality_estimators
        assert "environment" in gate.change_detectors

    def test_unknown_modality_gets_adaptive_threshold(self, gate):
        """New modality should get an adaptive threshold initialized."""
        assert "environment" not in gate._adaptive_thresholds

        inputs = {"environment": torch.randn(32)}
        gate.evaluate(inputs)

        assert "environment" in gate._adaptive_thresholds

    def test_multiple_unknown_modalities(self, gate):
        """Multiple unknown modalities in a single call should all register."""
        inputs = {
            "depth": torch.randn(32),
            "entity": torch.randn(32),
            "temporal": torch.randn(32),
        }
        result = gate.evaluate(inputs)
        assert "depth" in result.salience_scores
        assert "entity" in result.salience_scores
        assert "temporal" in result.salience_scores
        assert len(gate.modality_estimators) == 5  # 2 known + 3 new

    def test_lazy_estimator_persists(self, gate):
        """Once created, the same estimator should be reused on next call."""
        inputs = {"environment": torch.randn(32)}
        gate.evaluate(inputs)
        estimator_id = id(gate.modality_estimators["environment"])

        gate.evaluate(inputs)
        assert id(gate.modality_estimators["environment"]) == estimator_id

    def test_mixed_known_and_unknown(self, gate):
        """Known + unknown modalities in the same call both get scored."""
        inputs = {
            "text": torch.randn(32),
            "environment": torch.randn(32),
        }
        result = gate.evaluate(inputs)
        assert "text" in result.salience_scores
        assert "environment" in result.salience_scores

    def test_change_detection_works_for_new_modality(self, gate):
        """Second evaluation of a new modality should use change detection."""
        emb1 = torch.randn(32)
        emb2 = torch.randn(32)

        # First call: change_score defaults to 1.0 (first input)
        gate.evaluate({"environment": emb1})
        assert "environment" in gate._prev_inputs

        # Second call: should use change detector
        result = gate.evaluate({"environment": emb2})
        assert "environment" in result.salience_scores

    def test_skim_process_with_unknown_modality(self, gate):
        """skim_process should accept any modality string."""
        emb = torch.randn(32)
        result = gate.skim_process(emb, "new_game_modality")
        assert result["buffered"]
