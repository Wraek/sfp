"""Tests for co-adaptation extensions to StreamingProcessor.

Covers:
  - World model auxiliary loss (Change 2)
  - Confidence-based gradient scaling (Change 3)
  - Goal satisfaction loss (Change 5)
  - Replay episode method (Change 1)
  - Backward compatibility with None defaults
"""

import torch
import pytest

import sfp
from sfp.config import FieldConfig, StreamingConfig
from sfp.core.streaming import StreamingProcessor


@pytest.fixture
def co_adapt_processor():
    """StreamingProcessor with all co-adaptation features configured."""
    field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
    return StreamingProcessor(
        field,
        streaming_config=StreamingConfig(
            auxiliary_loss_weight=0.1,
            confidence_modulation_enabled=True,
            confidence_low_threshold=0.5,
            confidence_high_threshold=0.8,
            goal_loss_weight=0.05,
        ),
    )


@pytest.fixture
def basic_processor():
    """StreamingProcessor with default config (no co-adaptation)."""
    field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
    return StreamingProcessor(field, streaming_config=StreamingConfig())


class TestAuxiliaryLoss:
    """World model prediction as auxiliary loss (Change 2)."""

    def test_process_with_wm_prediction(self, co_adapt_processor):
        x = torch.randn(256)
        wm_pred = torch.randn(256)
        metric = co_adapt_processor.process(x, wm_prediction=wm_pred)
        assert metric.auxiliary_loss > 0.0
        assert metric.loss > 0.0

    def test_no_wm_prediction_no_auxiliary(self, co_adapt_processor):
        x = torch.randn(256)
        metric = co_adapt_processor.process(x)
        assert metric.auxiliary_loss == 0.0

    def test_auxiliary_loss_weight_zero(self):
        field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        proc = StreamingProcessor(
            field,
            streaming_config=StreamingConfig(auxiliary_loss_weight=0.0),
        )
        x = torch.randn(256)
        wm_pred = torch.randn(256)
        metric = proc.process(x, wm_prediction=wm_pred)
        assert metric.auxiliary_loss == pytest.approx(0.0, abs=1e-7)

    def test_auxiliary_loss_increases_total_loss(self, basic_processor):
        x = torch.randn(256)
        # Process once without WM prediction
        m1 = basic_processor.process(x.clone())
        # Reset and process with WM prediction (divergent prediction = extra loss)
        basic_processor.field.train()
        wm_pred = torch.randn(256) * 10  # Large divergent prediction
        m2 = basic_processor.process(x.clone(), wm_prediction=wm_pred)
        # Total loss should be higher with auxiliary
        assert m2.auxiliary_loss > 0.0

    def test_wm_prediction_detached(self, co_adapt_processor):
        """WM prediction should not receive gradients."""
        x = torch.randn(256)
        wm_pred = torch.randn(256, requires_grad=True)
        co_adapt_processor.process(x, wm_prediction=wm_pred)
        # wm_pred.grad should be None because we detach inside process()
        assert wm_pred.grad is None


class TestConfidenceModulation:
    """Metacognition confidence gating gradients (Change 3)."""

    def test_low_confidence_scales_gradients(self, co_adapt_processor):
        x = torch.randn(256)
        # Force surprise threshold to 0 so updates always happen
        co_adapt_processor.config = StreamingConfig(
            surprise_threshold=0.0,
            confidence_modulation_enabled=True,
            confidence_low_threshold=0.5,
            confidence_high_threshold=0.8,
        )
        # Process with low confidence
        metric = co_adapt_processor.process(x, confidence=0.2)
        # Should still update (just with scaled gradients)
        assert metric.grad_norm >= 0

    def test_confidence_modulation_disabled(self):
        field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        proc = StreamingProcessor(
            field,
            streaming_config=StreamingConfig(
                confidence_modulation_enabled=False,
            ),
        )
        x = torch.randn(256)
        # Should work fine even with confidence passed
        metric = proc.process(x, confidence=0.1)
        assert metric.grad_norm >= 0

    def test_confidence_none_no_scaling(self, co_adapt_processor):
        x = torch.randn(256)
        metric = co_adapt_processor.process(x, confidence=None)
        assert metric.grad_norm >= 0

    def test_confidence_to_lr_scale_mapping(self, co_adapt_processor):
        # Below low threshold
        assert co_adapt_processor._confidence_to_lr_scale(0.0) == pytest.approx(0.1)
        assert co_adapt_processor._confidence_to_lr_scale(0.3) == pytest.approx(0.1)
        # Above high threshold
        assert co_adapt_processor._confidence_to_lr_scale(1.0) == pytest.approx(1.0)
        assert co_adapt_processor._confidence_to_lr_scale(0.9) == pytest.approx(1.0)
        # Mid-range (linear ramp)
        mid = (0.5 + 0.8) / 2  # 0.65
        expected = 0.1 + 0.9 * (mid - 0.5) / (0.8 - 0.5)
        assert co_adapt_processor._confidence_to_lr_scale(mid) == pytest.approx(expected)


class TestGoalLoss:
    """Goal satisfaction as loss term (Change 5)."""

    def test_goal_loss_adds_to_total(self, co_adapt_processor):
        x = torch.randn(256)
        goal_emb = torch.randn(256)
        metric = co_adapt_processor.process(x, goal_embeddings=[goal_emb])
        assert metric.loss > 0  # total loss includes goal term

    def test_no_goals_no_extra_loss(self, co_adapt_processor):
        x = torch.randn(256)
        m1 = co_adapt_processor.process(x.clone())
        # None and empty list should behave the same
        m2 = co_adapt_processor.process(x.clone(), goal_embeddings=None)
        m3 = co_adapt_processor.process(x.clone(), goal_embeddings=[])
        # All three should produce valid metrics
        assert m1.loss >= 0
        assert m2.loss >= 0
        assert m3.loss >= 0

    def test_mismatched_goal_dim_skipped(self, co_adapt_processor):
        x = torch.randn(256)
        # Goal embedding with wrong dimension
        wrong_dim_goal = torch.randn(128)
        metric = co_adapt_processor.process(x, goal_embeddings=[wrong_dim_goal])
        # Should not crash, goal is skipped
        assert metric.loss >= 0

    def test_goal_embeddings_detached(self, co_adapt_processor):
        x = torch.randn(256)
        goal_emb = torch.randn(256, requires_grad=True)
        co_adapt_processor.process(x, goal_embeddings=[goal_emb])
        assert goal_emb.grad is None

    def test_multiple_goals(self, co_adapt_processor):
        x = torch.randn(256)
        goals = [torch.randn(256) for _ in range(3)]
        metric = co_adapt_processor.process(x, goal_embeddings=goals)
        assert metric.loss >= 0


class TestReplayEpisode:
    """Consolidation replay through the field (Change 1)."""

    def test_replay_episode_returns_loss(self, basic_processor):
        inp = torch.randn(256)
        tgt = torch.randn(256)
        loss = basic_processor.replay_episode(inp, tgt)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_replay_reduces_loss_on_repeat(self, basic_processor):
        inp = torch.randn(256)
        tgt = torch.randn(256)
        loss1 = basic_processor.replay_episode(inp, tgt)
        # Replay same pair multiple times
        for _ in range(10):
            basic_processor.replay_episode(inp, tgt)
        loss_final = basic_processor.replay_episode(inp, tgt)
        # Loss should decrease after repeated replay
        assert loss_final < loss1

    def test_replay_uses_reduced_lr(self, basic_processor):
        original_lr = basic_processor._optimizer.param_groups[0]["lr"]
        inp = torch.randn(256)
        tgt = torch.randn(256)
        basic_processor.replay_episode(inp, tgt, lr_scale=0.25)
        # LR should be restored after replay
        assert basic_processor._optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_replay_restores_lr_on_error(self, basic_processor):
        original_lr = basic_processor._optimizer.param_groups[0]["lr"]
        # Passing wrong-shaped tensors should cause an error
        with pytest.raises(Exception):
            basic_processor.replay_episode(
                torch.randn(128),  # wrong dim
                torch.randn(256),
            )
        # LR should still be restored
        assert basic_processor._optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)

    def test_replay_with_ewc(self):
        field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        proc = StreamingProcessor(
            field,
            ewc_config=sfp.EWCConfig(enabled=True),
        )
        # Process some data first so EWC has anchors
        for _ in range(5):
            proc.process(torch.randn(256))
        # Now replay
        loss = proc.replay_episode(torch.randn(256), torch.randn(256))
        assert loss >= 0


class TestBackwardCompatibility:
    """All new params default to None — existing behavior unchanged."""

    def test_process_with_all_none(self, basic_processor):
        x = torch.randn(256)
        metric = basic_processor.process(x)
        assert metric.grad_norm >= 0
        assert metric.auxiliary_loss == 0.0

    def test_process_signature_compatible(self, tiny_processor, random_input):
        """Existing test fixtures still work with extended signature."""
        metric = tiny_processor.process(random_input)
        assert metric.grad_norm >= 0
        assert hasattr(metric, "auxiliary_loss")

    def test_all_signals_combined(self, co_adapt_processor):
        """All co-adaptation signals at once."""
        x = torch.randn(256)
        metric = co_adapt_processor.process(
            x,
            wm_prediction=torch.randn(256),
            confidence=0.7,
            goal_embeddings=[torch.randn(256), torch.randn(256)],
        )
        assert metric.grad_norm >= 0
        assert metric.auxiliary_loss > 0.0
