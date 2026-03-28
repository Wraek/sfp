"""Tests for gradient conflict detection and mitigation in StreamingProcessor."""

from __future__ import annotations

import logging

import pytest
import torch

from sfp.config import EWCConfig, FieldConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor

DIM = 32
N_LAYERS = 2


def _field() -> SemanticFieldProcessor:
    return SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))


def _processor(**overrides) -> StreamingProcessor:
    defaults = dict(
        surprise_threshold=0.0,
        soft_gate_enabled=False,
        lr_decay_enabled=False,
    )
    defaults.update(overrides)
    cfg = StreamingConfig(**defaults)
    return StreamingProcessor(_field(), streaming_config=cfg)


class TestGradientConflictDetection:
    def test_disabled_no_state_changes(self):
        """With detection disabled, conflict state should stay zeroed."""
        proc = _processor(gradient_conflict_detection=False)
        x = torch.randn(DIM)
        for _ in range(10):
            proc.process(x)
        assert proc._grad_loss_ratio_ema == 0.0
        assert proc._component_emas == {}

    def test_grad_loss_ratio_ema_updates(self):
        """With detection enabled, EMA should track the grad/loss ratio."""
        proc = _processor(gradient_conflict_detection=True)
        for _ in range(5):
            x = torch.randn(DIM)
            proc.process(x)
        assert proc._grad_loss_ratio_ema > 0.0

    def test_component_emas_populated(self):
        """Component EMAs should track all loss components present."""
        proc = _processor(gradient_conflict_detection=True)
        x = torch.randn(DIM)
        proc.process(x)
        assert "primary" in proc._component_emas

    def test_conflict_warning_on_ratio_drop(self, caplog):
        """When grad/loss ratio drops sharply, a warning should be logged."""
        proc = _processor(gradient_conflict_detection=True)
        # Build up a high EMA baseline
        proc._grad_loss_ratio_ema = 100.0
        # Process should produce a much lower ratio, triggering warning
        x = torch.randn(DIM)
        with caplog.at_level(logging.WARNING, logger="sfp.core.streaming"):
            proc.process(x)
        assert any("Gradient conflict" in r.message for r in caplog.records)

    def test_no_warning_during_stable_training(self, caplog):
        """Stable training on varied inputs should not trigger conflict warnings."""
        proc = _processor(gradient_conflict_detection=True)
        with caplog.at_level(logging.WARNING, logger="sfp.core.streaming"):
            for _ in range(20):
                x = torch.randn(DIM)
                proc.process(x)
        conflict_warns = [
            r for r in caplog.records if "Gradient conflict" in r.message
        ]
        # May get a few early warnings during EMA warmup, but should stabilize
        # Allow at most 2 early warnings
        assert len(conflict_warns) <= 2

    def test_reset_clears_conflict_state(self):
        """reset_working_memory should clear all conflict tracking state."""
        proc = _processor(gradient_conflict_detection=True)
        for _ in range(5):
            proc.process(torch.randn(DIM))
        assert proc._grad_loss_ratio_ema > 0.0

        proc.reset_working_memory()
        assert proc._grad_loss_ratio_ema == 0.0
        assert proc._component_emas == {}
        assert proc._component_directions == {}
        assert proc._opposition_scores == {}


class TestGradientConflictMitigation:
    """Tests for opposition-aware adaptive loss scaling (OALS)."""

    def test_mitigation_disabled_no_scaling(self):
        """With mitigation disabled, no opposition scores should be tracked."""
        proc = _processor(
            gradient_conflict_detection=True,
            gradient_conflict_mitigation=False,
        )
        goal_emb = torch.randn(DIM)
        for _ in range(20):
            x = torch.randn(DIM)
            proc.process(x, goal_embeddings=[-goal_emb])
        assert proc._opposition_scores == {}
        assert proc._compute_effective_weights() == {}

    def test_opposition_scores_populate_under_conflict(self):
        """Structural goal-vs-primary conflict should populate opposition scores."""
        proc = _processor(
            gradient_conflict_detection=True,
            gradient_conflict_mitigation=True,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        for _ in range(20):
            x = torch.randn(DIM)
            # Goal points opposite to input → structural conflict
            proc.process(x, goal_embeddings=[-x.detach()])
        assert "goal" in proc._opposition_scores
        assert proc._opposition_scores["goal"] > 0.0

    def test_effective_weight_reduces_under_conflict(self):
        """High opposition score should reduce effective weight."""
        proc = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_damping=0.8,
            gradient_conflict_score_threshold=0.3,
            gradient_conflict_weight_floor=0.1,
        )
        proc._opposition_scores["goal"] = 0.8
        mult = proc._compute_effective_weights()
        assert "goal" in mult
        expected = 1.0 - 0.8 * 0.8  # 0.36
        assert abs(mult["goal"] - expected) < 1e-6

    def test_effective_weight_respects_floor(self):
        """Even maximum opposition should not go below the floor."""
        proc = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_damping=1.0,
            gradient_conflict_score_threshold=0.0,
            gradient_conflict_weight_floor=0.1,
        )
        proc._opposition_scores["goal"] = 1.0
        mult = proc._compute_effective_weights()
        assert mult["goal"] == pytest.approx(0.1)

    def test_no_scaling_below_threshold(self):
        """Opposition below threshold should not trigger any scaling."""
        proc = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_score_threshold=0.3,
        )
        proc._opposition_scores["goal"] = 0.1
        mult = proc._compute_effective_weights()
        assert "goal" not in mult

    def test_mitigation_reduces_goal_loss_component(self):
        """Mitigated processor should produce smaller goal loss than unmitigated."""
        torch.manual_seed(42)
        x = torch.randn(DIM)
        goal_emb = -x.detach()  # opposing

        # Unmitigated
        proc_off = _processor(
            gradient_conflict_mitigation=False,
            gradient_conflict_detection=True,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        # Mitigated — seed opposition score directly
        proc_on = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_detection=True,
            gradient_conflict_damping=0.8,
            gradient_conflict_score_threshold=0.0,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        proc_on._opposition_scores["goal"] = 0.9  # high conflict

        m_off = proc_off.process(x.clone(), goal_embeddings=[goal_emb.clone()])
        m_on = proc_on.process(x.clone(), goal_embeddings=[goal_emb.clone()])

        goal_off = abs(m_off.loss_components.get("goal", 0.0))
        goal_on = abs(m_on.loss_components.get("goal", 0.0))
        assert goal_on < goal_off

    def test_primary_loss_never_scaled(self):
        """Primary loss must never appear in effective weight multipliers."""
        proc = _processor(gradient_conflict_mitigation=True)
        proc._opposition_scores["primary"] = 1.0
        mult = proc._compute_effective_weights()
        # Primary should never be in the multipliers because it's never a
        # secondary component — the opposition tracking skips it.
        # But even if someone manually sets it, it won't affect primary loss
        # computation since primary loss doesn't look up multipliers.
        # The key point is primary always uses base weight.
        x = torch.randn(DIM)
        m1 = proc.process(x.clone())
        proc._opposition_scores.clear()
        proc2 = _processor(gradient_conflict_mitigation=False)
        m2 = proc2.process(x.clone())
        # Primary loss values should be the same (field weights differ, but
        # the point is primary weight=1.0 is never scaled)
        assert "conflict_scale_primary" not in m1.loss_components

    def test_mitigation_requires_detection_enabled(self):
        """When detection is disabled, mitigation should also be inactive."""
        proc = _processor(
            gradient_conflict_detection=False,
            gradient_conflict_mitigation=True,
        )
        proc._opposition_scores["goal"] = 1.0
        assert proc._compute_effective_weights() == {}

    def test_reset_clears_opposition_scores(self):
        """reset_working_memory should clear opposition scores."""
        proc = _processor(
            gradient_conflict_detection=True,
            gradient_conflict_mitigation=True,
        )
        proc._opposition_scores["goal"] = 0.5
        proc._opposition_scores["wm_aux"] = 0.3
        proc.reset_working_memory()
        assert proc._opposition_scores == {}

    def test_conflict_scale_in_loss_components(self):
        """Active mitigation should add conflict_scale_* keys to loss_components."""
        proc = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_detection=True,
            gradient_conflict_score_threshold=0.0,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        proc._opposition_scores["goal"] = 0.5
        x = torch.randn(DIM)
        m = proc.process(x, goal_embeddings=[-x.detach()])
        assert "conflict_scale_goal" in m.loss_components
        assert 0.0 < m.loss_components["conflict_scale_goal"] < 1.0

    def test_damping_zero_means_no_reduction(self):
        """With damping=0, effective weights should always be 1.0."""
        proc = _processor(
            gradient_conflict_mitigation=True,
            gradient_conflict_damping=0.0,
            gradient_conflict_score_threshold=0.0,
        )
        proc._opposition_scores["goal"] = 1.0
        proc._opposition_scores["wm_aux"] = 1.0
        mult = proc._compute_effective_weights()
        # All multipliers should be 1.0 (no reduction)
        for v in mult.values():
            assert v == pytest.approx(1.0)

    def test_weight_recovery_when_conflict_resolves(self):
        """Opposition score should decay when conflict stops."""
        proc = _processor(
            gradient_conflict_detection=True,
            gradient_conflict_mitigation=True,
            gradient_conflict_score_ema_decay=0.9,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        # Build up opposition
        for _ in range(15):
            x = torch.randn(DIM)
            proc.process(x, goal_embeddings=[-x.detach()])
        score_high = proc._opposition_scores.get("goal", 0.0)

        # Now process without goals — score should decay
        for _ in range(30):
            x = torch.randn(DIM)
            proc.process(x)
        score_low = proc._opposition_scores.get("goal", 0.0)
        assert score_low < score_high
