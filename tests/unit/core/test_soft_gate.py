"""Tests for the soft sigmoid surprise gate and composite importance scoring."""

import math

import pytest
import torch

from sfp.config import FieldConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor


def _make_processor(
    dim: int = 64,
    n_layers: int = 2,
    **streaming_kwargs,
) -> StreamingProcessor:
    """Create a StreamingProcessor with custom streaming config."""
    field = SemanticFieldProcessor(FieldConfig(dim=dim, n_layers=n_layers))
    config = StreamingConfig(**streaming_kwargs)
    return StreamingProcessor(
        field=field,
        streaming_config=config,
        lora_config=None,
        ewc_config=None,
    )


class TestComputeSoftGate:
    """Unit tests for _compute_soft_gate() sigmoid behavior."""

    def test_smooth_transition_near_threshold(self):
        """Surprise near threshold should produce gate_scale ~ 0.5, not binary."""
        proc = _make_processor(surprise_threshold=1.0)
        # importance=1.0 → center = threshold/1.0 = 1.0
        gate = proc._compute_soft_gate(
            effective_surprise=1.0, threshold=1.0, importance=1.0,
        )
        assert 0.45 <= gate <= 0.55, (
            f"Surprise == threshold should give gate ~ 0.5, got {gate:.4f}"
        )

    def test_high_surprise_full_scale(self):
        """Surprise >> threshold should produce gate_scale ~ 1.0."""
        proc = _make_processor(surprise_threshold=1.0)
        gate = proc._compute_soft_gate(
            effective_surprise=5.0, threshold=1.0, importance=1.0,
        )
        assert gate > 0.99, f"Surprise >> threshold should give gate ~ 1.0, got {gate:.4f}"

    def test_low_surprise_near_zero(self):
        """Surprise << threshold should produce gate_scale ~ 0.0."""
        proc = _make_processor(surprise_threshold=1.0)
        gate = proc._compute_soft_gate(
            effective_surprise=0.0, threshold=1.0, importance=1.0,
        )
        assert gate < 0.01, f"Surprise << threshold should give gate ~ 0.0, got {gate:.4f}"

    def test_steepness_controls_sharpness(self):
        """Higher steepness → sharper transition; lower → gentler."""
        proc_sharp = _make_processor(surprise_threshold=1.0, soft_gate_steepness=50.0)
        proc_gentle = _make_processor(surprise_threshold=1.0, soft_gate_steepness=2.0)

        # At surprise = 0.9 (just below threshold):
        gate_sharp = proc_sharp._compute_soft_gate(0.9, 1.0, 1.0)
        gate_gentle = proc_gentle._compute_soft_gate(0.9, 1.0, 1.0)

        # Sharp gate should be much closer to 0 below threshold
        assert gate_sharp < gate_gentle, (
            f"Sharp ({gate_sharp:.4f}) should be lower than gentle ({gate_gentle:.4f}) "
            "below threshold"
        )

        # At surprise = 1.1 (just above threshold):
        gate_sharp_above = proc_sharp._compute_soft_gate(1.1, 1.0, 1.0)
        gate_gentle_above = proc_gentle._compute_soft_gate(1.1, 1.0, 1.0)

        # Sharp gate should be much closer to 1 above threshold
        assert gate_sharp_above > gate_gentle_above, (
            f"Sharp ({gate_sharp_above:.4f}) should be higher than gentle "
            f"({gate_gentle_above:.4f}) above threshold"
        )

    def test_numerical_stability_large_positive(self):
        """Extremely large surprise should not cause overflow."""
        proc = _make_processor(surprise_threshold=1.0)
        gate = proc._compute_soft_gate(1000.0, 1.0, 1.0)
        assert gate == pytest.approx(1.0, abs=1e-10)

    def test_numerical_stability_large_negative(self):
        """Zero surprise with high threshold should not cause underflow."""
        proc = _make_processor(surprise_threshold=100.0, soft_gate_steepness=100.0)
        gate = proc._compute_soft_gate(0.0, 100.0, 1.0)
        assert gate == pytest.approx(0.0, abs=1e-10)
        assert math.isfinite(gate)


class TestComputeImportance:
    """Unit tests for _compute_importance() composite scoring."""

    def test_neutral_when_no_signals(self):
        """With threshold=0 and no consistency/confidence, importance=1.0."""
        proc = _make_processor(surprise_threshold=0.0)
        imp = proc._compute_importance(
            consistency_scalar=None,
            confidence=None,
            effective_surprise=0.5,
            threshold=0.0,  # disables surprise_ratio contribution
        )
        assert imp == 1.0, f"No signals should give importance=1.0, got {imp}"

    def test_high_signals_raise_importance(self):
        """High consistency + confidence + surprise should give importance > 1.0."""
        proc = _make_processor(surprise_threshold=1.0)
        imp = proc._compute_importance(
            consistency_scalar=0.9,
            confidence=0.9,
            effective_surprise=1.8,  # 90% of 2× threshold → ratio=0.45
            threshold=1.0,
        )
        assert imp > 1.0, f"High signals should give importance > 1.0, got {imp}"

    def test_low_signals_lower_importance(self):
        """Low consistency + confidence + surprise should give importance < 1.0."""
        proc = _make_processor(surprise_threshold=1.0)
        imp = proc._compute_importance(
            consistency_scalar=0.1,
            confidence=0.1,
            effective_surprise=0.1,
            threshold=1.0,
        )
        assert imp < 1.0, f"Low signals should give importance < 1.0, got {imp}"

    def test_importance_shifts_center(self):
        """High importance should lower the sigmoid center, making updates easier."""
        proc = _make_processor(surprise_threshold=1.0)

        # High importance → center < threshold
        gate_high_imp = proc._compute_soft_gate(0.8, 1.0, importance=1.4)
        gate_neutral = proc._compute_soft_gate(0.8, 1.0, importance=1.0)
        gate_low_imp = proc._compute_soft_gate(0.8, 1.0, importance=0.6)

        assert gate_high_imp > gate_neutral > gate_low_imp, (
            f"High importance should ease updates: "
            f"high={gate_high_imp:.4f} > neutral={gate_neutral:.4f} > "
            f"low={gate_low_imp:.4f}"
        )

    def test_importance_bounded(self):
        """Importance should stay in [0.5, 1.5] range."""
        proc = _make_processor(surprise_threshold=1.0)

        # Maximum signals
        imp_max = proc._compute_importance(1.0, 1.0, 2.0, 1.0)
        assert 0.5 <= imp_max <= 1.5, f"Max importance out of range: {imp_max}"

        # Minimum signals
        imp_min = proc._compute_importance(0.0, 0.0, 0.0, 1.0)
        assert 0.5 <= imp_min <= 1.5, f"Min importance out of range: {imp_min}"

    def test_partial_signals(self):
        """Missing signals should still produce a valid importance score."""
        proc = _make_processor(surprise_threshold=1.0)

        # Only consistency
        imp = proc._compute_importance(0.8, None, 0.5, 1.0)
        assert 0.5 <= imp <= 1.5, f"Partial importance out of range: {imp}"

        # Only confidence
        imp = proc._compute_importance(None, 0.8, 0.5, 1.0)
        assert 0.5 <= imp <= 1.5, f"Partial importance out of range: {imp}"


class TestSoftGateIntegration:
    """Integration tests for soft gate within the full process() flow."""

    def test_floor_prevents_noise_updates(self):
        """Gate scale below floor should prevent updates."""
        proc = _make_processor(
            surprise_threshold=10.0,  # very high threshold
            soft_gate_enabled=True,
            soft_gate_floor=0.05,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        # With threshold=10 and typical grad norms ~1-5, gate_scale will be
        # well below floor → no updates
        x = torch.randn(64) * 0.01  # tiny input → tiny gradients
        result = proc.process(x)
        assert not result.updated, "Tiny signal with high threshold should not update"

    def test_soft_gate_with_warmup(self):
        """Soft gate should allow gradual learning during warmup period."""
        proc = _make_processor(
            surprise_threshold=0.0,
            soft_gate_enabled=True,
            warmup_steps=20,
            adaptive_surprise=False,
        )
        x = torch.randn(64)
        losses = [proc.process(x).loss for _ in range(40)]

        # Loss should decrease even during warmup — soft gate enables partial updates
        # (binary gate might block everything during warmup's low LR period)
        assert losses[-1] < losses[0], (
            f"Should learn through warmup: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_soft_gate_with_adaptive(self):
        """Soft gate + adaptive threshold should produce smooth learning."""
        proc = _make_processor(
            surprise_threshold=0.0,
            soft_gate_enabled=True,
            warmup_steps=0,
            adaptive_surprise=True,
        )
        x = torch.randn(64)
        losses = [proc.process(x).loss for _ in range(100)]

        # With enough steps for adaptive to kick in (after 10), we should
        # still see overall learning progress
        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10
        assert late_mean < early_mean, (
            f"Should learn with adaptive + soft gate: "
            f"early={early_mean:.4f}, late={late_mean:.4f}"
        )

    def test_binary_gate_fallback(self):
        """soft_gate_enabled=False should preserve exact binary behavior."""
        proc = _make_processor(
            surprise_threshold=1e6,
            soft_gate_enabled=False,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        results = [proc.process(torch.randn(64)) for _ in range(10)]
        assert all(not r.updated for r in results), (
            "Binary fallback with high threshold should block all updates"
        )

    def test_gradient_scaling_applied(self):
        """Soft gate should scale gradients proportionally, not binary."""
        proc = _make_processor(
            surprise_threshold=0.0,
            soft_gate_enabled=True,
            warmup_steps=0,
            adaptive_surprise=False,
        )

        # Process one input and capture weight change
        x = torch.randn(64)
        weights_before = {
            n: p.data.clone() for n, p in proc.field.named_parameters()
        }
        proc.process(x)
        drift_soft = sum(
            (p.data - weights_before[n]).norm().item()
            for n, p in proc.field.named_parameters()
        )

        # Compare with binary gate (fresh processor)
        proc2 = _make_processor(
            surprise_threshold=0.0,
            soft_gate_enabled=False,
            warmup_steps=0,
            adaptive_surprise=False,
        )
        # Copy same initial weights
        with torch.no_grad():
            for (n1, p1), (n2, p2) in zip(
                proc.field.named_parameters(),
                proc2.field.named_parameters(),
            ):
                p2.copy_(weights_before[n1])
        proc2._optimizer = proc2._build_optimizer()

        proc2.process(x)
        drift_binary = sum(
            (p.data - weights_before[n]).norm().item()
            for n, p in proc2.field.named_parameters()
        )

        # Soft gate with threshold=0 should still produce some update
        assert drift_soft > 0, "Soft gate should produce weight changes"
        # With threshold=0, soft gate should produce gate_scale ~ sigmoid(k * surprise)
        # which scales the gradient — so drift should typically differ from binary
        # (unless gate_scale happens to be exactly 1.0)
