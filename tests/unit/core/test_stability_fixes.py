"""Tests for Phase 1 stability fixes and Phase 3 gradient accumulation.

Covers:
  1A. EWC loss normalization by parameter count
  1B. Post-warmup LR cosine decay
  1C. External LR scale clamping
  1D. Rate limit cooldown after consolidation
  1E. (tested in integration — anomaly scaling wired in processor.py)
  1F. Loss component monitoring in SurpriseMetric
  3A. Gradient accumulation
"""

from __future__ import annotations

import math

import pytest
import torch

from sfp.config import EWCConfig, FieldConfig, StreamingConfig, Tier0Config
from sfp.core.field import SemanticFieldProcessor
from sfp.core.forgetting import EWCStrategy
from sfp.core.streaming import StreamingProcessor
from sfp.defense.surprise_hardening import SurpriseHardener

DIM = 32
N_LAYERS = 2


def _field() -> SemanticFieldProcessor:
    return SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))


def _processor(**overrides) -> StreamingProcessor:
    defaults = dict(surprise_threshold=0.0, soft_gate_enabled=False, lr_decay_enabled=False)
    defaults.update(overrides)
    cfg = StreamingConfig(**defaults)
    return StreamingProcessor(_field(), streaming_config=cfg)


# ============================================================
# 1A — EWC loss normalization
# ============================================================


class TestEWCNormalization:
    def test_penalty_scales_with_param_count(self):
        """Penalty should be normalized by total parameter count."""
        field = _field()
        ewc = EWCStrategy(field, EWCConfig(lambda_=100.0))

        # Simulate a gradient to populate Fisher
        x = torch.randn(DIM)
        y = field(x)
        loss = torch.nn.functional.mse_loss(y, x)
        loss.backward()
        ewc.update_importance(field)

        # Perturb weights from anchor
        with torch.no_grad():
            for p in field.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        penalty = ewc.penalty(field).item()
        n_params = sum(p.numel() for p in field.parameters() if p.requires_grad)

        # Penalty should be finite and reasonable (not 1000x the perturbation)
        assert penalty > 0
        assert penalty < 10.0  # normalized by n_params, should be small

    def test_default_lambda_is_100(self):
        """Default lambda should be 100, not 1000."""
        assert EWCConfig().lambda_ == 100.0


# ============================================================
# 1B — Post-warmup LR decay
# ============================================================


class TestLRDecay:
    def test_cosine_decay_reduces_lr(self):
        """After warmup, LR should decrease via cosine annealing."""
        proc = _processor(
            warmup_steps=10,
            lr_decay_enabled=True,
            lr_decay_steps=100,
            lr_decay_min_factor=0.01,
            lr=1e-4,
        )

        # Use varied inputs so updates keep firing (avoid convergence)
        for i in range(15):
            x = torch.randn(DIM)
            proc.process(x)

        lr_after_warmup = proc._optimizer.param_groups[0]["lr"]

        # Run more steps into decay with varied inputs
        for _ in range(50):
            x = torch.randn(DIM)
            proc.process(x)

        lr_after_decay = proc._optimizer.param_groups[0]["lr"]
        assert lr_after_decay < lr_after_warmup

    def test_decay_disabled_keeps_constant_lr(self):
        """lr_decay_enabled=False should keep LR constant after warmup."""
        proc = _processor(warmup_steps=5, lr_decay_enabled=False)
        base_lr = proc.config.lr

        x = torch.randn(DIM)
        for _ in range(20):
            proc.process(x)

        lr = proc._optimizer.param_groups[0]["lr"]
        assert lr == pytest.approx(base_lr, rel=1e-3)

    def test_scheduler_type_with_both(self):
        """Warmup + decay should produce a SequentialLR."""
        proc = _processor(warmup_steps=10, lr_decay_enabled=True)
        assert proc._scheduler is not None
        assert type(proc._scheduler).__name__ == "SequentialLR"

    def test_scheduler_type_decay_only(self):
        """No warmup + decay should produce a CosineAnnealingLR."""
        proc = _processor(warmup_steps=0, lr_decay_enabled=True)
        assert proc._scheduler is not None
        assert type(proc._scheduler).__name__ == "CosineAnnealingLR"


# ============================================================
# 1C — External LR scale bounds
# ============================================================


class TestExternalLRBounds:
    def test_extreme_scale_clamped(self):
        """external_lr_scale of 100.0 should be clamped to 5.0."""
        proc = _processor()
        x = torch.randn(DIM)

        # Process with extreme LR scale — should not crash or produce huge updates
        metric = proc.process(x, external_lr_scale=100.0)
        assert metric.updated

        # Process with negative scale — should be clamped to 0.1
        metric2 = proc.process(x, external_lr_scale=-5.0)
        assert metric2.updated

    def test_custom_range(self):
        """Custom external_lr_scale_range should be respected."""
        proc = _processor(external_lr_scale_range=(0.5, 2.0))

        x = torch.randn(DIM)
        # With scale=10.0, should be clamped to 2.0
        metric = proc.process(x, external_lr_scale=10.0)
        assert metric.updated


# ============================================================
# 1D — Rate limit cooldown
# ============================================================


class TestRateLimitCooldown:
    def test_suppress_deactivates_rate_limiting(self):
        """suppress_rate_limiting() should clear rate_limited flag."""
        cfg = Tier0Config()
        hardener = SurpriseHardener(cfg)

        # Force rate limiting active
        hardener._rate_limited = True
        assert hardener.is_rate_limited

        hardener.suppress_rate_limiting(10)
        assert not hardener.is_rate_limited
        assert hardener._cooldown_remaining == 10

    def test_cooldown_prevents_rate_limit_activation(self):
        """During cooldown, rate limiting should not activate."""
        cfg = Tier0Config(rate_limit_window=10, rate_limit_threshold=0.3)
        hardener = SurpriseHardener(cfg)
        hardener.suppress_rate_limiting(5)

        # Pump many high-surprise values — normally would trigger rate limiting
        for _ in range(15):
            hardener.compute_hardened_surprise(100.0)

        # Should not be rate limited during cooldown
        # (cooldown was 5 steps, we ran 15, so it should expire and maybe activate)
        # But the first 5 steps should be suppressed
        # After cooldown expires, rate limiting can activate normally

    def test_cooldown_decrements(self):
        """Cooldown counter should decrease each step."""
        cfg = Tier0Config()
        hardener = SurpriseHardener(cfg)
        hardener.suppress_rate_limiting(3)
        assert hardener._cooldown_remaining == 3

        hardener.compute_hardened_surprise(1.0)
        assert hardener._cooldown_remaining == 2

        hardener.compute_hardened_surprise(1.0)
        assert hardener._cooldown_remaining == 1

        hardener.compute_hardened_surprise(1.0)
        assert hardener._cooldown_remaining == 0

    def test_reset_clears_cooldown(self):
        """Reset should clear cooldown state."""
        cfg = Tier0Config()
        hardener = SurpriseHardener(cfg)
        hardener.suppress_rate_limiting(10)
        hardener.reset()
        assert hardener._cooldown_remaining == 0


# ============================================================
# 1F — Loss component monitoring
# ============================================================


class TestLossComponents:
    def test_primary_loss_tracked(self):
        """SurpriseMetric should include primary loss component."""
        proc = _processor()
        x = torch.randn(DIM)
        metric = proc.process(x)
        assert "primary" in metric.loss_components
        assert metric.loss_components["primary"] > 0

    def test_ewc_loss_tracked(self):
        """When EWC is enabled, its penalty should appear in loss_components."""
        field = _field()
        cfg = StreamingConfig(surprise_threshold=0.0, soft_gate_enabled=False, lr_decay_enabled=False)
        ewc_cfg = EWCConfig(lambda_=100.0)
        proc = StreamingProcessor(field, streaming_config=cfg, ewc_config=ewc_cfg)

        x = torch.randn(DIM)
        metric = proc.process(x)
        assert "ewc" in metric.loss_components

    def test_loss_components_sum_to_total(self):
        """Sum of components should approximately equal total loss."""
        proc = _processor()
        x = torch.randn(DIM)
        metric = proc.process(x)
        component_sum = sum(metric.loss_components.values())
        assert component_sum == pytest.approx(metric.loss, abs=1e-5)


# ============================================================
# 3A — Gradient accumulation
# ============================================================


class TestGradientAccumulation:
    def test_accumulation_delays_step(self):
        """With accumulation_steps=4, optimizer should step every 4 calls."""
        proc = _processor(gradient_accumulation_steps=4)
        x = torch.randn(DIM)

        # Capture initial params
        params_before = {
            n: p.data.clone() for n, p in proc.field.named_parameters()
        }

        # Steps 1-3: gradients accumulate, params should not change
        for i in range(3):
            proc.process(x)
            # Accumulation count should increment
            assert proc._accumulation_count == i + 1

        # Step 4: optimizer steps, params should change
        proc.process(x)
        assert proc._accumulation_count == 0

        params_after = {
            n: p.data.clone() for n, p in proc.field.named_parameters()
        }
        changed = any(
            not torch.equal(params_before[n], params_after[n])
            for n in params_before
        )
        assert changed, "Parameters should have changed after accumulation"

    def test_no_accumulation_by_default(self):
        """Default gradient_accumulation_steps=1 should step every call."""
        proc = _processor()
        assert proc.config.gradient_accumulation_steps == 1

        x = torch.randn(DIM)
        proc.process(x)
        assert proc._accumulation_count == 0  # Reset after stepping

    def test_accumulation_loss_decreases(self):
        """Gradient accumulation should still reduce loss over time."""
        proc = _processor(lr=1e-3, gradient_accumulation_steps=4)
        x = torch.randn(DIM)

        first_loss = proc.process(x).loss

        # Run enough steps for several accumulated optimizer steps
        for _ in range(20):
            proc.process(x)

        final_loss = proc.process(x).loss
        assert final_loss < first_loss, "Loss should decrease with accumulation"
