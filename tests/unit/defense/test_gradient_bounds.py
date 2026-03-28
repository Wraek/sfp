"""Tests for defense.gradient_bounds — AdaptiveGradientClipper and UpdateBudget."""

import torch
import torch.nn as nn
import pytest

from sfp.defense.gradient_bounds import AdaptiveGradientClipper, UpdateBudget


def _make_simple_model():
    """Create a tiny linear model for testing."""
    model = nn.Linear(16, 16, bias=False)
    nn.init.ones_(model.weight)
    return model


class TestAdaptiveGradientClipper:
    def test_first_call_initializes_ema(self):
        model = _make_simple_model()
        clipper = AdaptiveGradientClipper(model)
        # Set a gradient
        x = torch.randn(4, 16)
        y = model(x)
        y.sum().backward()
        frac = clipper.clip(model)
        # First call initializes EMA — no clipping expected
        assert frac == 0.0

    def test_normal_gradients_low_clip_fraction(self):
        model = _make_simple_model()
        clipper = AdaptiveGradientClipper(model, clip_multiplier=3.0)
        # Build EMA with normal gradients
        for _ in range(10):
            model.zero_grad()
            x = torch.randn(4, 16)
            model(x).sum().backward()
            clipper.clip(model)
        # Normal gradient should have low clip fraction
        model.zero_grad()
        model(torch.randn(4, 16)).sum().backward()
        frac = clipper.clip(model)
        assert frac < 0.5

    def test_spike_gradient_gets_clipped(self):
        model = _make_simple_model()
        clipper = AdaptiveGradientClipper(model, clip_multiplier=2.0, ema_decay=0.9)
        # Build EMA with small gradients
        for _ in range(20):
            model.zero_grad()
            x = torch.randn(4, 16) * 0.01
            model(x).sum().backward()
            clipper.clip(model)
        # Now spike the gradient
        model.zero_grad()
        x = torch.randn(4, 16) * 100.0
        model(x).sum().backward()
        frac = clipper.clip(model)
        assert frac > 0.0  # Some elements should be clipped

    def test_no_grad_params_skipped(self):
        model = _make_simple_model()
        clipper = AdaptiveGradientClipper(model)
        # Don't set any gradients
        frac = clipper.clip(model)
        assert frac == 0.0


class TestUpdateBudget:
    def test_small_update_within_budget(self):
        model = _make_simple_model()
        budget = UpdateBudget(model, budget_fraction=0.1)
        # Tiny parameter change
        with torch.no_grad():
            model.weight.data += torch.randn_like(model.weight) * 1e-6
        exceeded = budget.enforce(model)
        assert not exceeded

    def test_large_update_exceeds_budget(self):
        model = _make_simple_model()
        budget = UpdateBudget(model, budget_fraction=0.001)
        # Large parameter change
        with torch.no_grad():
            model.weight.data += torch.randn_like(model.weight) * 10.0
        exceeded = budget.enforce(model)
        assert exceeded

    def test_enforce_scales_back_weights(self):
        model = _make_simple_model()
        budget = UpdateBudget(model, budget_fraction=0.001)
        original_weights = model.weight.data.clone()
        # Large update
        with torch.no_grad():
            model.weight.data += torch.ones_like(model.weight) * 5.0
        budget.enforce(model)
        # Weights should be closer to original than the raw update
        delta = (model.weight.data - original_weights).norm().item()
        assert delta < 5.0 * 16  # much less than the raw 5.0*16^0.5 per element

    def test_snapshot_updates_after_enforce(self):
        model = _make_simple_model()
        budget = UpdateBudget(model, budget_fraction=0.5)
        with torch.no_grad():
            model.weight.data += torch.randn_like(model.weight) * 0.001
        budget.enforce(model)
        # Second small change should not exceed
        with torch.no_grad():
            model.weight.data += torch.randn_like(model.weight) * 0.001
        exceeded = budget.enforce(model)
        assert not exceeded

    def test_budget_fraction_property(self):
        model = _make_simple_model()
        budget = UpdateBudget(model, budget_fraction=0.01)
        assert budget.budget_fraction == 0.01
