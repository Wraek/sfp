"""Tests for forgetting strategies."""

import torch
import pytest

from sfp.config import EWCConfig
from sfp.core.forgetting import EWCStrategy, WeightDecayStrategy


class TestWeightDecayStrategy:
    def test_penalty_is_zero(self, tiny_field):
        strategy = WeightDecayStrategy()
        penalty = strategy.penalty(tiny_field)
        assert penalty.item() == 0.0

    def test_update_importance_noop(self, tiny_field):
        strategy = WeightDecayStrategy()
        strategy.update_importance(tiny_field)  # Should not raise


class TestEWCStrategy:
    def test_initial_penalty_is_zero(self, tiny_field):
        config = EWCConfig(lambda_=1000.0)
        ewc = EWCStrategy(tiny_field, config)
        penalty = ewc.penalty(tiny_field)
        # At initialization, params == anchors, so penalty should be ~0
        assert penalty.item() < 1e-6

    def test_penalty_increases_after_update(self, tiny_field):
        config = EWCConfig(lambda_=1000.0)
        ewc = EWCStrategy(tiny_field, config)

        # Simulate a gradient step
        x = torch.randn(256)
        y = tiny_field(x)
        loss = y.pow(2).sum()
        loss.backward()

        # Update importance (Fisher)
        ewc.update_importance(tiny_field)

        # Modify parameters
        with torch.no_grad():
            for p in tiny_field.parameters():
                p.data += 0.1

        # Now penalty should be > 0
        penalty = ewc.penalty(tiny_field)
        assert penalty.item() > 0

    def test_update_anchors(self, tiny_field):
        config = EWCConfig()
        ewc = EWCStrategy(tiny_field, config)

        # Modify params
        with torch.no_grad():
            for p in tiny_field.parameters():
                p.data += 0.1

        ewc.update_anchors(tiny_field)

        # After anchor update, penalty should be ~0 again
        penalty = ewc.penalty(tiny_field)
        assert penalty.item() < 1e-6
