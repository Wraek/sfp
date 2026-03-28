"""Tests for goal persistence fixes: satisfaction hindsight and config fields."""

import torch
import pytest

from sfp.config import GoalPersistenceConfig
from sfp.goals.persistence import GoalRegister
from sfp.types import GoalStatus


@pytest.fixture
def register():
    """GoalRegister with d_model=32 for fast tests."""
    return GoalRegister(
        GoalPersistenceConfig(
            max_goals=8,
            d_goal=32,
            d_satisfaction=32,
            max_subgoals=3,
            satisfaction_threshold=0.95,
            stall_steps=5,
            stall_threshold=0.01,
            ttl_default=3600.0,
        ),
        d_model=32,
    )


# ------------------------------------------------------------------
# 1. Satisfaction hindsight training changes the embedding
# ------------------------------------------------------------------


class TestSatisfactionHindsightTraining:
    def test_satisfaction_embedding_changes_after_hindsight(self, register):
        """train_satisfaction_hindsight should update the goal's satisfaction_embedding."""
        goal = register.create_goal(torch.randn(32))
        original_sat = goal.satisfaction_embedding.clone()

        observation = torch.randn(32)
        register.train_satisfaction_hindsight(goal.id, observation)

        # The satisfaction embedding should have changed
        assert not torch.allclose(
            goal.satisfaction_embedding, original_sat, atol=1e-7
        ), "satisfaction_embedding should change after hindsight training"

    def test_hindsight_reduces_prediction_loss(self, register):
        """After repeated hindsight training, the MSE between the encoder
        output and the observation target should decrease."""
        goal = register.create_goal(torch.randn(32))
        observation = torch.randn(32)

        # Measure initial MSE from satisfaction_encoder
        with torch.no_grad():
            pred_before = register.satisfaction_encoder(goal.embedding.detach())
            loss_before = torch.nn.functional.mse_loss(
                pred_before, observation
            ).item()

        # Multiple hindsight steps
        for _ in range(50):
            register.train_satisfaction_hindsight(goal.id, observation)

        # Measure MSE after training
        with torch.no_grad():
            pred_after = register.satisfaction_encoder(goal.embedding.detach())
            loss_after = torch.nn.functional.mse_loss(
                pred_after, observation
            ).item()

        assert loss_after < loss_before, (
            f"After 50 hindsight steps, MSE loss should decrease "
            f"(before={loss_before:.4f}, after={loss_after:.4f})"
        )

    def test_hindsight_returns_none(self, register):
        """train_satisfaction_hindsight should return None."""
        goal = register.create_goal(torch.randn(32))
        result = register.train_satisfaction_hindsight(goal.id, torch.randn(32))
        assert result is None


# ------------------------------------------------------------------
# 2. Hindsight only for active goals
# ------------------------------------------------------------------


class TestHindsightOnlyActiveGoals:
    def test_hindsight_noop_for_completed_goal(self, register):
        """train_satisfaction_hindsight should not change a completed goal."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.COMPLETED
        original_sat = goal.satisfaction_embedding.clone()

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert torch.allclose(
            goal.satisfaction_embedding, original_sat
        ), "completed goal's satisfaction_embedding should not change"

    def test_hindsight_noop_for_expired_goal(self, register):
        """train_satisfaction_hindsight should not change an expired goal."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.EXPIRED
        original_sat = goal.satisfaction_embedding.clone()

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert torch.allclose(
            goal.satisfaction_embedding, original_sat
        ), "expired goal's satisfaction_embedding should not change"

    def test_hindsight_noop_for_failed_goal(self, register):
        """train_satisfaction_hindsight should not change a failed goal."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.FAILED
        original_sat = goal.satisfaction_embedding.clone()

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert torch.allclose(
            goal.satisfaction_embedding, original_sat
        ), "failed goal's satisfaction_embedding should not change"

    def test_hindsight_noop_for_paused_goal(self, register):
        """train_satisfaction_hindsight should not change a paused goal."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.PAUSED
        original_sat = goal.satisfaction_embedding.clone()

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert torch.allclose(
            goal.satisfaction_embedding, original_sat
        ), "paused goal's satisfaction_embedding should not change"

    def test_hindsight_noop_for_blocked_goal(self, register):
        """train_satisfaction_hindsight should not change a blocked goal."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.BLOCKED
        original_sat = goal.satisfaction_embedding.clone()

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert torch.allclose(
            goal.satisfaction_embedding, original_sat
        ), "blocked goal's satisfaction_embedding should not change"

    def test_hindsight_noop_for_nonexistent_goal(self, register):
        """train_satisfaction_hindsight should silently do nothing for invalid IDs."""
        # Should not raise
        register.train_satisfaction_hindsight(9999, torch.randn(32))

    def test_hindsight_works_for_active_goal(self, register):
        """Confirm that hindsight does work when the goal is active."""
        goal = register.create_goal(torch.randn(32))
        assert goal.status == GoalStatus.ACTIVE

        original_sat = goal.satisfaction_embedding.clone()
        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        assert not torch.allclose(
            goal.satisfaction_embedding, original_sat, atol=1e-7
        ), "active goal's satisfaction_embedding should change after hindsight"


# ------------------------------------------------------------------
# 3. Goal urgency LR config fields
# ------------------------------------------------------------------


class TestGoalUrgencyLRConfig:
    def test_goal_urgency_lr_enabled_field_exists(self):
        """GoalPersistenceConfig should have goal_urgency_lr_enabled."""
        cfg = GoalPersistenceConfig()
        assert hasattr(cfg, "goal_urgency_lr_enabled")

    def test_goal_urgency_lr_enabled_default_true(self):
        """goal_urgency_lr_enabled should default to True."""
        cfg = GoalPersistenceConfig()
        assert cfg.goal_urgency_lr_enabled is True

    def test_goal_urgency_max_multiplier_field_exists(self):
        """GoalPersistenceConfig should have goal_urgency_max_multiplier."""
        cfg = GoalPersistenceConfig()
        assert hasattr(cfg, "goal_urgency_max_multiplier")

    def test_goal_urgency_max_multiplier_default(self):
        """goal_urgency_max_multiplier should default to 2.0."""
        cfg = GoalPersistenceConfig()
        assert cfg.goal_urgency_max_multiplier == 2.0

    def test_goal_urgency_lr_enabled_can_be_set_false(self):
        """goal_urgency_lr_enabled should be settable to False at construction."""
        cfg = GoalPersistenceConfig(goal_urgency_lr_enabled=False)
        assert cfg.goal_urgency_lr_enabled is False

    def test_goal_urgency_max_multiplier_custom_value(self):
        """goal_urgency_max_multiplier should accept custom values."""
        cfg = GoalPersistenceConfig(goal_urgency_max_multiplier=3.5)
        assert cfg.goal_urgency_max_multiplier == 3.5


# ------------------------------------------------------------------
# 4. Satisfaction hindsight config fields
# ------------------------------------------------------------------


class TestSatisfactionHindsightConfig:
    def test_satisfaction_hindsight_enabled_field_exists(self):
        """GoalPersistenceConfig should have satisfaction_hindsight_enabled."""
        cfg = GoalPersistenceConfig()
        assert hasattr(cfg, "satisfaction_hindsight_enabled")

    def test_satisfaction_hindsight_enabled_default_true(self):
        """satisfaction_hindsight_enabled should default to True."""
        cfg = GoalPersistenceConfig()
        assert cfg.satisfaction_hindsight_enabled is True

    def test_satisfaction_hindsight_threshold_field_exists(self):
        """GoalPersistenceConfig should have satisfaction_hindsight_threshold."""
        cfg = GoalPersistenceConfig()
        assert hasattr(cfg, "satisfaction_hindsight_threshold")

    def test_satisfaction_hindsight_threshold_default(self):
        """satisfaction_hindsight_threshold should default to 0.8."""
        cfg = GoalPersistenceConfig()
        assert cfg.satisfaction_hindsight_threshold == 0.8

    def test_satisfaction_hindsight_enabled_can_be_set_false(self):
        """satisfaction_hindsight_enabled should be settable to False at construction."""
        cfg = GoalPersistenceConfig(satisfaction_hindsight_enabled=False)
        assert cfg.satisfaction_hindsight_enabled is False

    def test_satisfaction_hindsight_threshold_custom_value(self):
        """satisfaction_hindsight_threshold should accept custom values."""
        cfg = GoalPersistenceConfig(satisfaction_hindsight_threshold=0.6)
        assert cfg.satisfaction_hindsight_threshold == 0.6


# ------------------------------------------------------------------
# 5. Hindsight trains the satisfaction encoder
# ------------------------------------------------------------------


class TestHindsightTrainsEncoder:
    def test_satisfaction_encoder_weights_change(self, register):
        """After train_satisfaction_hindsight, the satisfaction_encoder weights
        should have been updated by the manual SGD step."""
        goal = register.create_goal(torch.randn(32))

        # Snapshot all satisfaction_encoder parameters before training
        params_before = [
            p.clone() for p in register.satisfaction_encoder.parameters()
        ]

        observation = torch.randn(32)
        register.train_satisfaction_hindsight(goal.id, observation)

        # At least one parameter tensor should have changed
        any_changed = False
        for p_before, p_after in zip(
            params_before, register.satisfaction_encoder.parameters()
        ):
            if not torch.allclose(p_before, p_after, atol=1e-10):
                any_changed = True
                break

        assert any_changed, (
            "satisfaction_encoder weights should change after hindsight training"
        )

    def test_encoder_weights_unchanged_for_inactive_goal(self, register):
        """If the goal is not active, the encoder weights should stay the same."""
        goal = register.create_goal(torch.randn(32))
        goal.status = GoalStatus.COMPLETED

        params_before = [
            p.clone() for p in register.satisfaction_encoder.parameters()
        ]

        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        for p_before, p_after in zip(
            params_before, register.satisfaction_encoder.parameters()
        ):
            assert torch.allclose(p_before, p_after), (
                "encoder weights should not change for inactive goals"
            )

    def test_multiple_hindsight_steps_continue_training(self, register):
        """Multiple calls to train_satisfaction_hindsight should keep updating
        the encoder weights (not just the first call)."""
        goal = register.create_goal(torch.randn(32))
        observation = torch.randn(32)

        # First training step
        register.train_satisfaction_hindsight(goal.id, observation)
        params_after_first = [
            p.clone() for p in register.satisfaction_encoder.parameters()
        ]

        # Second training step
        register.train_satisfaction_hindsight(goal.id, observation)
        params_after_second = [
            p.clone() for p in register.satisfaction_encoder.parameters()
        ]

        # Weights should differ between the first and second step
        any_changed = False
        for p1, p2 in zip(params_after_first, params_after_second):
            if not torch.allclose(p1, p2, atol=1e-10):
                any_changed = True
                break

        assert any_changed, (
            "encoder weights should continue to change across multiple "
            "hindsight training steps"
        )

    def test_encoder_gradients_are_zeroed_after_step(self, register):
        """After a hindsight step, all gradients on the satisfaction_encoder
        should be zeroed (cleaned up by the manual SGD implementation)."""
        goal = register.create_goal(torch.randn(32))
        register.train_satisfaction_hindsight(goal.id, torch.randn(32))

        for p in register.satisfaction_encoder.parameters():
            if p.grad is not None:
                assert torch.allclose(
                    p.grad, torch.zeros_like(p.grad)
                ), "gradients should be zeroed after hindsight step"
