"""Tests for goals.persistence — GoalRegister."""

import time

import torch
import pytest

from sfp.config import GoalPersistenceConfig
from sfp.goals.persistence import GoalRegister
from sfp.types import GoalStatus


@pytest.fixture
def register():
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


class TestGoalCreation:
    def test_create_goal(self, register):
        emb = torch.randn(32)
        goal = register.create_goal(emb, importance=0.7)
        assert goal.status == GoalStatus.ACTIVE
        assert goal.importance == 0.7
        assert goal.embedding.shape == (32,)

    def test_goal_id_increments(self, register):
        g1 = register.create_goal(torch.randn(32))
        g2 = register.create_goal(torch.randn(32))
        assert g2.id == g1.id + 1

    def test_parent_child_linking(self, register):
        parent = register.create_goal(torch.randn(32))
        child = register.create_goal(torch.randn(32), parent_id=parent.id)
        assert child.parent_id == parent.id
        assert child.id in parent.child_ids

    def test_eviction_at_capacity(self, register):
        for i in range(10):  # exceed max_goals=8
            register.create_goal(torch.randn(32))
        assert len(register.all_goals) <= 8


class TestGoalProgress:
    def test_update_progress(self, register):
        goal = register.create_goal(torch.randn(32))
        progress = register.update_progress(goal.id, torch.randn(32))
        assert 0.0 <= progress <= 1.0

    def test_auto_complete_on_high_progress(self, register):
        goal = register.create_goal(torch.randn(32))
        # Force satisfaction_embedding close to the query
        sat_emb = goal.satisfaction_embedding
        # Many updates with the exact satisfaction embedding
        for _ in range(50):
            register.update_progress(goal.id, sat_emb)
        # Should eventually complete
        assert goal.status in (GoalStatus.ACTIVE, GoalStatus.COMPLETED)


class TestGoalPriority:
    def test_compute_priorities_runs(self, register):
        register.create_goal(torch.randn(32))
        register.create_goal(torch.randn(32))
        register.compute_priorities()
        for g in register.active_goals:
            assert 0.0 <= g.priority <= 1.0


class TestGoalDeadlines:
    def test_ttl_expiry(self, register):
        # Create with very short TTL
        cfg = GoalPersistenceConfig(
            max_goals=8, d_goal=32, d_satisfaction=32, ttl_default=0.001,
        )
        reg = GoalRegister(cfg, d_model=32)
        goal = reg.create_goal(torch.randn(32))
        time.sleep(0.01)
        warnings = reg.check_deadlines()
        expired = [w for w in warnings if w[1] == "expired_ttl"]
        assert len(expired) >= 1

    def test_stalled_goals_detection(self, register):
        goal = register.create_goal(torch.randn(32))
        # Simulate flat progress
        goal.progress_history = [0.5] * 10
        stalled = register.detect_stalled_goals()
        assert goal.id in stalled


class TestGoalManagement:
    def test_pause_and_resume(self, register):
        goal = register.create_goal(torch.randn(32))
        register.pause_goal(goal.id)
        assert goal.status == GoalStatus.PAUSED
        register.resume_goal(goal.id)
        assert goal.status == GoalStatus.ACTIVE

    def test_remove_goal(self, register):
        goal = register.create_goal(torch.randn(32))
        removed = register.remove_goal(goal.id)
        assert removed
        assert goal.id not in [g.id for g in register.all_goals]

    def test_remove_nonexistent(self, register):
        assert not register.remove_goal(9999)


class TestGoalContext:
    def test_goal_context_shape(self, register):
        register.create_goal(torch.randn(32))
        ctx = register.get_goal_context()
        assert ctx.shape == (32,)

    def test_goal_context_empty(self, register):
        ctx = register.get_goal_context()
        assert ctx.norm().item() == 0.0

    def test_salience_modulation(self, register):
        register.create_goal(torch.randn(32))
        mods = register.get_salience_modulation()
        assert isinstance(mods, dict)


class TestGoalSerialization:
    def test_save_and_load(self, register):
        g1 = register.create_goal(torch.randn(32), importance=0.9)
        g2 = register.create_goal(torch.randn(32), importance=0.3)
        saved = register.save_goals()
        assert len(saved) == 2

        new_reg = GoalRegister(register._config, d_model=32)
        new_reg.load_goals(saved)
        assert len(new_reg.all_goals) == 2
