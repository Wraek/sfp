"""Tests for interface — SFPInterface thread-safe facade."""

import threading

import pytest
import torch

from sfp.config import (
    ConsolidationConfig,
    FieldConfig,
    GoalPersistenceConfig,
    StreamingConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
    ValenceConfig,
)
from sfp.interface import SFPInterface
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import (
    ConsolidationMode,
    GoalStatus,
    ReasoningResult,
    SurpriseMetric,
    ValenceSignal,
)

D = 64  # tiny dimension for fast tests


@pytest.fixture
def processor():
    """Minimal HierarchicalMemoryProcessor for interface tests."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=Tier1Config(
            hot_capacity=20, cold_capacity=40, surprise_threshold=0.0,
        ),
        tier2_config=Tier2Config(n_slots=16, d_value=D),
        tier3_config=Tier3Config(n_slots=8, d_value=D),
        streaming_config=StreamingConfig(surprise_threshold=0.0),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_with_modules():
    """HierarchicalMemoryProcessor with valence and goals enabled."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=Tier1Config(
            hot_capacity=20, cold_capacity=40, surprise_threshold=0.0,
        ),
        tier2_config=Tier2Config(n_slots=16, d_value=D),
        tier3_config=Tier3Config(n_slots=8, d_value=D),
        streaming_config=StreamingConfig(surprise_threshold=0.0),
        lora_config=None,
        ewc_config=None,
        valence_config=ValenceConfig(),
        goal_config=GoalPersistenceConfig(d_goal=D, d_satisfaction=D),
        device="cpu",
    )


@pytest.fixture
def iface(processor):
    return SFPInterface(processor)


@pytest.fixture
def iface_modules(processor_with_modules):
    return SFPInterface(processor_with_modules)


class TestConstruction:
    def test_wraps_processor(self, iface, processor):
        assert iface.processor is processor

    def test_rejects_non_processor(self):
        with pytest.raises(TypeError, match="HierarchicalMemoryProcessor"):
            SFPInterface("not a processor")

    def test_import_from_package(self):
        import sfp
        assert hasattr(sfp, "SFPInterface")
        assert sfp.SFPInterface is SFPInterface


class TestProperties:
    def test_d_model(self, iface):
        assert iface.d_model == D

    def test_step_count_starts_zero(self, iface):
        assert iface.step_count == 0

    def test_is_valence_enabled_false(self, iface):
        assert iface.is_valence_enabled is False

    def test_is_valence_enabled_true(self, iface_modules):
        assert iface_modules.is_valence_enabled is True

    def test_is_goals_enabled_false(self, iface):
        assert iface.is_goals_enabled is False

    def test_is_goals_enabled_true(self, iface_modules):
        assert iface_modules.is_goals_enabled is True

    def test_lock_is_rlock(self, iface):
        assert isinstance(iface.lock, type(threading.RLock()))


class TestProcess:
    def test_returns_surprise_metric(self, iface):
        x = torch.randn(D)
        result = iface.process(x)
        assert isinstance(result, SurpriseMetric)

    def test_increments_step_count(self, iface):
        x = torch.randn(D)
        iface.process(x)
        assert iface.step_count == 1

    def test_with_modality(self, iface):
        x = torch.randn(D)
        result = iface.process(x, modality="minecraft")
        assert isinstance(result, SurpriseMetric)


class TestQuery:
    def test_returns_reasoning_result(self, iface):
        x = torch.randn(D)
        result = iface.query(x)
        assert isinstance(result, ReasoningResult)
        assert result.knowledge.shape == (D,)

    def test_does_not_increment_step_count(self, iface):
        x = torch.randn(D)
        iface.query(x)
        assert iface.step_count == 0

    def test_with_trace(self, iface):
        x = torch.randn(D)
        result = iface.query(x, return_trace=True)
        assert isinstance(result, ReasoningResult)


class TestConsolidate:
    def test_consolidate_no_error(self, iface):
        # Should not raise even with empty memory
        iface.consolidate()

    def test_consolidate_with_force_mode(self, iface):
        iface.consolidate(force_mode=ConsolidationMode.MINI)


class TestInjectValence:
    def test_returns_none_when_disabled(self, iface):
        x = torch.randn(D)
        result = iface.inject_valence(x, reward=0.5)
        assert result is None

    def test_returns_signal_when_enabled(self, iface_modules):
        x = torch.randn(D)
        result = iface_modules.inject_valence(x, reward=0.5)
        assert isinstance(result, ValenceSignal)
        assert -1.0 <= result.scalar_valence <= 1.0

    def test_updates_processor_last_valence(self, iface_modules):
        x = torch.randn(D)
        signal = iface_modules.inject_valence(x, reward=0.8)
        assert iface_modules.processor._last_valence is signal

    def test_all_sources(self, iface_modules):
        x = torch.randn(D)
        signal = iface_modules.inject_valence(
            x,
            reward=0.3,
            user_feedback=-0.5,
            goal_alignment=0.7,
            prediction_satisfaction=0.2,
        )
        assert isinstance(signal, ValenceSignal)


class TestGoalManagement:
    def test_create_returns_none_when_disabled(self, iface):
        x = torch.randn(D)
        result = iface.create_goal(x)
        assert result is None

    def test_create_goal(self, iface_modules):
        x = torch.randn(D)
        goal = iface_modules.create_goal(x, importance=0.8, urgency=0.6)
        assert goal is not None
        assert goal.importance == 0.8
        assert goal.urgency == 0.6
        assert goal.status == GoalStatus.ACTIVE

    def test_remove_goal(self, iface_modules):
        x = torch.randn(D)
        goal = iface_modules.create_goal(x)
        assert iface_modules.remove_goal(goal.id) is True

    def test_remove_nonexistent_goal(self, iface_modules):
        assert iface_modules.remove_goal(9999) is False

    def test_remove_returns_false_when_disabled(self, iface):
        assert iface.remove_goal(0) is False

    def test_list_goals_empty(self, iface_modules):
        assert iface_modules.list_goals() == []

    def test_list_goals_returns_dicts(self, iface_modules):
        x = torch.randn(D)
        iface_modules.create_goal(x, importance=0.7)
        goals = iface_modules.list_goals()
        assert len(goals) == 1
        g = goals[0]
        assert "id" in g
        assert "status" in g
        assert "priority" in g
        assert "progress" in g
        assert "importance" in g
        assert g["importance"] == 0.7

    def test_list_goals_when_disabled(self, iface):
        assert iface.list_goals() == []


class TestStoreEpisode:
    def test_stores_episode(self, iface):
        x = torch.randn(D)
        result = iface.store_episode(x, modality="test", surprise=1.0)
        assert result is True
        assert iface.processor.tier1.total_count == 1

    def test_rejection_below_threshold(self):
        """Episode with surprise below threshold is rejected."""
        proc = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=D, n_layers=2),
            tier1_config=Tier1Config(
                hot_capacity=20, cold_capacity=40, surprise_threshold=0.5,
            ),
            tier2_config=Tier2Config(n_slots=16, d_value=D),
            tier3_config=Tier3Config(n_slots=8, d_value=D),
            device="cpu",
        )
        iface = SFPInterface(proc)
        result = iface.store_episode(
            torch.randn(D), surprise=0.01,
        )
        assert result is False

    def test_valence_annotation(self, iface):
        x = torch.randn(D)
        iface.store_episode(x, valence=0.7, surprise=1.0)
        ep = iface.processor.tier1._hot[0]
        assert ep.valence == 0.7

    def test_modality_tag(self, iface):
        x = torch.randn(D)
        iface.store_episode(x, modality="demo", surprise=1.0)
        ep = iface.processor.tier1._hot[0]
        assert ep.modality == "demo"

    def test_multiple_episodes(self, iface):
        for i in range(5):
            iface.store_episode(torch.randn(D), surprise=1.0)
        assert iface.processor.tier1.total_count == 5


class TestHealthAndStatus:
    def test_health_report(self, iface):
        report = iface.health_report()
        assert "step_count" in report
        assert "tier1" in report
        assert "tier2" in report

    def test_memory_footprint(self, iface):
        footprint = iface.memory_footprint()
        assert "total" in footprint
        assert footprint["total"] > 0

    def test_status_basic(self, iface):
        status = iface.status()
        assert status["step_count"] == 0
        assert status["tier1_episodes"] == 0
        assert status["tier2_basins"] == 0
        assert status["tier3_axioms"] == 0
        assert "active_goals" not in status  # goals disabled
        assert "mood" not in status  # valence disabled

    def test_status_with_modules(self, iface_modules):
        # Process once so valence signal exists
        iface_modules.inject_valence(torch.randn(D), reward=0.5)
        status = iface_modules.status()
        assert "active_goals" in status
        assert "mood" in status
        assert "valence" in status

    def test_status_after_processing(self, iface):
        iface.process(torch.randn(D))
        status = iface.status()
        assert status["step_count"] == 1


class TestProcessWithGoals:
    def test_process_with_active_goals(self, iface_modules):
        """Processing with active goals should not crash (regression test)."""
        iface_modules.create_goal(torch.randn(D), importance=0.8)
        result = iface_modules.process(torch.randn(D))
        assert isinstance(result, SurpriseMetric)

    def test_process_with_multiple_active_goals(self, iface_modules):
        """Processing with several active goals exercises reasoning bias path."""
        for _ in range(5):
            iface_modules.create_goal(torch.randn(D))
        for _ in range(3):
            result = iface_modules.process(torch.randn(D))
        assert isinstance(result, SurpriseMetric)
        assert iface_modules.step_count == 3


class TestSessionManagement:
    def test_reset_session(self, iface):
        iface.process(torch.randn(D))
        iface.reset_session()
        # Should not raise; tier0 still functional
        result = iface.process(torch.randn(D))
        assert isinstance(result, SurpriseMetric)


class TestThreadSafety:
    def test_concurrent_process_and_query(self, iface):
        """Multiple threads calling process/query concurrently should not crash."""
        errors = []
        n_iterations = 50

        def worker_process():
            try:
                for _ in range(n_iterations):
                    iface.process(torch.randn(D))
            except Exception as e:
                errors.append(e)

        def worker_query():
            try:
                for _ in range(n_iterations):
                    iface.query(torch.randn(D))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_process),
            threading.Thread(target=worker_query),
            threading.Thread(target=worker_process),
            threading.Thread(target=worker_query),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_mixed_operations(self, iface):
        """Mix of process, query, valence, status, episodes from multiple threads."""
        errors = []
        n_iterations = 30

        def worker_process():
            try:
                for _ in range(n_iterations):
                    iface.process(torch.randn(D))
            except Exception as e:
                errors.append(e)

        def worker_query():
            try:
                for _ in range(n_iterations):
                    iface.query(torch.randn(D))
            except Exception as e:
                errors.append(e)

        def worker_status():
            try:
                for _ in range(n_iterations):
                    iface.status()
                    iface.health_report()
            except Exception as e:
                errors.append(e)

        def worker_episodes():
            try:
                for _ in range(n_iterations):
                    iface.store_episode(torch.randn(D), surprise=1.0)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_process),
            threading.Thread(target=worker_query),
            threading.Thread(target=worker_status),
            threading.Thread(target=worker_episodes),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_valence_and_goals(self, iface_modules):
        """Valence injection and goal management from concurrent threads."""
        errors = []
        n_iterations = 30

        def worker_valence():
            try:
                for _ in range(n_iterations):
                    iface_modules.inject_valence(torch.randn(D), reward=0.1)
            except Exception as e:
                errors.append(e)

        def worker_goals():
            try:
                for _ in range(n_iterations):
                    g = iface_modules.create_goal(torch.randn(D))
                    if g is not None:
                        iface_modules.list_goals()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker_valence),
            threading.Thread(target=worker_goals),
            threading.Thread(target=worker_valence),
            threading.Thread(target=worker_goals),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=60)

        assert errors == [], f"Thread errors: {errors}"
