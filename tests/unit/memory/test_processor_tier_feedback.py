"""Tests for memory tier feedback features in HierarchicalMemoryProcessor.

Verifies that Tier 2 guidance, Tier 3 axiom anchors, chain valence bias,
goal-stall forced consolidation, and combined valence + urgency LR scaling
are wired correctly through the processor pipeline.

These are primarily smoke tests (no-crash verification); detailed behaviour
is validated in the individual module test suites.
"""

from __future__ import annotations

from unittest.mock import patch

import torch
import pytest

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
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import SurpriseMetric

D = 32  # tiny dimension for fast tests


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _field_config() -> FieldConfig:
    return FieldConfig(dim=D, n_layers=2)


def _tier1_config() -> Tier1Config:
    return Tier1Config(
        hot_capacity=20,
        cold_capacity=40,
        surprise_threshold=0.0,
    )


def _tier2_config() -> Tier2Config:
    return Tier2Config(n_slots=16, d_value=D)


def _tier3_config() -> Tier3Config:
    return Tier3Config(n_slots=8, d_value=D)


def _streaming_config() -> StreamingConfig:
    return StreamingConfig(surprise_threshold=0.0)


def _consolidation_config(**overrides) -> ConsolidationConfig:
    defaults = dict(
        mini_interval=5,
        standard_interval=10,
        deep_interval=100,
        goal_stall_consolidation_steps=5,
    )
    defaults.update(overrides)
    return ConsolidationConfig(**defaults)


def _goal_config(**overrides) -> GoalPersistenceConfig:
    defaults = dict(
        max_goals=4,
        d_goal=D,
        d_satisfaction=D,
        stall_steps=3,
        stall_threshold=0.001,
    )
    defaults.update(overrides)
    return GoalPersistenceConfig(**defaults)


def _valence_config() -> ValenceConfig:
    return ValenceConfig(d_valence_embedding=16)


@pytest.fixture
def base_processor() -> HierarchicalMemoryProcessor:
    """Minimal processor with no cognitive modules."""
    return HierarchicalMemoryProcessor(
        field_config=_field_config(),
        tier1_config=_tier1_config(),
        tier2_config=_tier2_config(),
        tier3_config=_tier3_config(),
        streaming_config=_streaming_config(),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_with_valence() -> HierarchicalMemoryProcessor:
    """Processor with valence system enabled."""
    return HierarchicalMemoryProcessor(
        field_config=_field_config(),
        tier1_config=_tier1_config(),
        tier2_config=_tier2_config(),
        tier3_config=_tier3_config(),
        streaming_config=_streaming_config(),
        valence_config=_valence_config(),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_with_goals() -> HierarchicalMemoryProcessor:
    """Processor with goals enabled and tight consolidation schedule."""
    return HierarchicalMemoryProcessor(
        field_config=_field_config(),
        tier1_config=_tier1_config(),
        tier2_config=_tier2_config(),
        tier3_config=_tier3_config(),
        streaming_config=_streaming_config(),
        consolidation_config=_consolidation_config(),
        goal_config=_goal_config(),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_with_valence_and_goals() -> HierarchicalMemoryProcessor:
    """Processor with both valence and goals enabled."""
    return HierarchicalMemoryProcessor(
        field_config=_field_config(),
        tier1_config=_tier1_config(),
        tier2_config=_tier2_config(),
        tier3_config=_tier3_config(),
        streaming_config=_streaming_config(),
        consolidation_config=_consolidation_config(),
        goal_config=_goal_config(),
        valence_config=_valence_config(),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_tier2_basin(proc: HierarchicalMemoryProcessor) -> None:
    """Manually activate one basin in Tier 2 so reasoning can visit it."""
    tier2 = proc._tier2
    with torch.no_grad():
        tier2.keys[0] = torch.randn(D)
        tier2.values[0] = torch.randn(D)
        tier2.confidence[0] = 0.9
        tier2.active_mask[0] = True
        tier2._n_active = 1


def _seed_tier3_axiom(proc: HierarchicalMemoryProcessor) -> None:
    """Manually activate one axiom in Tier 3 so retrieve returns non-zero."""
    tier3 = proc._tier3
    with torch.no_grad():
        tier3.keys.data[0] = torch.randn(D)
        tier3.values.data[0] = torch.randn(D)
        tier3.confidence[0] = 1.0
        tier3.active_mask[0] = True
        tier3._n_active = 1


# ---------------------------------------------------------------------------
# Test 1: Tier 2 guidance extracted
# ---------------------------------------------------------------------------


class TestTier2GuidanceExtracted:
    """After process(), if reasoning visits a basin, tier2_guidance is passed
    to the StreamingProcessor.  We verify indirectly by checking that a
    processor with an active basin completes process() without error."""

    def test_process_with_active_basin_completes(self, base_processor):
        """Process should complete when Tier 2 has an active basin that
        reasoning can visit, triggering tier2_guidance extraction."""
        _seed_tier2_basin(base_processor)
        x = torch.randn(D)
        result = base_processor.process(x, modality="text")
        assert isinstance(result, SurpriseMetric)

    def test_multiple_process_calls_with_basin(self, base_processor):
        """Repeated process() calls with an active basin should not accumulate
        state corruption from tier2_guidance being passed each time."""
        _seed_tier2_basin(base_processor)
        for _ in range(5):
            x = torch.randn(D)
            result = base_processor.process(x, modality="text")
            assert isinstance(result, SurpriseMetric)


# ---------------------------------------------------------------------------
# Test 2: Axiom anchor extracted
# ---------------------------------------------------------------------------


class TestAxiomAnchorExtracted:
    """When Tier 3 has stored knowledge, axiom_anchor flows to field learning.
    We verify by manually seeding Tier 3 and confirming process() completes."""

    def test_process_with_tier3_knowledge_completes(self, base_processor):
        """Process should complete when Tier 3 has a stored axiom, causing
        axiom_anchor to be passed to Tier 0's process call."""
        _seed_tier3_axiom(base_processor)
        x = torch.randn(D)
        result = base_processor.process(x, modality="text")
        assert isinstance(result, SurpriseMetric)

    def test_process_with_both_tiers_populated(self, base_processor):
        """Both tier2_guidance AND axiom_anchor should flow together without
        conflict when both tiers have active knowledge."""
        _seed_tier2_basin(base_processor)
        _seed_tier3_axiom(base_processor)
        x = torch.randn(D)
        result = base_processor.process(x, modality="text")
        assert isinstance(result, SurpriseMetric)


# ---------------------------------------------------------------------------
# Test 3: Chain valence bias initialized
# ---------------------------------------------------------------------------


class TestChainValenceBiasInitialized:
    """After constructing with valence enabled, _last_chain_valence_bias
    should be an empty dict."""

    def test_initial_chain_valence_bias_is_empty_dict(self, processor_with_valence):
        """The bias dict should start empty so that the first process() step
        does not inject stale bias into the reasoning router."""
        assert processor_with_valence._last_chain_valence_bias == {}
        assert isinstance(processor_with_valence._last_chain_valence_bias, dict)

    def test_chain_valence_bias_is_empty_without_valence(self, base_processor):
        """Even without valence enabled, the bias dict should exist and be
        empty (it is always initialized)."""
        assert base_processor._last_chain_valence_bias == {}

    def test_chain_valence_bias_after_process(self, processor_with_valence):
        """After one process() call the bias dict may still be empty (no
        negative chain valence computed) — the key point is it does not
        raise and remains a dict."""
        x = torch.randn(D)
        processor_with_valence.process(x, modality="text")
        assert isinstance(processor_with_valence._last_chain_valence_bias, dict)


# ---------------------------------------------------------------------------
# Test 4: Goal stall forced consolidation
# ---------------------------------------------------------------------------


class TestGoalStallForcedConsolidation:
    """With goals module and a stalled goal (progress history flat for
    goal_stall_consolidation_steps), step 10 triggers forced standard
    consolidation."""

    def test_stalled_goal_triggers_forced_consolidation(self, processor_with_goals):
        """A goal that is stalled long enough should cause consolidation to
        fire even outside the normal consolidation schedule."""
        proc = processor_with_goals

        # Create a goal (embedding does not matter for stall detection)
        goal_emb = torch.randn(D)
        goal = proc._goals.create_goal(goal_emb, importance=0.8, urgency=0.5)

        # Manually inject a long progress history so the processor's
        # len(progress_history) >= goal_stall_consolidation_steps (5) check passes.
        goal.progress_history = [0.1] * 10

        # Record the consolidation engine's last-standard step before
        consolidation_before = proc._consolidation._last_standard

        # Set step_count so we are NOT on a scheduled consolidation boundary
        # (standard_interval=10, so step 3 is safe).
        proc._step_count = 2

        # Patch detect_stalled_goals to return our goal ID — step 4.6's
        # update_progress appends a non-flat value to the history which
        # would make the real stall check fail.  The stall detection itself
        # is tested in the goals module; here we test the processor's
        # reaction to a stalled goal.
        with patch.object(
            proc._goals, "detect_stalled_goals", return_value=[goal.id],
        ):
            x = torch.randn(D)
            result = proc.process(x, modality="text")
            assert isinstance(result, SurpriseMetric)

        # After process(), consolidation should have run because the goal
        # was stalled — _last_standard should have been updated.
        assert proc._consolidation._last_standard > consolidation_before

    def test_no_forced_consolidation_when_history_too_short(self, processor_with_goals):
        """If the stalled goal's progress_history is shorter than
        goal_stall_consolidation_steps, no forced consolidation occurs."""
        proc = processor_with_goals

        goal_emb = torch.randn(D)
        goal = proc._goals.create_goal(goal_emb, importance=0.8, urgency=0.5)

        # History shorter than goal_stall_consolidation_steps (5).
        # Step 4.6 will append one value (making len=4), still < 5.
        goal.progress_history = [0.1] * 3

        consolidation_before = proc._consolidation._last_standard
        proc._step_count = 2

        # Patch detect_stalled_goals to return the goal ID (simulate stall
        # detection succeeding) — the processor should still skip forced
        # consolidation because len(progress_history) < goal_stall_consolidation_steps.
        with patch.object(
            proc._goals, "detect_stalled_goals", return_value=[goal.id],
        ):
            x = torch.randn(D)
            proc.process(x, modality="text")

        # Forced consolidation should NOT have triggered (history too short)
        assert proc._consolidation._last_standard == consolidation_before

    def test_stalled_goal_with_active_basin(self, processor_with_goals):
        """Forced consolidation from goal stall works even when Tier 2 has
        active basins (the consolidation should not crash on populated state)."""
        proc = processor_with_goals
        _seed_tier2_basin(proc)

        goal_emb = torch.randn(D)
        goal = proc._goals.create_goal(goal_emb, importance=0.8, urgency=0.5)
        goal.progress_history = [0.1] * 10

        proc._step_count = 2
        x = torch.randn(D)
        result = proc.process(x, modality="text")
        assert isinstance(result, SurpriseMetric)


# ---------------------------------------------------------------------------
# Test 5: External LR scale combines valence + urgency
# ---------------------------------------------------------------------------


class TestExternalLRScaleCombinesValenceAndUrgency:
    """Verify the processor creates and completes process() with both valence
    and goals enabled, exercising the combined_lr_scale code path."""

    def test_process_with_valence_and_goals_completes(
        self, processor_with_valence_and_goals,
    ):
        """A single process() with both modules enabled should not crash."""
        x = torch.randn(D)
        result = processor_with_valence_and_goals.process(x, modality="text")
        assert isinstance(result, SurpriseMetric)

    def test_multiple_steps_with_valence_and_goals(
        self, processor_with_valence_and_goals,
    ):
        """Several process() calls should not accumulate errors from the
        combined LR scaling path (valence_lr_scale * goal_urgency_lr_scale)."""
        proc = processor_with_valence_and_goals

        # Create a goal so the urgency path activates
        goal_emb = torch.randn(D)
        proc._goals.create_goal(goal_emb, importance=0.9, urgency=0.8)

        for _ in range(8):
            x = torch.randn(D)
            result = proc.process(x, modality="text")
            assert isinstance(result, SurpriseMetric)

    def test_valence_and_goals_with_seeded_tiers(
        self, processor_with_valence_and_goals,
    ):
        """Combined LR scaling should work alongside tier2_guidance and
        axiom_anchor without interference."""
        proc = processor_with_valence_and_goals
        _seed_tier2_basin(proc)
        _seed_tier3_axiom(proc)

        goal_emb = torch.randn(D)
        proc._goals.create_goal(goal_emb, importance=0.9, urgency=0.8)

        for _ in range(5):
            x = torch.randn(D)
            result = proc.process(x, modality="text")
            assert isinstance(result, SurpriseMetric)
