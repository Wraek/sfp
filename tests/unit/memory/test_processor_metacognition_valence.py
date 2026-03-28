"""Tests for metacognition and valence signal wiring in HierarchicalMemoryProcessor.

Verifies that the new cognitive module integrations (info-seeking goal creation,
vigilance surprise modulation, valence LR modulation, uncertainty/mood history
accumulation) work correctly when wired through the processor pipeline.
"""

import torch
import pytest

from sfp.config import (
    BackboneConfig,
    ConsolidationConfig,
    FieldConfig,
    GoalPersistenceConfig,
    MetacognitionConfig,
    PerceiverConfig,
    SelectiveAttentionConfig,
    StreamingConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
    ValenceConfig,
    WorldModelConfig,
)
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import SurpriseMetric

D = 32  # small dimension for fast tests


def _streaming_cfg() -> StreamingConfig:
    """Streaming config with warmup disabled and zero surprise threshold."""
    return StreamingConfig(
        warmup_steps=0,
        adaptive_surprise=False,
        surprise_threshold=0.0,
    )


def _tier1_cfg() -> Tier1Config:
    return Tier1Config(
        hot_capacity=20,
        cold_capacity=40,
        surprise_threshold=0.0,
    )


def _tier2_cfg() -> Tier2Config:
    return Tier2Config(n_slots=16, d_value=D)


def _tier3_cfg() -> Tier3Config:
    return Tier3Config(n_slots=8, d_value=D)


def _consolidation_cfg() -> ConsolidationConfig:
    """Push consolidation intervals far out so they don't fire during tests."""
    return ConsolidationConfig(
        mini_interval=100_000,
        standard_interval=1_000_000,
        deep_interval=10_000_000,
    )


# ---- fixtures ----


@pytest.fixture
def processor_metacognition_goals():
    """Processor with metacognition + goals enabled (metacognition_goal_generation=True)."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=_tier1_cfg(),
        tier2_config=_tier2_cfg(),
        tier3_config=_tier3_cfg(),
        streaming_config=_streaming_cfg(),
        consolidation_config=_consolidation_cfg(),
        metacognition_config=MetacognitionConfig(
            d_uncertainty_embedding=16,
            estimator_hidden=16,
            metacognition_goal_generation=True,
        ),
        goal_config=GoalPersistenceConfig(
            max_goals=32,
            d_goal=D,
            d_satisfaction=D,
        ),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_valence():
    """Processor with valence enabled (vigilance_surprise_modulation=True)."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=_tier1_cfg(),
        tier2_config=_tier2_cfg(),
        tier3_config=_tier3_cfg(),
        streaming_config=_streaming_cfg(),
        consolidation_config=_consolidation_cfg(),
        valence_config=ValenceConfig(
            d_valence_embedding=16,
            vigilance_surprise_modulation=True,
            valence_lr_modulation=True,
            valence_lr_scale_range=(0.5, 1.5),
        ),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


@pytest.fixture
def processor_all_modules():
    """Processor with ALL cognitive modules enabled."""
    return HierarchicalMemoryProcessor(
        field_config=FieldConfig(dim=D, n_layers=2),
        tier1_config=_tier1_cfg(),
        tier2_config=_tier2_cfg(),
        tier3_config=_tier3_cfg(),
        streaming_config=_streaming_cfg(),
        consolidation_config=_consolidation_cfg(),
        world_model_config=WorldModelConfig(
            d_deterministic=D,
            d_stochastic_categories=4,
            d_stochastic_classes=4,
            d_observation=D,
        ),
        goal_config=GoalPersistenceConfig(
            max_goals=32,
            d_goal=D,
            d_satisfaction=D,
        ),
        metacognition_config=MetacognitionConfig(
            d_uncertainty_embedding=16,
            estimator_hidden=16,
            metacognition_goal_generation=True,
        ),
        valence_config=ValenceConfig(
            d_valence_embedding=16,
            vigilance_surprise_modulation=True,
            valence_lr_modulation=True,
        ),
        attention_config=SelectiveAttentionConfig(
            n_modalities=2,
            modality_names=("text", "sensor"),
            d_salience=16,
            d_context=16,
        ),
        lora_config=None,
        ewc_config=None,
        device="cpu",
    )


# ---- tests ----


class TestInfoSeekingGoalCreation:
    """F1: Metacognition info-seeking creates exploratory goals."""

    def test_goals_accumulate_after_processing(self, processor_metacognition_goals):
        """After several process() calls, info-seeking should create goals.

        With metacognition + goals enabled and metacognition_goal_generation=True,
        the uncertainty estimation triggers suggest_information_seeking which
        auto-creates exploratory goals via create_goal.
        """
        proc = processor_metacognition_goals
        assert len(proc.goals._goals) == 0, "No goals initially"

        for _ in range(5):
            x = torch.randn(D)
            proc.process(x, modality="text")

        assert len(proc.goals._goals) > 0, (
            "Info-seeking should have created at least one exploratory goal "
            "after 5 process() calls with metacognition_goal_generation=True"
        )


class TestVigilanceSurpriseModulation:
    """F3: Vigilance modulates surprise_boost in the processing pipeline."""

    def test_process_completes_with_vigilance(self, processor_valence):
        """Smoke test: process() runs without error when vigilance_surprise_modulation=True.

        The valence system's vigilance field modulates the surprise_boost in
        step 7.2 of the pipeline. After the first step _last_valence is populated,
        subsequent steps exercise the vigilance path.
        """
        proc = processor_valence
        for _ in range(3):
            x = torch.randn(D)
            result = proc.process(x, modality="text")
            assert isinstance(result, SurpriseMetric)


class TestValenceLRModulationConfig:
    """F2: ValenceConfig defaults for LR modulation."""

    def test_valence_lr_modulation_defaults(self):
        """ValenceConfig should have valence_lr_modulation=True and range (0.5, 1.5)."""
        cfg = ValenceConfig()
        assert cfg.valence_lr_modulation is True
        assert cfg.valence_lr_scale_range == (0.5, 1.5)

    def test_valence_lr_modulation_custom_range(self):
        """Custom LR scale ranges should be preserved."""
        cfg = ValenceConfig(valence_lr_scale_range=(0.3, 2.0))
        assert cfg.valence_lr_scale_range == (0.3, 2.0)


class TestAllCognitiveModules:
    """Integration smoke test with every cognitive module enabled."""

    def test_full_pipeline_10_steps(self, processor_all_modules):
        """Process 10 steps with all modules (world model, goals, metacognition,
        valence, salience gate) enabled. Verifies no crashes or assertion errors.
        """
        proc = processor_all_modules
        for step in range(10):
            x = torch.randn(D)
            result = proc.process(x, modality="text")
            assert isinstance(result, SurpriseMetric), (
                f"Step {step}: expected SurpriseMetric, got {type(result)}"
            )

        # Basic sanity: step counter advanced
        assert proc._step_count == 10


class TestUncertaintyHistoryAccumulates:
    """C: _uncertainty_history grows as metacognition runs."""

    def test_uncertainty_history_grows(self, processor_metacognition_goals):
        """After processing with metacognition enabled, _uncertainty_history
        should accumulate one entry per step (uncertainty = 1 - confidence).
        """
        proc = processor_metacognition_goals
        assert len(proc._uncertainty_history) == 0

        n_steps = 4
        for _ in range(n_steps):
            x = torch.randn(D)
            proc.process(x, modality="text")

        assert len(proc._uncertainty_history) == n_steps, (
            f"Expected {n_steps} uncertainty entries, got {len(proc._uncertainty_history)}"
        )
        # Each entry should be a float in [0, 1]
        for val in proc._uncertainty_history:
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0, f"Uncertainty {val} out of [0, 1]"


class TestMoodHistoryAccumulates:
    """C: _mood_history grows as valence runs."""

    def test_mood_history_grows(self, processor_valence):
        """After processing with valence enabled, _mood_history should grow.

        The mood_history is appended once per step *after* _last_valence is
        populated, so the first step creates _last_valence but doesn't yet
        append to mood_history (it checks _last_valence which is None on the
        first step). From the second step onward, each step appends one entry.
        """
        proc = processor_valence
        assert len(proc._mood_history) == 0

        n_steps = 5
        for _ in range(n_steps):
            x = torch.randn(D)
            proc.process(x, modality="text")

        # After step 1: _last_valence gets populated but mood_history not appended
        # (because _last_valence was None when the append check ran).
        # Steps 2..n_steps each append one entry => n_steps - 1 entries.
        assert len(proc._mood_history) == n_steps - 1, (
            f"Expected {n_steps - 1} mood entries (first step populates "
            f"_last_valence), got {len(proc._mood_history)}"
        )
        for val in proc._mood_history:
            assert isinstance(val, float)
