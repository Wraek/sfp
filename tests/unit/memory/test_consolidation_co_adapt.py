"""Tests for co-adaptation extensions to ConsolidationEngine.

Covers:
  - Field replay during standard consolidation (Change 1)
  - Valence-weighted episode sampling (Change 4)
  - set_valence_system late binding
"""

import time

import torch
import pytest

import sfp
from sfp.config import (
    ConsolidationConfig,
    FieldConfig,
    StreamingConfig,
    Tier1Config,
    Tier2Config,
)
from sfp.core.streaming import StreamingProcessor
from sfp.memory.consolidation import ConsolidationEngine
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.integrity import compute_episode_hash, compute_weight_hash
from sfp.types import Episode


def _make_episode(
    ep_id: int,
    dim: int = 256,
    basin_id: int = 0,
    surprise: float = 1.0,
    valence: float = 0.0,
) -> Episode:
    """Create a synthetic episode for testing."""
    inp = torch.randn(dim)
    logit = torch.randn(dim)
    wm = torch.randn(dim)
    integrity = compute_episode_hash(inp, logit, b"\x00" * 32)
    return Episode(
        id=ep_id,
        timestamp=time.monotonic(),
        modality="test",
        provenance_hash=b"\x00" * 16,
        input_embedding=inp,
        working_memory_state=wm,
        logit_snapshot=logit,
        surprise_at_storage=surprise,
        attractor_basin_id=basin_id,
        attractor_distance=0.5,
        preceding_episode_id=None,
        following_episode_id=None,
        integrity_hash=integrity,
        weight_hash_at_storage=b"\x00" * 32,
        valence=valence,
    )


@pytest.fixture
def replay_engine():
    """ConsolidationEngine configured for field replay."""
    dim = 256
    field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
    tier0 = StreamingProcessor(field, streaming_config=StreamingConfig())
    tier1 = EpisodicMemory(
        Tier1Config(hot_capacity=100, cold_capacity=100, surprise_threshold=0.0),
        d_model=dim,
    )
    tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)

    config = ConsolidationConfig(
        replay_through_field_enabled=True,
        replay_through_field_batch_size=4,
        replay_lr_scale=0.5,
        replay_batch_size=8,
        standard_interval=10,
        mini_interval=5,
    )
    engine = ConsolidationEngine(
        config=config,
        tier0=tier0,
        tier1=tier1,
        tier2=tier2,
    )

    # Populate Tier 1 with episodes
    for i in range(20):
        ep = _make_episode(i, dim=dim, basin_id=i % 3)
        tier1.maybe_store(ep)

    return engine, tier0, tier1, tier2


class TestConsolidationReplay:
    """Field replay during standard consolidation (Change 1)."""

    def test_standard_consolidate_with_replay(self, replay_engine):
        engine, tier0, tier1, tier2 = replay_engine
        # Run standard consolidation (includes replay phase)
        engine.standard_consolidate(step_count=100)
        # Should not crash and tier0 should have been trained
        assert len(tier0.surprise_history) == 0  # replay doesn't add to history

    def test_replay_explicitly_disabled(self):
        dim = 256
        field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        tier0 = StreamingProcessor(field, streaming_config=StreamingConfig())
        tier1 = EpisodicMemory(
            Tier1Config(hot_capacity=100, cold_capacity=100, surprise_threshold=0.0),
            d_model=dim,
        )
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)

        # Explicitly disable replay
        engine = ConsolidationEngine(
            config=ConsolidationConfig(replay_through_field_enabled=False),
            tier0=tier0,
            tier1=tier1,
            tier2=tier2,
        )

        # Populate Tier 1
        for i in range(10):
            ep = _make_episode(i, dim=dim)
            tier1.maybe_store(ep)

        # Get initial field state
        initial_params = {
            n: p.clone() for n, p in tier0.field.named_parameters()
        }

        engine.standard_consolidate(step_count=100)

        # Field params should be unchanged (no replay)
        for n, p in tier0.field.named_parameters():
            assert torch.allclose(p, initial_params[n]), f"Param {n} changed despite replay being disabled"

    def test_replay_modifies_field_weights(self, replay_engine):
        engine, tier0, tier1, tier2 = replay_engine
        # Snapshot initial params
        initial_params = {
            n: p.clone() for n, p in tier0.field.named_parameters()
        }
        engine.standard_consolidate(step_count=100)
        # At least some params should have changed
        changed = False
        for n, p in tier0.field.named_parameters():
            if not torch.allclose(p, initial_params[n]):
                changed = True
                break
        assert changed, "Field params should change after replay"

    def test_replay_with_empty_tier1(self):
        dim = 256
        field = sfp.SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        tier0 = StreamingProcessor(field, streaming_config=StreamingConfig())
        tier1 = EpisodicMemory(
            Tier1Config(hot_capacity=100, cold_capacity=100),
            d_model=dim,
        )
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)

        engine = ConsolidationEngine(
            config=ConsolidationConfig(replay_through_field_enabled=True),
            tier0=tier0,
            tier1=tier1,
            tier2=tier2,
        )
        # Should not crash with empty Tier 1
        engine.standard_consolidate(step_count=100)


class TestValenceWeightedSampling:
    """Valence-weighted consolidation sampling (Change 4)."""

    def test_set_valence_system(self):
        engine = ConsolidationEngine()
        assert engine._valence is None
        # Can set to a mock
        engine.set_valence_system("mock_valence")
        assert engine._valence == "mock_valence"
        # Can clear
        engine.set_valence_system(None)
        assert engine._valence is None

    def test_valence_weighted_sampling_enabled_by_default(self):
        """Default config uses valence weighting."""
        config = ConsolidationConfig()
        assert config.valence_weighted_sampling is True

    def test_uniform_fallback_without_valence(self, replay_engine):
        engine, tier0, tier1, tier2 = replay_engine
        # Enable valence sampling but don't set valence system
        engine._config = ConsolidationConfig(
            valence_weighted_sampling=True,
            replay_batch_size=8,
            standard_interval=10,
        )
        assert engine._valence is None
        # Should fall back to uniform sampling, not crash
        engine.standard_consolidate(step_count=100)
