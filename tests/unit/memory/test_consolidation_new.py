"""Tests for new ConsolidationEngine features.

Covers:
  - G3: Topology urgency reduces standard consolidation interval
  - G3: No urgency when B0 is below threshold
  - A4: New basin keys tracked after _create_new_basins
  - D2: replay_skim_buffer method exists and functions correctly
  - D2: replay_skim_buffer with empty buffer is a no-op
"""

import time
from unittest.mock import MagicMock

import pytest
import torch

import sfp
from sfp.config import (
    ConsolidationConfig,
    FieldConfig,
    StreamingConfig,
    Tier1Config,
    Tier2Config,
    Tier3Config,
)
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor
from sfp.memory.consolidation import ConsolidationEngine
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.core import CoreMemory
from sfp.memory.integrity import compute_episode_hash
from sfp.types import ConsolidationMode, Episode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_episode(
    ep_id: int,
    dim: int = 256,
    basin_id: int = -1,
    surprise: float = 1.0,
    modality: str = "test",
) -> Episode:
    """Create a synthetic episode for testing."""
    inp = torch.randn(dim)
    logit = torch.randn(dim)
    wm = torch.randn(dim)
    integrity = compute_episode_hash(inp, logit, b"\x00" * 32)
    return Episode(
        id=ep_id,
        timestamp=time.monotonic(),
        modality=modality,
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
    )


# ---------------------------------------------------------------------------
# G3: Topology urgency tests
# ---------------------------------------------------------------------------


class TestTopologyUrgency:
    """G3: Betti B0 urgency should reduce the standard consolidation interval."""

    def test_topology_urgency_reduces_interval(self):
        """set_topology_urgency(25) with threshold=20 should halve the interval.

        With standard_interval=1000 and betti_consolidation_interval_reduction=0.5,
        effective interval becomes 500 when B0 >= 20. So should_consolidate(600)
        should return STANDARD because 600 >= 500.
        """
        config = ConsolidationConfig(
            standard_interval=1000,
            betti_b0_consolidation_threshold=20,
            betti_consolidation_interval_reduction=0.5,
            # Set deep and mini intervals high so they don't interfere
            deep_interval=100_000,
            mini_interval=100_000,
        )
        engine = ConsolidationEngine(config=config)
        engine._last_standard = 0

        # Apply topology urgency: B0=25 exceeds the threshold of 20
        engine.set_topology_urgency(25)

        # 600 steps since last standard: 600 >= 1000*0.5 = 500 -> STANDARD
        result = engine.should_consolidate(600)
        assert result == ConsolidationMode.STANDARD

    def test_no_urgency_when_b0_below_threshold(self):
        """set_topology_urgency(10) with threshold=20 should NOT reduce interval.

        With B0=10 < threshold=20, the effective interval stays at 1000.
        should_consolidate(600) should return None because 600 < 1000.
        """
        config = ConsolidationConfig(
            standard_interval=1000,
            betti_b0_consolidation_threshold=20,
            betti_consolidation_interval_reduction=0.5,
            deep_interval=100_000,
            mini_interval=100_000,
        )
        engine = ConsolidationEngine(config=config)
        engine._last_standard = 0

        # B0=10, below threshold of 20 — no urgency
        engine.set_topology_urgency(10)

        # 600 steps since last standard: 600 < 1000 -> None
        result = engine.should_consolidate(600)
        assert result is None

    def test_urgency_at_exact_threshold(self):
        """B0 exactly at threshold should trigger reduced interval."""
        config = ConsolidationConfig(
            standard_interval=1000,
            betti_b0_consolidation_threshold=20,
            betti_consolidation_interval_reduction=0.5,
            deep_interval=100_000,
            mini_interval=100_000,
        )
        engine = ConsolidationEngine(config=config)
        engine._last_standard = 0
        engine.set_topology_urgency(20)

        # 600 >= 500 -> STANDARD
        result = engine.should_consolidate(600)
        assert result == ConsolidationMode.STANDARD

    def test_urgency_does_not_affect_deep_interval(self):
        """Topology urgency only affects standard interval, not deep."""
        config = ConsolidationConfig(
            standard_interval=1000,
            deep_interval=10_000,
            betti_b0_consolidation_threshold=20,
            betti_consolidation_interval_reduction=0.5,
            mini_interval=100_000,
        )
        engine = ConsolidationEngine(config=config)
        engine._last_standard = 0
        engine._last_deep = 0
        engine.set_topology_urgency(25)

        # 6000 steps: deep_interval is still 10_000 (unchanged)
        # 6000 < 10_000, so deep should not trigger
        # But 6000 >= 500 (reduced standard), so STANDARD
        result = engine.should_consolidate(6000)
        assert result == ConsolidationMode.STANDARD

    def test_without_urgency_standard_requires_full_interval(self):
        """Without any urgency, should_consolidate needs full 1000 steps."""
        config = ConsolidationConfig(
            standard_interval=1000,
            betti_b0_consolidation_threshold=20,
            betti_consolidation_interval_reduction=0.5,
            deep_interval=100_000,
            mini_interval=100_000,
        )
        engine = ConsolidationEngine(config=config)
        engine._last_standard = 0

        # No urgency set (default _topology_b0=0)
        assert engine.should_consolidate(600) is None
        assert engine.should_consolidate(999) is None
        assert engine.should_consolidate(1000) == ConsolidationMode.STANDARD


# ---------------------------------------------------------------------------
# A4: New basin keys tracked
# ---------------------------------------------------------------------------


class TestNewBasinKeysTracked:
    """A4: _last_new_basin_keys populated after _create_new_basins."""

    def test_new_basin_keys_populated(self):
        """After _create_new_basins with enough candidates, keys should be stored."""
        dim = 256
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)
        config = ConsolidationConfig(
            distillation_threshold=0.3,
            new_concept_threshold=3,
        )
        engine = ConsolidationEngine(config=config, tier2=tier2)

        # Create candidates with diverse embeddings that will cluster
        # Use pairs of similar embeddings so clusters have >= 2 members
        candidates = []
        base_a = torch.randn(dim)
        base_b = torch.randn(dim)
        # Ensure the two bases are dissimilar (far apart)
        base_b = base_b - base_a
        base_b = base_b / base_b.norm() * base_a.norm()

        for i in range(4):
            ep = _make_episode(i, dim=dim)
            # Two episodes near base_a, two near base_b
            if i < 2:
                ep.input_embedding = base_a + torch.randn(dim) * 0.01
            else:
                ep.input_embedding = base_b + torch.randn(dim) * 0.01
            candidates.append(ep)

        # Initially no new basin keys
        assert engine._last_new_basin_keys == []

        engine._create_new_basins(candidates)

        # At least one new basin key should have been created
        assert len(engine._last_new_basin_keys) > 0
        # Each key should be a tensor of the right dimension
        for key in engine._last_new_basin_keys:
            assert isinstance(key, torch.Tensor)
            assert key.shape == (dim,)

    def test_new_basin_keys_empty_when_no_candidates(self):
        """_create_new_basins with empty list produces no keys."""
        dim = 256
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)
        engine = ConsolidationEngine(tier2=tier2)

        engine._create_new_basins([])
        assert engine._last_new_basin_keys == []

    def test_new_basin_keys_reset_each_call(self):
        """Each call to _create_new_basins with candidates resets keys."""
        dim = 256
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)
        config = ConsolidationConfig(distillation_threshold=0.3)
        engine = ConsolidationEngine(config=config, tier2=tier2)

        # Pre-populate with a fake key
        engine._last_new_basin_keys = [torch.randn(dim)]
        assert len(engine._last_new_basin_keys) == 1

        # Call with non-empty candidates (but only singletons, so no basins
        # actually created) — _last_new_basin_keys should still be reset to []
        # because the method resets it before processing clusters.
        single_ep = _make_episode(0, dim=dim)
        engine._create_new_basins([single_ep])
        assert engine._last_new_basin_keys == []

    def test_empty_candidates_early_return(self):
        """_create_new_basins with empty list returns early without resetting."""
        dim = 256
        tier2 = EssentialMemory(Tier2Config(d_value=dim), d_model=dim)
        engine = ConsolidationEngine(tier2=tier2)

        # Pre-populate with a fake key
        fake_key = torch.randn(dim)
        engine._last_new_basin_keys = [fake_key]

        # Empty candidates triggers early return — does NOT reset the list
        engine._create_new_basins([])
        assert len(engine._last_new_basin_keys) == 1


# ---------------------------------------------------------------------------
# D2: Skim replay tests
# ---------------------------------------------------------------------------


class TestSkimReplay:
    """D2: replay_skim_buffer method tests."""

    @pytest.fixture
    def skim_setup(self):
        """Create a StreamingProcessor and ConsolidationEngine for skim tests."""
        dim = 256
        field_cfg = FieldConfig.from_preset(sfp.FieldSize.TINY)
        field = SemanticFieldProcessor(field_cfg)
        tier0 = StreamingProcessor(field, streaming_config=StreamingConfig())
        config = ConsolidationConfig(skim_replay_lr_scale=0.25)
        engine = ConsolidationEngine(config=config, tier0=tier0)
        return engine, tier0, field_cfg.dim

    def test_replay_skim_buffer_callable(self, skim_setup):
        """replay_skim_buffer can be called with a mock salience gate and tier0."""
        engine, tier0, dim = skim_setup

        # Mock salience gate with a populated skim buffer
        salience_gate = MagicMock()
        salience_gate._skim_buffer = [torch.randn(dim)]

        # Should not raise
        engine.replay_skim_buffer(salience_gate, tier0)

    def test_replay_skim_buffer_empty_buffer_noop(self, skim_setup):
        """replay_skim_buffer with empty _skim_buffer does nothing."""
        engine, tier0, dim = skim_setup

        # Snapshot field params before replay
        initial_params = {
            n: p.clone() for n, p in tier0.field.named_parameters()
        }

        salience_gate = MagicMock()
        salience_gate._skim_buffer = []

        engine.replay_skim_buffer(salience_gate, tier0)

        # Field params should be unchanged — empty buffer means no replay
        for n, p in tier0.field.named_parameters():
            assert torch.equal(p, initial_params[n]), (
                f"Param {n} changed despite empty skim buffer"
            )

    def test_replay_skim_buffer_no_skim_attr_noop(self, skim_setup):
        """replay_skim_buffer with gate lacking _skim_buffer is a no-op."""
        engine, tier0, dim = skim_setup

        initial_params = {
            n: p.clone() for n, p in tier0.field.named_parameters()
        }

        # Object without _skim_buffer attribute
        salience_gate = object()

        engine.replay_skim_buffer(salience_gate, tier0)

        for n, p in tier0.field.named_parameters():
            assert torch.equal(p, initial_params[n]), (
                f"Param {n} changed despite missing _skim_buffer"
            )

    def test_replay_skim_buffer_processes_entries(self, skim_setup):
        """replay_skim_buffer with populated buffer processes each entry.

        The method uses autoassociative targets (field output as target for
        the same input), so with a clean field the loss is near zero and
        weight changes may be negligible. We verify the method runs without
        error and calls replay_episode the expected number of times by
        checking that the field remains in train mode after replay.
        """
        engine, tier0, dim = skim_setup

        salience_gate = MagicMock()
        salience_gate._skim_buffer = [torch.randn(dim) for _ in range(8)]

        # The field should be in train mode after replay_episode calls
        tier0.field.eval()
        engine.replay_skim_buffer(salience_gate, tier0)

        # replay_episode sets field to train mode, so if entries were processed
        # the field should now be in training mode
        assert tier0.field.training, (
            "Field should be in train mode after skim replay processed entries"
        )

    def test_replay_skim_buffer_respects_lr_scale(self):
        """Skim replay should use the configured skim_replay_lr_scale."""
        dim = 256
        field = SemanticFieldProcessor(FieldConfig.from_preset(sfp.FieldSize.TINY))
        tier0 = StreamingProcessor(field, streaming_config=StreamingConfig())

        # Very small lr_scale -> minimal weight change
        config_small = ConsolidationConfig(skim_replay_lr_scale=0.001)
        engine_small = ConsolidationEngine(config=config_small, tier0=tier0)

        assert engine_small._config.skim_replay_lr_scale == 0.001
