"""Tests for world model continue_head and spatial_predictor consumption.

Covers:
  - continue_head → episode boundary detection + force-store
  - spatial_predictor → spatial prediction error as surprise boost
  - force_store in EpisodicMemory (bypasses surprise threshold)
"""

from __future__ import annotations

import hashlib

import pytest
import torch

from sfp.config import Tier1Config, WorldModelConfig
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.integrity import compute_episode_hash
from sfp.prediction.world_model import PredictiveWorldModel
from sfp.types import Episode


DIM = 32


# ============================================================
# World model head methods
# ============================================================


class TestContinueHead:
    def test_predict_continue_no_state_returns_1(self):
        """Fresh WM with zero state should return 1.0 (no boundary)."""
        wm = PredictiveWorldModel(WorldModelConfig(), d_model=DIM)
        assert wm.predict_continue_probability() == pytest.approx(1.0)

    def test_predict_continue_after_step(self):
        """After a step, continue probability should be in [0, 1]."""
        wm = PredictiveWorldModel(WorldModelConfig(), d_model=DIM)
        obs = torch.randn(DIM)
        wm.train_step(obs)
        prob = wm.predict_continue_probability()
        assert 0.0 <= prob <= 1.0

    def test_continue_in_train_step_losses(self):
        """train_step should include continue_probability in return dict."""
        wm = PredictiveWorldModel(WorldModelConfig(), d_model=DIM)
        obs = torch.randn(DIM)
        losses = wm.train_step(obs)
        assert "continue_probability" in losses
        assert 0.0 <= losses["continue_probability"] <= 1.0


class TestSpatialPredictionError:
    def test_no_prev_position_returns_none(self):
        """Without a previous position, error should be None."""
        wm = PredictiveWorldModel(WorldModelConfig(), d_model=DIM)
        obs = torch.randn(DIM)
        wm.train_step(obs)  # no spatial_position
        assert wm.compute_spatial_prediction_error((1.0, 2.0, 3.0)) is None

    def test_returns_float_after_two_steps(self):
        """After two steps with positions, error should be a positive float."""
        wm = PredictiveWorldModel(WorldModelConfig(), d_model=DIM)
        obs = torch.randn(DIM)
        wm.train_step(obs, spatial_position=(0.0, 0.0, 0.0))
        wm.train_step(obs, spatial_position=(10.0, 0.0, 0.0))
        err = wm.compute_spatial_prediction_error((20.0, 0.0, 0.0))
        assert err is not None
        assert err >= 0.0


# ============================================================
# Force-store in EpisodicMemory
# ============================================================


def _make_episode(ep_id: int, surprise: float = 0.5, dim: int = DIM) -> Episode:
    """Create a test episode with a valid integrity hash."""
    inp = torch.randn(dim)
    logit = torch.randn(dim)
    wh = hashlib.sha256(b"dummy").digest()
    ih = compute_episode_hash(inp, logit, wh)
    return Episode(
        id=ep_id,
        timestamp=0.0,
        modality="test",
        provenance_hash="test",
        input_embedding=inp,
        working_memory_state=torch.zeros(dim),
        logit_snapshot=logit,
        surprise_at_storage=surprise,
        attractor_basin_id=0,
        attractor_distance=0.1,
        preceding_episode_id=None,
        following_episode_id=None,
        integrity_hash=ih,
        weight_hash_at_storage=wh,
    )


class TestForceStore:
    def test_force_store_bypasses_surprise_threshold(self):
        """force_store should accept episodes that maybe_store would reject."""
        mem = EpisodicMemory(
            Tier1Config(
                hot_capacity=100,
                surprise_threshold=10.0,  # very high threshold
                dedup_threshold=1.0,
            ),
            d_model=DIM,
        )
        ep = _make_episode(0, surprise=0.01)  # very low surprise

        # maybe_store should reject (below threshold)
        assert mem.maybe_store(ep) is False
        assert mem.hot_count == 0

        # force_store should accept (bypasses surprise gate)
        assert mem.force_store(ep) is True
        assert mem.hot_count == 1

    def test_force_store_respects_dedup(self):
        """force_store should still reject near-duplicates."""
        mem = EpisodicMemory(
            Tier1Config(
                hot_capacity=100,
                surprise_threshold=0.0,
                dedup_threshold=0.99,  # strict dedup
            ),
            d_model=DIM,
        )
        ep1 = _make_episode(0)
        assert mem.force_store(ep1) is True

        # Same episode (same embedding) should be rejected by dedup
        ep2 = Episode(
            id=1,
            timestamp=0.0,
            modality="test",
            provenance_hash="test",
            input_embedding=ep1.input_embedding.clone(),  # same embedding
            working_memory_state=ep1.working_memory_state.clone(),
            logit_snapshot=ep1.logit_snapshot.clone(),
            surprise_at_storage=0.5,
            attractor_basin_id=0,
            attractor_distance=0.1,
            preceding_episode_id=None,
            following_episode_id=None,
            integrity_hash=ep1.integrity_hash,
            weight_hash_at_storage=ep1.weight_hash_at_storage,
        )
        assert mem.force_store(ep2) is False
        assert mem.hot_count == 1

    def test_force_store_respects_integrity(self):
        """force_store should reject episodes with invalid integrity hash."""
        mem = EpisodicMemory(
            Tier1Config(
                hot_capacity=100,
                surprise_threshold=0.0,
                dedup_threshold=1.0,
            ),
            d_model=DIM,
        )
        ep = _make_episode(0)
        # Corrupt the integrity hash
        ep.integrity_hash = b"\x00" * 32
        assert mem.force_store(ep) is False
        assert mem.hot_count == 0
