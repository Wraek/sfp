"""Tests for Phase 2 feedback loop closures.

Covers:
  2A. Metacognition health → consolidation action
  2B. Salience hindsight ground truth population
  2C. Surprise-weighted replay prioritization
  2D. World model reward → valence system
  2E. Edge valence population
"""

from __future__ import annotations

import random

import pytest
import torch

import hashlib

from sfp.config import Tier1Config
from sfp.memory.episodic import EpisodicMemory
from sfp.types import Episode


# ============================================================
# 2C — Surprise-weighted replay prioritization
# ============================================================


def _make_episode(
    ep_id: int,
    basin_id: int = 0,
    surprise: float = 0.5,
    dim: int = 32,
) -> Episode:
    """Create a test episode with given surprise."""
    inp = torch.randn(dim)
    logit = torch.randn(dim)
    # Dummy hashes — integrity is not under test here
    wh = hashlib.sha256(b"dummy").digest()
    ih = hashlib.sha256(inp.numpy().tobytes() + logit.numpy().tobytes()).digest()
    return Episode(
        id=ep_id,
        timestamp=0.0,
        modality="test",
        provenance_hash="test",
        input_embedding=inp,
        working_memory_state=torch.zeros(dim),
        logit_snapshot=logit,
        surprise_at_storage=surprise,
        attractor_basin_id=basin_id,
        attractor_distance=0.1,
        preceding_episode_id=None,
        following_episode_id=None,
        integrity_hash=ih,
        weight_hash_at_storage=wh,
    )


class TestSurpriseWeightedReplay:
    def test_high_surprise_episodes_sampled_more(self):
        """Episodes with high surprise should be sampled more frequently."""
        mem = EpisodicMemory(
            Tier1Config(
                hot_capacity=200, surprise_threshold=0.0, dedup_threshold=1.0,
            ),
            d_model=32,
        )

        # Store 50 low-surprise and 50 high-surprise episodes in same basin
        for i in range(50):
            mem._hot.append(_make_episode(i, basin_id=0, surprise=0.01))
        for i in range(50, 100):
            mem._hot.append(_make_episode(i, basin_id=0, surprise=10.0))

        # Sample many times and count which group appears more
        random.seed(42)
        high_count = 0
        low_count = 0
        for _ in range(20):
            batch = mem.sample_for_replay(20)
            for ep in batch:
                if ep.surprise_at_storage > 1.0:
                    high_count += 1
                else:
                    low_count += 1

        # High-surprise should be sampled significantly more
        assert high_count > low_count * 2, (
            f"Expected high-surprise to dominate: high={high_count}, low={low_count}"
        )

    def test_weighted_sampling_still_stratifies(self):
        """Surprise weighting should work within basin stratification."""
        mem = EpisodicMemory(
            Tier1Config(
                hot_capacity=200, surprise_threshold=0.0, dedup_threshold=1.0,
            ),
            d_model=32,
        )

        # Two basins with different sizes
        for i in range(30):
            mem._hot.append(_make_episode(i, basin_id=0, surprise=1.0))
        for i in range(30, 50):
            mem._hot.append(_make_episode(i, basin_id=1, surprise=1.0))

        random.seed(42)
        batch = mem.sample_for_replay(20)
        basins_sampled = {ep.attractor_basin_id for ep in batch}
        # Both basins should be represented
        assert 0 in basins_sampled
        assert 1 in basins_sampled


# ============================================================
# 2A — Metacognition health (unit-level check)
# ============================================================


class TestMetacognitionHealth:
    def test_monitor_health_returns_report(self):
        """monitor_memory_health should return a health report dict."""
        from sfp.config import MetacognitionConfig
        from sfp.metacognition.uncertainty import MetacognitionEngine

        cfg = MetacognitionConfig()
        estimator = MetacognitionEngine(cfg, d_model=32)

        # Create fake basin data
        keys = torch.randn(100, 32)
        confidence = torch.full((100,), 0.8)
        n_active = 10

        report = estimator.monitor_memory_health(keys, confidence, n_active)
        assert "dormant_basins" in report
        assert "declining_basins" in report
        assert "total_active_basins" in report
        assert report["total_active_basins"] == 10


# ============================================================
# 2B — Salience hindsight (unit-level check)
# ============================================================


class TestSalienceHindsight:
    def test_hindsight_populates_buffer(self):
        """train_hindsight should add entries to the hindsight buffer."""
        from sfp.config import SelectiveAttentionConfig
        from sfp.attention.salience import SalienceGate

        cfg = SelectiveAttentionConfig()
        gate = SalienceGate(cfg, d_model=32)

        assert len(gate._hindsight_buffer) == 0

        emb = torch.randn(32)
        gate.train_hindsight(0.7, True, emb)
        gate.train_hindsight(0.3, False, emb)

        assert len(gate._hindsight_buffer) == 2
        assert gate._hindsight_buffer[0]["label"] == 1.0
        assert gate._hindsight_buffer[1]["label"] == 0.0


# ============================================================
# 2D — World model reward → valence (unit-level check)
# ============================================================


class TestWMRewardToValence:
    def test_valence_accepts_reward(self):
        """ValenceSystem.compute_valence should use the reward parameter."""
        from sfp.config import ValenceConfig
        from sfp.affect.valence import ValenceSystem

        cfg = ValenceConfig()
        vs = ValenceSystem(cfg, d_model=32)

        emb = torch.randn(32)

        # Compute with positive reward
        sig_pos = vs.compute_valence(emb, reward=1.0)
        # Compute with negative reward
        sig_neg = vs.compute_valence(emb, reward=-1.0)

        # The scalar valence should differ with different rewards
        # (exact difference depends on normalization, but direction should differ)
        assert sig_pos.scalar_valence != sig_neg.scalar_valence


# ============================================================
# 2E — Edge valence (unit-level check)
# ============================================================


class TestEdgeValence:
    def test_annotate_edge_updates_buffer(self):
        """annotate_edge should update the edge valence buffer."""
        from sfp.config import ValenceConfig
        from sfp.affect.valence import ValenceSystem

        cfg = ValenceConfig()
        vs = ValenceSystem(cfg, d_model=32)

        # Edge 0 starts at 0.0
        assert vs.edge_valence[0].item() == pytest.approx(0.0)

        vs.annotate_edge(0, 0.8)
        assert vs.edge_valence[0].item() != 0.0

        # Multiple annotations should EMA
        vs.annotate_edge(0, -0.5)
        # Value should shift toward -0.5
