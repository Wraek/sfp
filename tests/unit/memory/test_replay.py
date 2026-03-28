"""Tests for memory.replay — GenerativeReplay."""

import torch
import pytest

from sfp.config import GenerativeReplayConfig
from sfp.memory.replay import GenerativeReplay


class _MockTier2:
    """Lightweight mock for EssentialMemory (duck-typed)."""

    def __init__(self, n_active: int = 20, d: int = 32):
        self.n_active = n_active
        self.keys = torch.randn(n_active, d)
        # Normalize keys so cosine similarity behaves predictably
        self.keys = self.keys / self.keys.norm(dim=-1, keepdim=True) * 3.0
        self.confidence = torch.ones(n_active) * 0.8


class TestGenerativeReplayShouldGenerate:
    def test_disabled_during_warmup(self):
        cfg = GenerativeReplayConfig(warmup_episodes=100)
        gr = GenerativeReplay(cfg, d_model=32)
        should, n = gr.should_generate(cycle_count=1, total_episodes=50)
        assert not should

    def test_enabled_after_warmup(self):
        cfg = GenerativeReplayConfig(
            warmup_episodes=10,
            middle_episodes=100,
            middle_cycle_interval=1,
            middle_synthetics=5,
        )
        gr = GenerativeReplay(cfg, d_model=32)
        should, n = gr.should_generate(cycle_count=1, total_episodes=50)
        assert should
        assert n == 5

    def test_idle_daydreaming(self):
        cfg = GenerativeReplayConfig(
            warmup_episodes=10,
            idle_timeout_seconds=5.0,
        )
        gr = GenerativeReplay(cfg, d_model=32)
        should, n = gr.should_generate(cycle_count=1, total_episodes=50, idle_seconds=10.0)
        assert should


class TestGenerativeReplayValidation:
    def test_validate_nan_rejected(self):
        gr = GenerativeReplay(d_model=32)
        emb = torch.tensor([float("nan")] * 32)
        tier2 = _MockTier2(n_active=5)
        assert not gr.validate_synthetic(emb, tier2)

    def test_validate_inf_rejected(self):
        gr = GenerativeReplay(d_model=32)
        emb = torch.tensor([float("inf")] * 32)
        tier2 = _MockTier2(n_active=5)
        assert not gr.validate_synthetic(emb, tier2)

    def test_validate_zero_norm_rejected(self):
        gr = GenerativeReplay(d_model=32)
        emb = torch.zeros(32)
        tier2 = _MockTier2(n_active=5)
        assert not gr.validate_synthetic(emb, tier2)

    def test_validate_normal_embedding_passes(self):
        gr = GenerativeReplay(
            GenerativeReplayConfig(manifold_proximity_threshold=2.0),
            d_model=32,
        )
        tier2 = _MockTier2(n_active=20)
        # Use a key close to existing basins
        emb = tier2.keys[0] + torch.randn(32) * 0.1
        assert gr.validate_synthetic(emb, tier2)


class TestGenerativeReplayDriftMonitoring:
    def test_drift_not_excessive_initially(self):
        gr = GenerativeReplay(d_model=32)
        assert not gr.is_drift_excessive(0)

    def test_excessive_drift_detection(self):
        cfg = GenerativeReplayConfig(drift_throttle_multiplier=1.5, drift_ema_decay=0.5)
        gr = GenerativeReplay(cfg, d_model=32)
        # Build baseline with small drifts
        old = torch.randn(32)
        for _ in range(25):
            new = old + torch.randn(32) * 0.001
            gr.update_drift_monitoring(0, old, new)
            old = new
        # Now big drift
        for _ in range(25):
            new = old + torch.randn(32) * 10.0
            gr.update_drift_monitoring(0, old, new)
            old = new
        assert gr.is_drift_excessive(0)

    def test_excluded_basins(self):
        gr = GenerativeReplay(d_model=32)
        # No drift data → no exclusions
        excluded = gr.get_excluded_basins()
        assert len(excluded) == 0


class TestGenerativeReplayStats:
    def test_stats_initial(self):
        gr = GenerativeReplay(d_model=32)
        stats = gr.get_generation_stats()
        assert stats["total_generated"] == 0
        assert stats["total_validated"] == 0

    def test_reset_stats(self):
        gr = GenerativeReplay(d_model=32)
        gr._total_generated = 100
        gr._total_validated = 50
        gr.reset_stats()
        assert gr._total_generated == 0
        assert gr._total_validated == 0

    def test_record_inference(self):
        gr = GenerativeReplay(d_model=32)
        t0 = gr._last_inference_time
        import time
        time.sleep(0.01)
        gr.record_inference()
        assert gr._last_inference_time > t0


class TestGenerativeReplayInterpolation:
    def test_interpolation_needs_min_basins(self):
        gr = GenerativeReplay(d_model=32)
        tier2 = _MockTier2(n_active=5)  # Less than 10
        result = gr.generate_interpolation(tier2)
        assert result is None

    def test_boundary_probe_needs_min_basins(self):
        gr = GenerativeReplay(d_model=32)
        tier2 = _MockTier2(n_active=3)
        result = gr.generate_boundary_probe(tier2)
        assert result is None
