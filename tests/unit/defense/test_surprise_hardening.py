"""Tests for defense.surprise_hardening — SurpriseHardener."""

import torch
import torch.nn as nn
import pytest

from sfp.config import Tier0Config
from sfp.defense.surprise_hardening import SurpriseHardener


class TestSurpriseHardener:
    def test_surprise_ratio_clamped(self):
        cfg = Tier0Config(max_surprise_ratio=3.0)
        sh = SurpriseHardener(cfg)
        # Build baseline with small grad norms
        for _ in range(20):
            sh.compute_hardened_surprise(1.0)
        # Spike
        result = sh.compute_hardened_surprise(1000.0)
        assert result <= 3.0

    def test_normal_surprise_passes_through(self):
        cfg = Tier0Config(max_surprise_ratio=10.0)
        sh = SurpriseHardener(cfg)
        # Build EMA
        for _ in range(20):
            sh.compute_hardened_surprise(1.0)
        # Normal
        result = sh.compute_hardened_surprise(1.0)
        assert 0.0 <= result <= 10.0

    def test_rate_limiting_activates(self):
        cfg = Tier0Config(
            max_surprise_ratio=100.0,
            rate_limit_window=10,
            rate_limit_threshold=0.3,
            surprise_momentum=0.99,  # slow EMA so ratio stays high
        )
        sh = SurpriseHardener(cfg)
        # Build baseline with slow EMA
        for _ in range(5):
            sh.compute_hardened_surprise(1.0)
        # Burst of escalating high-surprise events
        activated = False
        for i in range(20):
            sh.compute_hardened_surprise(100.0 * (i + 1))
            if sh.is_rate_limited:
                activated = True
                break
        assert activated, "Rate limiting should have activated during burst"

    def test_rate_limiting_deactivates(self):
        cfg = Tier0Config(
            max_surprise_ratio=100.0,
            rate_limit_window=10,
            rate_limit_threshold=0.3,
            surprise_momentum=0.99,  # slow EMA
        )
        sh = SurpriseHardener(cfg)
        # Build baseline
        for _ in range(5):
            sh.compute_hardened_surprise(1.0)
        # Trigger rate limiting with escalating values
        for i in range(20):
            sh.compute_hardened_surprise(100.0 * (i + 1))
        # Now feed normal values matching current EMA level to deactivate
        for _ in range(50):
            sh.compute_hardened_surprise(sh.surprise_ema)
        assert not sh.is_rate_limited

    def test_dual_path_reduces_surprise_on_disagreement(self):
        cfg = Tier0Config(max_surprise_ratio=10.0)
        sh = SurpriseHardener(cfg)
        # Build baseline
        for _ in range(20):
            sh.compute_hardened_surprise(1.0)
        # High grad norm but low latent distance = disagreement
        high_grad = sh.compute_hardened_surprise(5.0)
        sh2 = SurpriseHardener(cfg)
        for _ in range(20):
            sh2.compute_hardened_surprise(1.0)
        low_latent = sh2.compute_hardened_surprise(5.0, latent_distance=0.01)
        assert low_latent < high_grad

    def test_clip_gradients_reduces_outliers(self):
        cfg = Tier0Config()
        sh = SurpriseHardener(cfg)
        model = nn.Linear(16, 16)
        named_params = list(model.named_parameters())

        # Build EMA with small gradients
        for _ in range(10):
            model.zero_grad()
            x = torch.randn(4, 16) * 0.01
            model(x).sum().backward()
            sh.clip_gradients(named_params)

        # Spike gradient
        model.zero_grad()
        x = torch.randn(4, 16) * 100.0
        model(x).sum().backward()
        ratio = sh.clip_gradients(named_params)
        assert ratio < 1.0  # Clipping occurred

    def test_reset_clears_state(self):
        cfg = Tier0Config()
        sh = SurpriseHardener(cfg)
        for _ in range(20):
            sh.compute_hardened_surprise(1.0)
        assert sh.surprise_ema > 0
        sh.reset()
        assert sh.surprise_ema == 0.0
        assert not sh.is_rate_limited
