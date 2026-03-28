"""Tests for curriculum/difficulty scheduling in StreamingProcessor."""

from __future__ import annotations

import pytest
import torch

from sfp.config import FieldConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor

DIM = 32
N_LAYERS = 2


def _field() -> SemanticFieldProcessor:
    return SemanticFieldProcessor(FieldConfig(dim=DIM, n_layers=N_LAYERS))


def _processor(**overrides) -> StreamingProcessor:
    defaults = dict(
        surprise_threshold=0.0,
        soft_gate_enabled=False,
        lr_decay_enabled=False,
        gradient_conflict_detection=False,
    )
    defaults.update(overrides)
    cfg = StreamingConfig(**defaults)
    return StreamingProcessor(_field(), streaming_config=cfg)


class TestCurriculumScheduling:
    def test_disabled_by_default(self):
        """Default config has curriculum disabled; competence EMA stays 0."""
        proc = _processor()
        assert proc.config.curriculum_enabled is False
        x = torch.randn(DIM)
        for _ in range(10):
            proc.process(x)
        assert proc._competence_ema == 0.0

    def test_competence_ema_tracks_loss(self):
        """With curriculum enabled, competence EMA should track average loss."""
        proc = _processor(curriculum_enabled=True)
        for _ in range(10):
            x = torch.randn(DIM)
            proc.process(x)
        assert proc._competence_ema > 0.0

    def test_warmup_prevents_early_scaling(self):
        """During warmup, curriculum scale should always be 1.0."""
        proc = _processor(curriculum_enabled=True, curriculum_warmup_steps=20)
        # Simulate a high competence but we're still in warmup
        proc._competence_ema = 1.0
        proc._curriculum_active_step = 5
        scale = proc._compute_curriculum_scale(0.01)  # very easy
        assert scale == 1.0  # still in warmup

    def test_too_easy_scales_down(self):
        """Loss far below competence should return too_easy_scale."""
        proc = _processor(
            curriculum_enabled=True,
            curriculum_warmup_steps=0,
            curriculum_too_easy_threshold=0.2,
            curriculum_too_easy_scale=0.3,
        )
        # Prime competence EMA
        proc._competence_ema = 1.0
        proc._curriculum_active_step = 200  # past warmup
        # Loss = 0.1 < competence (1.0) * 0.2 = 0.2 → too easy
        scale = proc._compute_curriculum_scale(0.1)
        assert scale == pytest.approx(0.3)

    def test_too_hard_scales_down(self):
        """Loss far above competence should return too_hard_scale."""
        proc = _processor(
            curriculum_enabled=True,
            curriculum_warmup_steps=0,
            curriculum_too_hard_ratio=5.0,
            curriculum_too_hard_scale=0.3,
        )
        proc._competence_ema = 1.0
        proc._curriculum_active_step = 200
        # Loss = 6.0 > competence (1.0) * 5.0 = 5.0 → too hard
        scale = proc._compute_curriculum_scale(6.0)
        assert scale == pytest.approx(0.3)

    def test_zpd_returns_full_scale(self):
        """Loss within ZPD should return 1.0."""
        proc = _processor(
            curriculum_enabled=True,
            curriculum_warmup_steps=0,
            curriculum_too_easy_threshold=0.2,
            curriculum_too_hard_ratio=5.0,
        )
        proc._competence_ema = 1.0
        proc._curriculum_active_step = 200
        # Loss = 0.5 is between 0.2 and 5.0 → in ZPD
        scale = proc._compute_curriculum_scale(0.5)
        assert scale == 1.0

    def test_curriculum_scale_in_loss_components(self):
        """When curriculum scales, it should appear in loss_components."""
        proc = _processor(
            curriculum_enabled=True,
            curriculum_warmup_steps=0,
            curriculum_too_hard_ratio=0.01,  # almost everything is "too hard"
            curriculum_too_hard_scale=0.5,
        )
        # Prime with a very low competence so almost any input is "too hard"
        proc._competence_ema = 1e-6
        proc._curriculum_active_step = 200
        x = torch.randn(DIM)
        metric = proc.process(x)
        # The curriculum scale should be recorded
        assert "curriculum_scale" in metric.loss_components

    def test_reset_clears_curriculum_state(self):
        """reset_working_memory should clear curriculum state."""
        proc = _processor(curriculum_enabled=True)
        for _ in range(5):
            proc.process(torch.randn(DIM))
        assert proc._competence_ema > 0.0
        assert proc._curriculum_active_step > 0

        proc.reset_working_memory()
        assert proc._competence_ema == 0.0
        assert proc._curriculum_active_step == 0
