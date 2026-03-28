"""Tests for memory.events — PromotionEventEmitter and CriteriaAuthorizationHandler."""

import torch
import pytest

from sfp.memory.events import CriteriaAuthorizationHandler, PromotionEventEmitter
from sfp.types import PromotionRequest


def _make_request(**overrides) -> PromotionRequest:
    defaults = dict(
        basin_id=0,
        confidence=0.95,
        episode_count=2000,
        modality_count=3,
        age_days=10.0,
        key_snapshot=torch.randn(32),
        value_snapshot=torch.randn(32),
    )
    defaults.update(overrides)
    return PromotionRequest(**defaults)


class TestPromotionEventEmitter:
    def test_default_deny(self):
        emitter = PromotionEventEmitter(default_approve=False)
        assert not emitter.emit(_make_request())

    def test_default_approve(self):
        emitter = PromotionEventEmitter(default_approve=True)
        assert emitter.emit(_make_request())

    def test_handler_approve(self):
        emitter = PromotionEventEmitter(default_approve=False)
        emitter.register(lambda req: True)
        assert emitter.emit(_make_request())

    def test_handler_deny(self):
        emitter = PromotionEventEmitter(default_approve=True)
        emitter.register(lambda req: False)
        assert not emitter.emit(_make_request())

    def test_handler_defer(self):
        emitter = PromotionEventEmitter(default_approve=True)
        emitter.register(lambda req: None)  # defer
        assert emitter.emit(_make_request())  # falls to default

    def test_handler_priority_order(self):
        emitter = PromotionEventEmitter(default_approve=False)
        emitter.register(lambda req: True)  # first handler approves
        emitter.register(lambda req: False)  # second would deny
        assert emitter.emit(_make_request())  # first wins

    def test_unregister(self):
        emitter = PromotionEventEmitter(default_approve=False)
        handler = lambda req: True
        emitter.register(handler)
        assert emitter.handler_count == 1
        emitter.unregister(handler)
        assert emitter.handler_count == 0
        assert not emitter.emit(_make_request())  # back to default deny


class TestCriteriaAuthorizationHandler:
    def test_all_criteria_met_approves(self):
        handler = CriteriaAuthorizationHandler(
            min_confidence=0.9,
            min_episode_count=1000,
            min_modalities=2,
            min_age_days=7.0,
        )
        req = _make_request(confidence=0.95, episode_count=2000, modality_count=3, age_days=10.0)
        assert handler(req) is True

    def test_low_confidence_defers(self):
        handler = CriteriaAuthorizationHandler(min_confidence=0.9)
        req = _make_request(confidence=0.5)
        assert handler(req) is None

    def test_low_episodes_defers(self):
        handler = CriteriaAuthorizationHandler(min_episode_count=5000)
        req = _make_request(episode_count=100)
        assert handler(req) is None

    def test_low_modalities_defers(self):
        handler = CriteriaAuthorizationHandler(min_modalities=5)
        req = _make_request(modality_count=2)
        assert handler(req) is None

    def test_young_defers(self):
        handler = CriteriaAuthorizationHandler(min_age_days=30.0)
        req = _make_request(age_days=1.0)
        assert handler(req) is None

    def test_integrated_with_emitter(self):
        emitter = PromotionEventEmitter(default_approve=False)
        handler = CriteriaAuthorizationHandler(
            min_confidence=0.5, min_episode_count=10, min_modalities=1, min_age_days=0.01,
        )
        emitter.register(handler)
        req = _make_request(confidence=0.95, episode_count=100, modality_count=2, age_days=1.0)
        assert emitter.emit(req)
