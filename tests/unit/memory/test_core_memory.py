"""Tests for memory.core — CoreMemory (Tier 3)."""

import time

import torch
import pytest

from sfp.config import Tier2Config, Tier3Config
from sfp.memory.core import CoreMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.events import PromotionEventEmitter


def _make_tier2(d: int = 32, n_slots: int = 8) -> EssentialMemory:
    tier2 = EssentialMemory(Tier2Config(n_slots=n_slots, d_value=d), d_model=d)
    return tier2


class TestCoreMemory:
    def test_retrieve_empty_returns_zeros(self):
        core = CoreMemory(Tier3Config(n_slots=4, d_value=32), d_model=32)
        q = torch.randn(32)
        out = core.retrieve(q)
        assert out.shape == (32,)
        assert out.norm().item() < 1e-6

    def test_write_slot_and_retrieve(self):
        core = CoreMemory(Tier3Config(n_slots=4, d_value=32), d_model=32)
        key = torch.randn(32)
        value = torch.randn(32)
        core._write_slot(key, value, confidence=0.9, episode_count=100, modality_mask=3)
        assert core.n_active == 1
        # Retrieve should return non-zero
        out = core.retrieve(key)
        assert out.norm().item() > 0

    def test_integrity_verification_passes(self):
        core = CoreMemory(Tier3Config(n_slots=4, d_value=32), d_model=32)
        core._write_slot(torch.randn(32), torch.randn(32), 0.9, 100, 1)
        failed = core.verify_integrity()
        assert failed == []

    def test_integrity_verification_fails_on_tamper(self):
        core = CoreMemory(Tier3Config(n_slots=4, d_value=32), d_model=32)
        core._write_slot(torch.randn(32), torch.randn(32), 0.9, 100, 1)
        # Tamper with the key
        with torch.no_grad():
            core.keys.data[0] = torch.randn(32)
        failed = core.verify_integrity()
        assert 0 in failed

    def test_promote_rejected_low_confidence(self):
        d = 32
        tier2 = _make_tier2(d)
        tier2.allocate_slot(torch.randn(d))
        # Set low confidence
        with torch.no_grad():
            tier2.confidence[0] = 0.1
            tier2.episode_count[0] = 10000
            tier2.modality_mask[0] = 0b111
            tier2.created_at[0] = time.time() - 86400 * 30
        emitter = PromotionEventEmitter(default_approve=True)
        core = CoreMemory(
            Tier3Config(n_slots=4, d_value=d, min_confidence=0.9),
            d_model=d,
            event_emitter=emitter,
        )
        result = core.promote_from_tier2(tier2, 0)
        assert not result

    def test_promote_rejected_by_event_system(self):
        d = 32
        tier2 = _make_tier2(d)
        tier2.allocate_slot(torch.randn(d))
        with torch.no_grad():
            tier2.confidence[0] = 0.95
            tier2.episode_count[0] = 10000
            tier2.modality_mask[0] = 0b111
            tier2.created_at[0] = time.time() - 86400 * 30

        # Event emitter default denies
        emitter = PromotionEventEmitter(default_approve=False)
        core = CoreMemory(
            Tier3Config(n_slots=4, d_value=d, min_confidence=0.5, min_episode_count=10, min_modalities=1, min_age_days=0.001),
            d_model=d,
            event_emitter=emitter,
        )
        result = core.promote_from_tier2(tier2, 0)
        assert not result

    def test_eviction_when_full(self):
        core = CoreMemory(Tier3Config(n_slots=2, d_value=16), d_model=16)
        core._write_slot(torch.randn(16), torch.randn(16), 0.5, 10, 1)
        core._write_slot(torch.randn(16), torch.randn(16), 0.9, 10, 1)
        assert core.n_active == 2
        # Adding a third should evict the lowest-confidence
        core._write_slot(torch.randn(16), torch.randn(16), 0.8, 10, 1)
        assert core.n_active == 2  # evicted one, added one => still 2 (3 - 1)
        # The slot with 0.5 confidence should have been evicted

    def test_batch_retrieve(self):
        core = CoreMemory(Tier3Config(n_slots=4, d_value=32), d_model=32)
        core._write_slot(torch.randn(32), torch.randn(32), 0.9, 100, 1)
        q = torch.randn(3, 32)
        out = core.retrieve(q)
        assert out.shape == (3, 32)
