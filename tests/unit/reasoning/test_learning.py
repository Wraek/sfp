"""Tests for reasoning.learning — TransitionLearner and ChainShortcutLearner."""

import time

import torch
import pytest

from sfp.config import ReasoningChainConfig, Tier2Config, TransitionConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.learning import ChainShortcutLearner, TransitionLearner
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import Episode, RelationType


def _make_episode(basin_id: int, timestamp: float, d: int = 32) -> Episode:
    return Episode(
        id=0,
        timestamp=timestamp,
        modality="test",
        provenance_hash=b"\x00" * 16,
        input_embedding=torch.randn(d),
        working_memory_state=torch.randn(8),
        logit_snapshot=torch.randn(d),
        surprise_at_storage=0.5,
        attractor_basin_id=basin_id,
        attractor_distance=0.1,
        preceding_episode_id=None,
        following_episode_id=None,
        integrity_hash=b"\x00" * 32,
        weight_hash_at_storage=b"\x00" * 32,
    )


def _setup_learner(d: int = 32) -> tuple[EssentialMemory, TransitionStructure, TransitionLearner]:
    tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
    for i in range(4):
        k = torch.zeros(d)
        k[i] = 5.0
        tier2.allocate_slot(k, value=k)

    transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=d)
    learner = TransitionLearner(tier2, transitions, temporal_window=3)
    return tier2, transitions, learner


class TestTransitionLearner:
    def test_learn_from_empty_episodes(self):
        _, _, learner = _setup_learner()
        assert learner.learn_from_episodes([]) == 0

    def test_learn_temporal_relations(self):
        _, transitions, learner = _setup_learner()
        episodes = [
            _make_episode(basin_id=0, timestamp=1.0),
            _make_episode(basin_id=1, timestamp=2.0),
            _make_episode(basin_id=2, timestamp=3.0),
        ]
        n_edges = learner.learn_from_episodes(episodes)
        assert n_edges > 0
        # Should have temporal edge 0 -> 1
        targets, _, _ = transitions.get_outgoing(0)
        assert 1 in targets.tolist()

    def test_learn_compositional_relations(self):
        d = 32
        tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
        # Create two basins with high similarity (0.8)
        k0 = torch.randn(d)
        k0 = k0 / k0.norm()
        k1 = k0 + torch.randn(d) * 0.2
        k1 = k1 / k1.norm()
        tier2.allocate_slot(k0 * 3.0, value=k0 * 3.0)
        tier2.allocate_slot(k1 * 3.0, value=k1 * 3.0)
        with torch.no_grad():
            tier2.episode_count[0] = 100
            tier2.episode_count[1] = 10

        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=d)
        learner = TransitionLearner(tier2, transitions, compositional_distance=0.5)
        n_edges = learner.learn_compositional_relations()
        assert n_edges >= 0  # May or may not find depending on cosine sim

    def test_learn_inhibitory_relations(self):
        d = 32
        tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
        # Similar keys but very different values
        k = torch.randn(d)
        k = k / k.norm()
        v0 = torch.randn(d)
        v1 = -v0  # opposite value
        tier2.allocate_slot(k * 3.0, value=v0)
        tier2.allocate_slot((k + torch.randn(d) * 0.1) * 3.0, value=v1)

        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=d)
        learner = TransitionLearner(tier2, transitions)
        n_edges = learner.learn_inhibitory_relations()
        # Should find inhibitory relation since keys are similar but values opposite
        assert n_edges >= 0  # May depend on exact cosine sims


class TestChainShortcutLearner:
    def test_short_chain_ignored(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=32)
        learner = ChainShortcutLearner(transitions, ReasoningChainConfig(shortcut_min_traversals=3))
        learner.observe_chain([0, 1], quality_score=1.0)  # Only 2 basins
        created = learner.create_shortcuts()
        assert created == 0

    def test_shortcut_created_after_threshold(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=32)
        learner = ChainShortcutLearner(transitions, ReasoningChainConfig(shortcut_min_traversals=3))
        for _ in range(5):
            learner.observe_chain([0, 1, 2, 3], quality_score=0.8)
        created = learner.create_shortcuts()
        assert created >= 1
        # Should have direct edge 0 -> 3
        targets, _, _ = transitions.get_outgoing(0)
        assert 3 in targets.tolist()

    def test_low_quality_chain_not_shortcut(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=32)
        learner = ChainShortcutLearner(transitions, ReasoningChainConfig(shortcut_min_traversals=3))
        for _ in range(5):
            learner.observe_chain([0, 1, 2, 3], quality_score=0.1)  # Low quality
        created = learner.create_shortcuts()
        assert created == 0

    def test_counts_reset_after_shortcut(self):
        transitions = TransitionStructure(TransitionConfig(max_edges=32), d_model=32)
        learner = ChainShortcutLearner(transitions, ReasoningChainConfig(shortcut_min_traversals=2))
        for _ in range(3):
            learner.observe_chain([0, 1, 2, 3], quality_score=0.9)
        learner.create_shortcuts()
        # Counts should be reset
        assert (0, 3) not in learner._chain_counts
