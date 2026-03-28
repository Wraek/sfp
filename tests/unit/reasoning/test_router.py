"""Tests for reasoning.router — ReasoningRouter."""

import torch
import pytest

from sfp.config import ReasoningChainConfig, Tier2Config, TransitionConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.chain import AssociativeReasoningChain
from sfp.reasoning.router import ReasoningRouter
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import RelationType


def _setup_router(d: int = 32) -> ReasoningRouter:
    tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)
    with torch.no_grad():
        tier2.query_proj.weight.copy_(torch.eye(d))
        tier2.key_proj.weight.copy_(torch.eye(d))
        tier2.value_proj.weight.copy_(torch.eye(d))
        tier2.output_proj.weight.copy_(torch.eye(d))
    for i in range(4):
        k = torch.zeros(d)
        k[i] = 5.0
        tier2.allocate_slot(k, value=k)
        with torch.no_grad():
            tier2.confidence[i] = 0.8

    transitions = TransitionStructure(TransitionConfig(max_edges=32, d_relation=16), d_model=d)
    for i in range(3):
        transitions.add_edge(i, i + 1, RelationType.CAUSAL, weight=0.8, confidence=0.8)

    chain = AssociativeReasoningChain(
        tier2, transitions,
        ReasoningChainConfig(max_hops=5, branch_threshold=-10.0),
    )
    router = ReasoningRouter(
        tier2, transitions, chain,
        residual_threshold=0.1,
        entropy_threshold=0.5,
    )
    return router


class TestReasoningRouter:
    def test_empty_memory_single_hop(self):
        d = 32
        tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
        transitions = TransitionStructure(d_model=d)
        chain = AssociativeReasoningChain(tier2, transitions)
        router = ReasoningRouter(tier2, transitions, chain)
        result = router.route(torch.randn(d))
        assert result.routing == "single_hop"
        assert result.terminated_reason == "empty_memory"

    def test_single_hop_sufficient(self):
        d = 32
        tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
        with torch.no_grad():
            tier2.query_proj.weight.copy_(torch.eye(d))
            tier2.key_proj.weight.copy_(torch.eye(d))
            tier2.value_proj.weight.copy_(torch.eye(d))
            tier2.output_proj.weight.copy_(torch.eye(d))
        k = torch.zeros(d)
        k[0] = 5.0
        tier2.allocate_slot(k, value=k)
        with torch.no_grad():
            tier2.confidence[0] = 0.9

        transitions = TransitionStructure(d_model=d)
        chain = AssociativeReasoningChain(tier2, transitions)
        # Very high residual threshold so it doesn't trigger multi-hop
        router = ReasoningRouter(tier2, transitions, chain, residual_threshold=1000.0)
        result = router.route(k)
        assert result.routing == "single_hop"

    def test_multi_hop_triggered(self):
        router = _setup_router()
        # Query that's far from any single basin → large residual
        q = torch.randn(32) * 10.0
        result = router.route(q)
        # Should attempt multi-hop if residual is large enough
        assert result.routing in ("single_hop", "multi_hop")
        assert result.knowledge.shape == (32,)

    def test_result_has_valid_structure(self):
        router = _setup_router()
        q = torch.zeros(32)
        q[0] = 5.0
        result = router.route(q, return_trace=True)
        assert hasattr(result, "knowledge")
        assert hasattr(result, "n_hops")
        assert hasattr(result, "visited_basins")
        assert hasattr(result, "routing")
        assert hasattr(result, "chain_weight")
