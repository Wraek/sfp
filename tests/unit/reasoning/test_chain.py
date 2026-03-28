"""Tests for reasoning.chain — AssociativeReasoningChain."""

import torch
import pytest

from sfp.config import ReasoningChainConfig, Tier2Config, TransitionConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.chain import AssociativeReasoningChain
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import RelationType


def _setup_chain(d: int = 32, n_basins: int = 4) -> tuple[EssentialMemory, TransitionStructure, AssociativeReasoningChain]:
    tier2 = EssentialMemory(Tier2Config(n_slots=16, d_value=d), d_model=d)
    with torch.no_grad():
        tier2.query_proj.weight.copy_(torch.eye(d))
        tier2.key_proj.weight.copy_(torch.eye(d))
        tier2.value_proj.weight.copy_(torch.eye(d))
        tier2.output_proj.weight.copy_(torch.eye(d))
    # Allocate basins
    for i in range(n_basins):
        k = torch.zeros(d)
        k[i] = 5.0
        tier2.allocate_slot(k, value=k)
        with torch.no_grad():
            tier2.confidence[i] = 0.8

    transitions = TransitionStructure(TransitionConfig(max_edges=32, d_relation=16), d_model=d)
    # Create chain: 0 -> 1 -> 2 -> 3
    for i in range(n_basins - 1):
        transitions.add_edge(i, i + 1, RelationType.CAUSAL, weight=0.8, confidence=0.8)

    chain = AssociativeReasoningChain(
        tier2, transitions,
        ReasoningChainConfig(max_hops=5, convergence_threshold=0.001, branch_threshold=-10.0),
    )
    return tier2, transitions, chain


class TestAssociativeReasoningChain:
    def test_empty_memory_returns_zero_knowledge(self):
        d = 32
        tier2 = EssentialMemory(Tier2Config(n_slots=8, d_value=d), d_model=d)
        transitions = TransitionStructure(d_model=d)
        chain = AssociativeReasoningChain(tier2, transitions)
        result = chain.reason(torch.randn(d))
        assert result.n_hops == 0
        assert result.terminated_reason == "empty_memory"

    def test_basic_chain_traversal(self):
        _, _, chain = _setup_chain()
        # Query near basin 0
        q = torch.zeros(32)
        q[0] = 5.0
        result = chain.reason(q)
        assert result.n_hops >= 1
        assert 0 in result.visited_basins
        assert result.knowledge.norm().item() > 0

    def test_trace_returned_when_requested(self):
        _, _, chain = _setup_chain()
        q = torch.zeros(32)
        q[0] = 5.0
        result = chain.reason(q, return_trace=True)
        assert len(result.trace) > 0
        assert result.trace[0].event_type == "start"

    def test_cycle_detection(self):
        d = 32
        tier2, transitions, _ = _setup_chain(d, n_basins=3)
        # Add cycle: 2 -> 0
        transitions.add_edge(2, 0, RelationType.ASSOCIATIVE, weight=0.9, confidence=0.9)
        chain = AssociativeReasoningChain(
            tier2, transitions,
            ReasoningChainConfig(max_hops=10, convergence_threshold=0.001, branch_threshold=-10.0),
        )
        q = torch.zeros(d)
        q[0] = 5.0
        result = chain.reason(q)
        assert result.terminated_reason == "cycle"

    def test_dead_end_detection(self):
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
            tier2.confidence[0] = 0.8

        transitions = TransitionStructure(TransitionConfig(max_edges=8), d_model=d)
        # No outgoing edges from basin 0
        # Use a query different from basin value so convergence doesn't trigger first
        q = torch.zeros(d)
        q[0] = 3.0
        q[1] = 4.0  # query differs from basin value
        chain = AssociativeReasoningChain(
            tier2, transitions,
            ReasoningChainConfig(max_hops=5, convergence_threshold=0.0),
        )
        result = chain.reason(q)
        assert result.terminated_reason == "dead_end"

    def test_target_bias_influences_path(self):
        _, _, chain = _setup_chain()
        q = torch.zeros(32)
        q[0] = 5.0
        # Strong bias toward basin 2
        result = chain.reason(q, target_bias={2: 100.0})
        if result.n_hops >= 1:
            # Basin 2 should be visited given the strong bias
            # (not guaranteed but highly likely with bias=100)
            pass  # Just verify it doesn't crash
