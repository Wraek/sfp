"""Reasoning Router — decides single-hop vs multi-hop retrieval.

Examines the query residual after single-hop Tier 2 retrieval, the connectivity
of the matched basin, and the attention entropy to determine whether multi-hop
reasoning is warranted.
"""

from __future__ import annotations

import math

import torch

from sfp.memory.essential import EssentialMemory
from sfp.reasoning.chain import AssociativeReasoningChain
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import ReasoningResult
from sfp.utils.logging import get_logger

logger = get_logger("reasoning.router")


class ReasoningRouter:
    """Decides whether to use single-hop or multi-hop retrieval for a given query.

    Decision criteria:
      1. Query residual size after single-hop — large residuals suggest multi-hop.
      2. Outgoing edge count from matched basin — well-connected basins enable traversal.
      3. Attention entropy — distributed attention suggests ambiguity needing resolution.

    When multi-hop is chosen, runs the AssociativeReasoningChain and blends the
    result with the single-hop output based on relative confidence.

    Args:
        tier2: The EssentialMemory instance.
        transitions: The TransitionStructure.
        chain: The AssociativeReasoningChain.
        residual_threshold: Min query residual norm to trigger multi-hop.
        entropy_threshold: Min attention entropy to trigger multi-hop.
        min_outgoing_edges: Min outgoing edges from the matched basin.
    """

    def __init__(
        self,
        tier2: EssentialMemory,
        transitions: TransitionStructure,
        chain: AssociativeReasoningChain,
        residual_threshold: float = 0.5,
        entropy_threshold: float = 1.0,
        min_outgoing_edges: int = 1,
    ) -> None:
        self._tier2 = tier2
        self._transitions = transitions
        self._chain = chain
        self._residual_threshold = residual_threshold
        self._entropy_threshold = entropy_threshold
        self._min_outgoing = min_outgoing_edges

    def route(
        self,
        query: torch.Tensor,
        return_trace: bool = False,
        target_bias: dict[int, float] | None = None,
    ) -> ReasoningResult:
        """Route a query to single-hop or multi-hop retrieval.

        Args:
            query: (d_model,) query vector.
            return_trace: Whether to include trace in multi-hop results.
            target_bias: Optional mapping of basin_id → additive score bias,
                passed through to the reasoning chain for goal/valence steering.

        Returns:
            ReasoningResult with the routing decision and accumulated knowledge.
        """
        d_model = query.shape[-1]

        if self._tier2.n_active == 0:
            return ReasoningResult(
                knowledge=torch.zeros(d_model, device=query.device),
                n_hops=0,
                visited_basins=[],
                terminated_reason="empty_memory",
                routing="single_hop",
                chain_weight=0.0,
            )

        # Single-hop retrieval
        with torch.no_grad():
            single_output, basin_id, attn = self._tier2.retrieve(query)

        bid = basin_id.item() if basin_id.dim() == 0 else basin_id[0].item()
        if bid < 0:
            return ReasoningResult(
                knowledge=torch.zeros(d_model, device=query.device),
                n_hops=0,
                visited_basins=[],
                terminated_reason="no_basin",
                routing="single_hop",
                chain_weight=0.0,
            )

        # Compute decision criteria
        query_residual = query - (single_output if single_output.dim() == 1 else single_output[0])
        residual_norm = query_residual.norm().item()

        # Attention entropy
        attn_vec = attn if attn.dim() == 1 else attn[0]
        if attn_vec.shape[0] > 0:
            # Shannon entropy: -sum(p * log(p))
            attn_clamped = attn_vec.clamp(min=1e-10)
            entropy = -(attn_clamped * attn_clamped.log()).sum().item()
        else:
            entropy = 0.0

        # Outgoing edge count
        targets, _, _ = self._transitions.get_outgoing(bid)
        n_outgoing = targets.shape[0]

        # Decision: multi-hop if any criterion is met
        needs_multihop = (
            residual_norm > self._residual_threshold
            and n_outgoing >= self._min_outgoing
        ) or (
            entropy > self._entropy_threshold
            and n_outgoing >= self._min_outgoing
        )

        if not needs_multihop:
            # Single-hop is sufficient
            return ReasoningResult(
                knowledge=single_output if single_output.dim() == 1 else single_output[0],
                n_hops=0,
                visited_basins=[bid],
                terminated_reason="single_hop_sufficient",
                routing="single_hop",
                chain_weight=0.0,
            )

        # Multi-hop reasoning
        chain_result = self._chain.reason(
            query, return_trace=return_trace, target_bias=target_bias,
        )

        if chain_result.n_hops == 0:
            # Chain didn't go anywhere — fall back to single-hop
            return ReasoningResult(
                knowledge=single_output if single_output.dim() == 1 else single_output[0],
                n_hops=0,
                visited_basins=[bid],
                terminated_reason="chain_fallback",
                routing="single_hop",
                chain_weight=0.0,
            )

        # Blend single-hop and multi-hop based on relative knowledge norms
        single_norm = single_output.norm().item() if single_output.dim() == 1 else single_output[0].norm().item()
        chain_norm = chain_result.knowledge.norm().item()
        total_norm = single_norm + chain_norm

        if total_norm < 1e-8:
            chain_weight = 0.5
        else:
            chain_weight = chain_norm / total_norm

        single_vec = single_output if single_output.dim() == 1 else single_output[0]
        blended = (1.0 - chain_weight) * single_vec + chain_weight * chain_result.knowledge

        return ReasoningResult(
            knowledge=blended,
            n_hops=chain_result.n_hops,
            visited_basins=chain_result.visited_basins,
            terminated_reason=chain_result.terminated_reason,
            routing="multi_hop",
            chain_weight=chain_weight,
            trace=chain_result.trace,
        )
