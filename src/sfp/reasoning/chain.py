"""Associative Reasoning Chain — multi-hop traversal through the Tier 2 concept graph.

Implements the reasoning chain from sfp-reasoning-chains.md. Starting from a query,
iteratively hops through connected concept basins, accumulating knowledge with context
decay, and terminates when the query is resolved or no viable path remains.
"""

from __future__ import annotations

import torch

from sfp.config import ReasoningChainConfig
from sfp.memory.essential import EssentialMemory
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import ChainTrace, ReasoningResult
from sfp.utils.logging import get_logger

logger = get_logger("reasoning.chain")


class AssociativeReasoningChain:
    """Multi-hop reasoning through the Tier 2 concept graph.

    At each hop:
      1. Compute what part of the query remains unanswered (query residual).
      2. Score outgoing transitions by relevance to the residual.
      3. Follow the top-k most promising paths (branching).
      4. Accumulate knowledge from each visited basin with context decay.
      5. Terminate when query is resolved, dead-end, cycle, convergence, or max hops.

    Args:
        tier2: The EssentialMemory instance (provides basin retrieval).
        transitions: The TransitionStructure (provides the concept graph).
        config: ReasoningChainConfig for traversal parameters.
    """

    def __init__(
        self,
        tier2: EssentialMemory,
        transitions: TransitionStructure,
        config: ReasoningChainConfig | None = None,
    ) -> None:
        self._tier2 = tier2
        self._transitions = transitions
        self._config = config or ReasoningChainConfig()

    def reason(
        self,
        query: torch.Tensor,
        return_trace: bool = False,
        target_bias: dict[int, float] | None = None,
    ) -> ReasoningResult:
        """Execute a multi-hop reasoning chain starting from a query.

        Args:
            query: (d_model,) query vector.
            return_trace: If True, include per-hop trace in the result.
            target_bias: Optional mapping of basin_id → additive score bias.
                Applied after transition scoring to steer the chain toward
                goal-relevant or valence-relevant basins.

        Returns:
            ReasoningResult with accumulated knowledge, hop count, and trace.
        """
        cfg = self._config
        device = query.device
        d_model = query.shape[-1]

        # Initial retrieval from Tier 2
        if self._tier2.n_active == 0:
            return ReasoningResult(
                knowledge=torch.zeros(d_model, device=device),
                n_hops=0,
                visited_basins=[],
                terminated_reason="empty_memory",
                routing="multi_hop",
                chain_weight=0.0,
                trace=[],
            )

        with torch.no_grad():
            initial_output, initial_basin, _ = self._tier2.retrieve(query)

        current_basin = initial_basin.item() if initial_basin.dim() == 0 else initial_basin[0].item()
        if current_basin < 0:
            return ReasoningResult(
                knowledge=torch.zeros(d_model, device=device),
                n_hops=0,
                visited_basins=[],
                terminated_reason="no_basin",
                routing="multi_hop",
                chain_weight=0.0,
                trace=[],
            )

        # Initialize chain state
        knowledge = initial_output if initial_output.dim() == 1 else initial_output[0]
        query_residual = query - knowledge
        visited: list[int] = [current_basin]
        trace: list[ChainTrace] = []
        terminated_reason = "max_hops"

        if return_trace:
            trace.append(
                ChainTrace(
                    hop=0,
                    basin_id=current_basin,
                    event_type="start",
                    confidence=self._tier2.confidence[current_basin].item(),
                    knowledge_norm=knowledge.norm().item(),
                    query_residual_norm=query_residual.norm().item(),
                )
            )

        for hop in range(1, cfg.max_hops + 1):
            # Check convergence: if query residual is small, we're done
            residual_norm = query_residual.norm().item()
            if residual_norm < cfg.convergence_threshold:
                terminated_reason = "convergence"
                break

            # Score outgoing transitions from current basin
            scores, target_ids = self._transitions.compute_transition_scores(
                current_basin, query_residual
            )

            # Apply optional goal/valence reasoning bias
            if target_bias:
                for i in range(target_ids.shape[0]):
                    tid = target_ids[i].item()
                    if tid in target_bias:
                        scores[i] = scores[i] + target_bias[tid]

            if target_ids.shape[0] == 0:
                terminated_reason = "dead_end"
                if return_trace:
                    trace.append(
                        ChainTrace(
                            hop=hop,
                            basin_id=current_basin,
                            event_type="dead_end",
                            query_residual_norm=residual_norm,
                        )
                    )
                break

            # Filter by branch threshold
            viable_mask = scores > cfg.branch_threshold
            if not viable_mask.any():
                # Try softer threshold: take the best edge if its score is positive
                if scores.max().item() > 0:
                    viable_mask = scores == scores.max()
                else:
                    terminated_reason = "dead_end"
                    if return_trace:
                        trace.append(
                            ChainTrace(
                                hop=hop,
                                basin_id=current_basin,
                                event_type="dead_end",
                                query_residual_norm=residual_norm,
                            )
                        )
                    break

            viable_scores = scores[viable_mask]
            viable_targets = target_ids[viable_mask]

            # Take top-k branches
            n_branches = min(cfg.max_branches, viable_targets.shape[0])
            if n_branches < viable_targets.shape[0]:
                topk_idx = viable_scores.topk(n_branches).indices
                viable_scores = viable_scores[topk_idx]
                viable_targets = viable_targets[topk_idx]

            # Follow the best branch (highest score)
            best_idx = viable_scores.argmax()
            next_basin = viable_targets[best_idx].item()

            # Cycle detection
            if next_basin in visited:
                terminated_reason = "cycle"
                if return_trace:
                    trace.append(
                        ChainTrace(
                            hop=hop,
                            basin_id=next_basin,
                            event_type="cycle",
                            score=viable_scores[best_idx].item(),
                            n_branches=n_branches,
                            query_residual_norm=residual_norm,
                        )
                    )
                break

            # Retrieve knowledge from the next basin
            with torch.no_grad():
                basin_key = self._tier2.keys[next_basin]
                hop_output, _, _ = self._tier2.retrieve(basin_key)

            if hop_output.dim() > 1:
                hop_output = hop_output[0]

            # Accumulate with context decay
            decay = cfg.context_decay ** hop
            knowledge = knowledge + decay * hop_output

            # Update query residual
            query_residual = cfg.query_retention * query - (1.0 - cfg.query_retention) * knowledge
            query_residual = query_residual / (query_residual.norm() + 1e-8) * residual_norm * cfg.context_decay

            visited.append(next_basin)
            current_basin = next_basin

            if return_trace:
                trace.append(
                    ChainTrace(
                        hop=hop,
                        basin_id=next_basin,
                        event_type="hop",
                        confidence=self._tier2.confidence[next_basin].item(),
                        score=viable_scores[best_idx].item(),
                        n_branches=n_branches,
                        knowledge_norm=knowledge.norm().item(),
                        query_residual_norm=query_residual.norm().item(),
                    )
                )

            # Check if query is resolved
            resolved_sim = torch.nn.functional.cosine_similarity(
                knowledge.unsqueeze(0), query.unsqueeze(0)
            ).item()
            if resolved_sim > 0.95:
                terminated_reason = "resolved"
                break

        return ReasoningResult(
            knowledge=knowledge,
            n_hops=len(visited) - 1,
            visited_basins=visited,
            terminated_reason=terminated_reason,
            routing="multi_hop",
            chain_weight=1.0,
            trace=trace if return_trace else [],
        )
