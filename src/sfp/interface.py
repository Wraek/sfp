"""SFPInterface — thread-safe external facade for the SFP system.

Provides a unified, thread-safe API for external integrations (bridges,
adapters, GUIs) to interact with a HierarchicalMemoryProcessor.  Wraps
core processing, valence injection, goal management, episode storage,
and health reporting behind a single class with ``threading.RLock``
serialization.

Example::

    import sfp

    processor = sfp.create_field("small", hierarchical=True, valence=True, goals=True)
    interface = sfp.SFPInterface(processor)

    metric = interface.process(obs_tensor, modality="minecraft")
    result = interface.query(obs_tensor)
    interface.inject_valence(obs_tensor, reward=0.5, user_feedback=0.8)
"""

from __future__ import annotations

import threading
import time

import torch

from sfp.memory.integrity import compute_episode_hash, compute_weight_hash
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import (
    ConsolidationMode,
    Episode,
    Goal,
    ReasoningResult,
    SurpriseMetric,
    ValenceSignal,
)
from sfp.utils.logging import get_logger

logger = get_logger("interface")


class SFPInterface:
    """Thread-safe external interface to the SFP system.

    Wraps a :class:`HierarchicalMemoryProcessor` with:

    * **Thread-safe access** via ``threading.RLock`` — safe to call from
      GUI threads, bridge threads, and background workers concurrently.
    * **High-level convenience methods** for valence injection, goal
      creation, and episode storage that hide internal complexity.
    * **Clean return types** suitable for external callers.

    Args:
        processor: A fully initialized ``HierarchicalMemoryProcessor``
            (typically created via ``sfp.create_field(..., hierarchical=True)``).
    """

    def __init__(self, processor: HierarchicalMemoryProcessor) -> None:
        if not isinstance(processor, HierarchicalMemoryProcessor):
            raise TypeError(
                f"SFPInterface requires a HierarchicalMemoryProcessor, "
                f"got {type(processor).__name__}"
            )
        self._processor = processor
        self._lock = threading.RLock()
        logger.info("SFPInterface initialized")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        x: torch.Tensor,
        modality: str = "tensor",
        target: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> SurpriseMetric:
        """Submit an observation through the full hierarchical pipeline.

        Args:
            x: Input tensor — shape depends on pipeline stage:
                ``(d_model,)``, ``(B, d_model)``, or ``(B, N, d_input)``.
            modality: Input modality identifier (e.g. ``"tensor"``,
                ``"environment"``, ``"text"``).
            target: Optional target for supervised updates.
            metadata: Optional dict of bridge-provided metadata.
                Supported keys:

                * ``"entity_positions"`` — ``(N, 3)`` tensor of entity
                  positions for scene graph spatial reasoning.
                * ``"entity_embeddings"`` — ``(N, d_model)`` tensor of
                  entity embeddings for scene graph.

        Returns:
            :class:`SurpriseMetric` from the Tier 0 update.
        """
        with self._lock:
            return self._processor.process(
                x, modality=modality, target=target, metadata=metadata,
            )

    def query(
        self,
        x: torch.Tensor,
        return_trace: bool = False,
    ) -> ReasoningResult:
        """Query the memory system without updating weights.

        Args:
            x: Query tensor — same shape rules as :meth:`process`.
            return_trace: Whether to include per-hop reasoning trace.

        Returns:
            :class:`ReasoningResult` with accumulated knowledge.
        """
        with self._lock:
            return self._processor.query(x, return_trace=return_trace)

    def consolidate(
        self,
        force_mode: ConsolidationMode | None = None,
    ) -> None:
        """Trigger memory consolidation.

        Args:
            force_mode: If specified, run this consolidation mode
                regardless of schedule.  Pass ``None`` to let the
                engine decide based on step count.
        """
        with self._lock:
            self._processor.consolidate(force_mode=force_mode)

    # ------------------------------------------------------------------
    # Valence injection
    # ------------------------------------------------------------------

    def inject_valence(
        self,
        embedding: torch.Tensor,
        *,
        reward: float = 0.0,
        user_feedback: float = 0.0,
        goal_alignment: float = 0.0,
        prediction_satisfaction: float = 0.0,
    ) -> ValenceSignal | None:
        """Inject an external valence signal into the affect system.

        Delegates to :meth:`ValenceSystem.compute_valence` and updates
        the processor's internal mood state so subsequent processing
        steps reflect the injected signal.

        Args:
            embedding: ``(d_model,)`` embedding to associate valence with.
            reward: Raw RL reward value (will be normalized internally).
            user_feedback: Explicit human feedback ``[-1, 1]``.
            goal_alignment: Goal alignment score ``[-1, 1]``.
            prediction_satisfaction: Prediction confirmation ``[-1, 1]``.

        Returns:
            :class:`ValenceSignal` if the valence module is enabled,
            ``None`` otherwise.
        """
        with self._lock:
            if self._processor.valence is None:
                return None
            signal = self._processor.valence.compute_valence(
                embedding.to(self._processor._device),
                reward=reward,
                user_feedback=user_feedback,
                goal_alignment=goal_alignment,
                prediction_satisfaction=prediction_satisfaction,
            )
            self._processor._last_valence = signal
            return signal

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def create_goal(
        self,
        instruction_embedding: torch.Tensor,
        *,
        importance: float = 0.5,
        urgency: float = 0.5,
        deadline: float | None = None,
        ttl: float | None = None,
        description: str = "",
    ) -> Goal | None:
        """Create a goal from an instruction embedding.

        Args:
            instruction_embedding: ``(d_model,)`` embedding of the goal.
            importance: Static importance ``[0, 1]``.
            urgency: Initial urgency ``[0, 1]``.
            deadline: Absolute monotonic-time deadline, or ``None``.
            ttl: Time-to-live in seconds (default from config).

        Returns:
            The created :class:`Goal` if the goals module is enabled,
            ``None`` otherwise.
        """
        with self._lock:
            if self._processor.goals is None:
                return None
            return self._processor.goals.create_goal(
                instruction_embedding.to(self._processor._device),
                importance=importance,
                urgency=urgency,
                deadline=deadline,
                ttl=ttl,
                description=description,
            )

    def remove_goal(self, goal_id: int) -> bool:
        """Remove a goal by ID.

        Returns:
            ``True`` if removed, ``False`` if goals are disabled or
            the goal was not found.
        """
        with self._lock:
            if self._processor.goals is None:
                return False
            return self._processor.goals.remove_goal(goal_id)

    def pause_goal(self, goal_id: int) -> bool:
        """Pause an active goal.

        Returns:
            ``True`` if paused, ``False`` if goals are disabled.
        """
        with self._lock:
            if self._processor.goals is None:
                return False
            self._processor.goals.pause_goal(goal_id)
            return True

    def resume_goal(self, goal_id: int) -> bool:
        """Resume a paused goal.

        Returns:
            ``True`` if resumed, ``False`` if goals are disabled.
        """
        with self._lock:
            if self._processor.goals is None:
                return False
            self._processor.goals.resume_goal(goal_id)
            return True

    def list_goals(self) -> list[dict]:
        """Return all goals as serializable dicts.

        Each dict contains: ``id``, ``status``, ``priority``, ``progress``,
        ``importance``, ``urgency``.

        Returns:
            List of goal summary dicts, or empty list if goals disabled.
        """
        with self._lock:
            if self._processor.goals is None:
                return []
            return [
                {
                    "id": g.id,
                    "description": g.description,
                    "status": g.status.name,
                    "priority": g.priority,
                    "progress": g.progress,
                    "importance": g.importance,
                    "urgency": g.urgency,
                }
                for g in self._processor.goals.all_goals
            ]

    # ------------------------------------------------------------------
    # Episode storage
    # ------------------------------------------------------------------

    def store_episode(
        self,
        input_embedding: torch.Tensor,
        *,
        modality: str = "external",
        surprise: float = 1.0,
        valence: float = 0.0,
    ) -> bool:
        """Store an externally-created episode in Tier 1.

        Handles all internal complexity: provenance hashing, weight
        summaries, logit snapshots, integrity hashing, and basin
        assignment.  The caller only provides the embedding and metadata.

        Args:
            input_embedding: ``(d_model,)`` embedding to store.
            modality: Modality tag (e.g. ``"minecraft"``, ``"demo"``).
            surprise: Surprise score — must exceed Tier 1's surprise
                threshold for admission.
            valence: Valence annotation for the episode.

        Returns:
            ``True`` if episode was admitted, ``False`` if rejected by
            admission gates (surprise threshold, deduplication, or
            integrity check).
        """
        with self._lock:
            proc = self._processor
            field = proc.tier0.field

            emb = input_embedding.to(proc._device)
            if emb.dim() > 1:
                emb = emb.squeeze(0)

            # Provenance and integrity hashing
            weight_hash = compute_weight_hash(field)
            weight_summary = field.get_weight_summary()
            with torch.no_grad():
                logit_snapshot = field(emb).detach()
            integrity_hash = compute_episode_hash(
                emb, logit_snapshot, weight_hash,
            )

            # Basin assignment via reasoning router
            with torch.no_grad():
                result = proc._router.route(emb)
            basin_id = (
                result.visited_basins[0] if result.visited_basins else -1
            )
            basin_distance = 0.0
            if basin_id >= 0 and proc.tier2.active_mask[basin_id]:
                basin_distance = (
                    (emb - proc.tier2.keys[basin_id]).norm().item()
                )

            episode = Episode(
                id=proc.tier1.allocate_id(),
                timestamp=time.monotonic(),
                modality=modality,
                provenance_hash=weight_hash[:16],
                input_embedding=emb.detach().cpu(),
                working_memory_state=weight_summary.detach().cpu(),
                logit_snapshot=logit_snapshot.detach().cpu(),
                surprise_at_storage=surprise,
                attractor_basin_id=basin_id,
                attractor_distance=basin_distance,
                preceding_episode_id=proc.tier1.last_episode_id,
                following_episode_id=None,
                integrity_hash=integrity_hash,
                weight_hash_at_storage=weight_hash,
                valence=valence,
            )
            return proc.tier1.maybe_store(episode)

    # ------------------------------------------------------------------
    # Status and health
    # ------------------------------------------------------------------

    def health_report(self) -> dict:
        """Generate a comprehensive health report across all tiers and modules.

        Returns:
            Nested dict with per-tier and per-module health metrics.
        """
        with self._lock:
            return self._processor.health_report()

    def memory_footprint(self) -> dict[str, int]:
        """Estimate VRAM usage per component in bytes.

        Returns:
            Dict mapping component name to byte count, plus ``"total"``.
        """
        with self._lock:
            return self._processor.memory_footprint()

    def status(self) -> dict:
        """Quick status summary for overlay rendering or monitoring.

        Lighter than :meth:`health_report` — returns only the most
        frequently needed metrics.

        Returns:
            Dict with keys: ``step_count``, ``tier1_episodes``,
            ``tier2_basins``, ``tier3_axioms``, and optionally
            ``active_goals``, ``mood``, ``valence``.
        """
        with self._lock:
            proc = self._processor
            result: dict = {
                "step_count": proc._step_count,
                "tier1_episodes": proc.tier1.total_count,
                "tier2_basins": proc.tier2.n_active,
                "tier3_axioms": proc.tier3.n_active,
            }
            if proc.goals is not None:
                result["active_goals"] = len(proc.goals.active_goals)
            if proc.valence is not None and proc._last_valence is not None:
                result["mood"] = proc._last_valence.composite_mood
                result["valence"] = proc._last_valence.scalar_valence
            if proc._world_model is not None and proc._last_wm_state is not None:
                result["world_model_surprise"] = proc._world_model.compute_enhanced_surprise(
                    proc._last_wm_state,
                )
            if proc._salience_gate is not None:
                result["salience_stats"] = proc._salience_stats
            return result

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def retrieve_by_location(
        self,
        position: tuple[float, float, float],
        *,
        radius: float = 50.0,
        max_results: int = 10,
        embedding: torch.Tensor | None = None,
        spatial_weight: float = 0.7,
    ) -> list[dict]:
        """Retrieve episodes stored near a spatial position.

        "I was here before" — returns episodes near the query position,
        optionally blended with semantic similarity.

        Args:
            position: ``(x, y, z)`` world position to search near.
            radius: Maximum Euclidean distance to consider.
            max_results: Maximum episodes to return.
            embedding: Optional ``(d_model,)`` query for semantic blending.
            spatial_weight: Balance between spatial (1.0) and semantic (0.0).

        Returns:
            List of dicts with keys: ``id``, ``score``, ``distance``,
            ``position``, ``modality``, ``basin_id``, ``surprise``.
        """
        with self._lock:
            results = self._processor.tier1.retrieve_by_location(
                position,
                radius=radius,
                max_results=max_results,
                embedding=embedding,
                spatial_weight=spatial_weight,
            )
            return [
                {
                    "id": ep.id,
                    "score": score,
                    "distance": (
                        sum((a - b) ** 2 for a, b in zip(position, ep.spatial_position)) ** 0.5
                        if ep.spatial_position else 0.0
                    ),
                    "position": ep.spatial_position,
                    "modality": ep.modality,
                    "basin_id": ep.attractor_basin_id,
                    "surprise": ep.surprise_at_storage,
                }
                for ep, score in results
            ]

    def reset_session(self) -> None:
        """Reset Tier 0 working memory (session volatility)."""
        with self._lock:
            self._processor.reset_session()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def processor(self) -> HierarchicalMemoryProcessor:
        """Direct access to the underlying processor (for advanced use).

        .. warning::
            Bypasses thread-safety. Use with caution.
        """
        return self._processor

    @property
    def lock(self) -> threading.RLock:
        """The interface's RLock — for callers that need to hold the lock
        across multiple operations."""
        return self._lock

    @property
    def is_valence_enabled(self) -> bool:
        """Whether the valence / affect module is active."""
        return self._processor.valence is not None

    @property
    def is_goals_enabled(self) -> bool:
        """Whether the goal persistence module is active."""
        return self._processor.goals is not None

    @property
    def step_count(self) -> int:
        """Number of process() steps completed."""
        return self._processor._step_count

    @property
    def d_model(self) -> int:
        """Embedding dimensionality — callers need this to produce
        correctly shaped input tensors."""
        return self._processor.tier0.field.config.dim
