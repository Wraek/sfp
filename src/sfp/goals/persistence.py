"""Goal Persistence — 32-slot hierarchical goal register.

Maintains explicit goals across inference cycles, decomposes complex goals
into subgoals, computes dynamic priorities, tracks progress via cosine
similarity to satisfaction embeddings, and provides reasoning bias and
salience modulation context vectors.
"""

from __future__ import annotations

import hashlib
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import GoalPersistenceConfig
from sfp.types import Goal, GoalStatus
from sfp.utils.logging import get_logger
from sfp.utils.math import cosine_similarity_matrix

logger = get_logger("goals.persistence")


class GoalRegister(nn.Module):
    """32-slot goal register with hierarchy, decomposition, and priority.

    Manages a fixed-capacity set of active goals, each with an embedding,
    satisfaction embedding (what "done" looks like), priority score, progress
    tracking, temporal bounds, and hierarchical parent/child relationships.

    Args:
        config: GoalPersistenceConfig with capacity and threshold parameters.
        d_model: Embedding dimensionality (must match field/backbone dim).
    """

    def __init__(
        self,
        config: GoalPersistenceConfig | None = None,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        cfg = config or GoalPersistenceConfig()
        self._config = cfg
        self._d_model = d_model

        # --- Networks ---
        # Goal encoder: instruction embedding -> goal embedding
        self.goal_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, cfg.d_goal),
        )

        # Satisfaction encoder: instruction embedding -> completion target
        self.satisfaction_encoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, cfg.d_satisfaction),
        )

        # Decomposition network: goal -> up to max_subgoals subgoal embeddings
        self.decomposition_net = nn.Sequential(
            nn.Linear(cfg.d_goal, cfg.d_goal * 2),
            nn.GELU(),
            nn.Linear(cfg.d_goal * 2, cfg.d_goal * cfg.max_subgoals),
        )

        # Priority MLP: [goal_emb, progress, urgency+pressure, importance, age] -> priority
        self.priority_mlp = nn.Sequential(
            nn.Linear(cfg.d_goal + 4, cfg.priority_mlp_hidden),
            nn.GELU(),
            nn.Linear(cfg.priority_mlp_hidden, 1),
            nn.Sigmoid(),
        )

        # --- Goal storage ---
        self._goals: list[Goal] = []
        self._next_goal_id: int = 0

        logger.info(
            "GoalRegister initialized: max_goals=%d, d_goal=%d",
            cfg.max_goals, cfg.d_goal,
        )

    # ------------------------------------------------------------------
    # Goal creation and decomposition
    # ------------------------------------------------------------------

    def create_goal(
        self,
        instruction_embedding: torch.Tensor,
        importance: float = 0.5,
        urgency: float = 0.5,
        deadline: float | None = None,
        ttl: float | None = None,
        parent_id: int | None = None,
        description: str = "",
    ) -> Goal:
        """Create a new goal from an instruction embedding.

        Args:
            instruction_embedding: (d_model,) embedding of the goal instruction.
            importance: Static importance [0, 1].
            urgency: Initial urgency [0, 1].
            deadline: Absolute monotonic time deadline, or None.
            ttl: Time-to-live in seconds (default from config).
            parent_id: Parent goal ID for hierarchical decomposition.

        Returns:
            The newly created Goal.
        """
        device = instruction_embedding.device

        with torch.no_grad():
            goal_emb = self.goal_encoder(instruction_embedding).detach()
            # Use goal embedding as satisfaction target — states similar to
            # the goal representation indicate progress toward completion.
            # (The satisfaction_encoder has random init and no training signal,
            # so its output is arbitrary and blocks completion detection.)
            satisfaction_emb = goal_emb.clone()

        # Compute description hash for dedup/integrity
        desc_hash = hashlib.sha256(
            goal_emb.cpu().numpy().tobytes()
        ).digest()[:16]

        now = time.monotonic()
        goal = Goal(
            id=self._next_goal_id,
            embedding=goal_emb,
            satisfaction_embedding=satisfaction_emb,
            description=description,
            description_hash=desc_hash,
            priority=0.5,
            urgency=urgency,
            importance=importance,
            progress=0.0,
            status=GoalStatus.ACTIVE,
            parent_id=parent_id,
            child_ids=[],
            dependency_ids=[],
            created_at=now,
            deadline=deadline,
            ttl=ttl or self._config.ttl_default,
            last_progress_update=now,
            progress_history=[],
        )
        self._next_goal_id += 1

        # Link to parent
        if parent_id is not None:
            parent = self._find_goal(parent_id)
            if parent is not None:
                parent.child_ids.append(goal.id)

        # Evict if at capacity
        if len(self._goals) >= self._config.max_goals:
            self._evict_one()

        self._goals.append(goal)
        logger.debug("Created goal %d (parent=%s)", goal.id, parent_id)
        return goal

    def decompose_goal(self, goal_id: int) -> list[Goal]:
        """Decompose a goal into subgoals via the decomposition network.

        Args:
            goal_id: ID of the goal to decompose.

        Returns:
            List of created subgoal Goals.
        """
        goal = self._find_goal(goal_id)
        if goal is None:
            return []

        cfg = self._config
        with torch.no_grad():
            raw = self.decomposition_net(goal.embedding)
            subgoal_embs = raw.view(cfg.max_subgoals, cfg.d_goal)

        subgoals: list[Goal] = []
        for i in range(cfg.max_subgoals):
            emb = subgoal_embs[i]
            # Only create subgoals with significant norm
            if emb.norm().item() < 0.1:
                continue

            sub = self.create_goal(
                emb,
                importance=goal.importance * 0.8,
                urgency=goal.urgency,
                deadline=goal.deadline,
                ttl=goal.ttl,
                parent_id=goal_id,
            )
            subgoals.append(sub)

        # Set sequential dependencies between subgoals
        for i in range(1, len(subgoals)):
            subgoals[i].dependency_ids.append(subgoals[i - 1].id)
            subgoals[i].status = GoalStatus.BLOCKED

        logger.debug(
            "Decomposed goal %d into %d subgoals", goal_id, len(subgoals)
        )
        return subgoals

    # ------------------------------------------------------------------
    # Progress tracking
    # ------------------------------------------------------------------

    def update_progress(
        self, goal_id: int, current_state: torch.Tensor
    ) -> float:
        """Update goal progress based on cosine similarity to satisfaction.

        Args:
            goal_id: Goal to update.
            current_state: (d_model,) current embedding from backbone.

        Returns:
            Updated progress value [0, 1].
        """
        goal = self._find_goal(goal_id)
        if goal is None or goal.status != GoalStatus.ACTIVE:
            return 0.0

        # Cosine similarity to satisfaction embedding
        sat = goal.satisfaction_embedding
        if sat.device != current_state.device:
            sat = sat.to(current_state.device)
        raw_progress = F.cosine_similarity(
            current_state.unsqueeze(0), sat.unsqueeze(0)
        ).item()
        raw_progress = max(0.0, raw_progress)

        # EMA smoothing
        decay = self._config.progress_ema_decay
        goal.progress = decay * goal.progress + (1 - decay) * raw_progress
        goal.last_progress_update = time.monotonic()
        goal.progress_history.append(goal.progress)

        # Trim progress history to avoid unbounded growth
        if len(goal.progress_history) > 1000:
            goal.progress_history = goal.progress_history[-500:]

        # Auto-complete check
        if goal.progress >= self._config.satisfaction_threshold:
            self._complete_goal(goal)

        return goal.progress

    def train_satisfaction_hindsight(
        self, goal_id: int, observation: torch.Tensor
    ) -> None:
        """E1: Train satisfaction encoder via hindsight when goal is progressing well.

        When a goal has high progress, the current observation is a good example
        of what "satisfaction" looks like. We train the satisfaction_encoder to
        map the goal embedding toward this observation, then update the goal's
        satisfaction_embedding.

        Args:
            goal_id: Goal to train hindsight for.
            observation: (d_model,) current observation embedding.
        """
        goal = self._find_goal(goal_id)
        if goal is None or goal.status != GoalStatus.ACTIVE:
            return

        # Single SGD step on satisfaction encoder
        target = observation.detach()
        pred = self.satisfaction_encoder(goal.embedding.detach())

        loss = F.mse_loss(pred, target)
        loss.backward()

        with torch.no_grad():
            for p in self.satisfaction_encoder.parameters():
                if p.grad is not None:
                    p.data -= 1e-3 * p.grad
                    p.grad.zero_()

        # Update the goal's satisfaction embedding
        with torch.no_grad():
            goal.satisfaction_embedding = self.satisfaction_encoder(
                goal.embedding.detach(),
            ).detach()

    # ------------------------------------------------------------------
    # Priority computation
    # ------------------------------------------------------------------

    def compute_priorities(self) -> None:
        """Recompute priority for all active goals."""
        now = time.monotonic()
        for goal in self._goals:
            if goal.status != GoalStatus.ACTIVE:
                continue

            time_pressure = self._compute_time_pressure(goal, now)
            age_hours = (now - goal.created_at) / 3600.0

            # Build feature vector: [embedding, progress, urgency+pressure, importance, age]
            features = torch.cat([
                goal.embedding,
                torch.tensor(
                    [goal.progress, goal.urgency + time_pressure,
                     goal.importance, min(age_hours, 100.0) / 100.0],
                    device=goal.embedding.device,
                ),
            ])

            with torch.no_grad():
                goal.priority = self.priority_mlp(features).item()

    def _compute_time_pressure(self, goal: Goal, now: float) -> float:
        """Compute time pressure [0, 1] based on deadline proximity."""
        if goal.deadline is None:
            return 0.0
        total_time = goal.deadline - goal.created_at
        if total_time <= 0:
            return 1.0
        elapsed_ratio = (now - goal.created_at) / total_time
        return max(0.0, min(1.0, elapsed_ratio))

    # ------------------------------------------------------------------
    # Context vectors for other modules
    # ------------------------------------------------------------------

    def get_goal_context(self) -> torch.Tensor:
        """Compute priority-weighted context vector from active goals.

        Returns:
            (d_goal,) context vector, or zeros if no active goals.
        """
        active = [g for g in self._goals if g.status == GoalStatus.ACTIVE]
        if not active:
            device = next(self.parameters()).device
            return torch.zeros(self._config.d_goal, device=device)

        total_priority = sum(g.priority for g in active) + 1e-8
        context = torch.zeros_like(active[0].embedding)
        for g in active:
            context = context + (g.priority / total_priority) * g.embedding
        return context

    def get_reasoning_bias(
        self,
        basin_keys: torch.Tensor,
        active_indices: torch.Tensor,
    ) -> dict[int, float]:
        """Compute per-basin additive reasoning bias from active goals.

        Basins similar to any active goal receive a positive additive bias
        to transition scores.

        Args:
            basin_keys: (n_slots, d_model) all basin keys.
            active_indices: (n_active,) indices of active basins.

        Returns:
            Dict mapping basin index to additive bias value.
        """
        active_goals = [g for g in self._goals if g.status == GoalStatus.ACTIVE]
        if not active_goals or active_indices.shape[0] == 0:
            return {}

        bias_val = self._config.reasoning_bias
        device = basin_keys.device
        active_keys = basin_keys[active_indices]  # (A, d)

        bias: dict[int, float] = {}
        for goal in active_goals:
            goal_emb = goal.embedding.to(device)
            sims = cosine_similarity_matrix(
                goal_emb.unsqueeze(0), active_keys
            )  # (1, A)
            for i, idx_tensor in enumerate(active_indices):
                idx = idx_tensor.item()
                if sims[0, i].item() > 0.3:
                    bias[idx] = bias.get(idx, 0.0) + bias_val

        return bias

    def get_salience_modulation(self) -> dict[str, float]:
        """Compute per-modality salience threshold adjustments.

        Active goals with strong modality affinity lower thresholds
        for relevant modalities.

        Returns:
            Dict mapping modality name to threshold adjustment (negative = lower).
        """
        active = [g for g in self._goals if g.status == GoalStatus.ACTIVE]
        if not active:
            return {}

        # For now, provide a uniform boost proportional to number of active goals
        # Specific per-modality tuning would require modality-tagged goals
        boost = min(
            self._config.salience_boost,
            self._config.salience_boost * len(active) / 3.0,
        )
        return {
            "text": -boost,
            "vision": -boost,
            "audio": -boost,
            "sensor": -boost,
            "structured": -boost,
        }

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def check_deadlines(self) -> list[tuple[int, str]]:
        """Check for expired and approaching-deadline goals.

        Returns:
            List of (goal_id, warning_type) pairs.
        """
        warnings: list[tuple[int, str]] = []
        now = time.monotonic()

        for goal in self._goals:
            if goal.status not in (GoalStatus.ACTIVE, GoalStatus.PAUSED, GoalStatus.BLOCKED):
                continue

            # TTL expiry
            if now - goal.created_at > goal.ttl:
                goal.status = GoalStatus.EXPIRED
                warnings.append((goal.id, "expired_ttl"))
                continue

            # Deadline expiry
            if goal.deadline is not None and now > goal.deadline:
                goal.status = GoalStatus.EXPIRED
                warnings.append((goal.id, "expired_deadline"))
                continue

            # Approaching deadline warning
            if goal.deadline is not None:
                total = goal.deadline - goal.created_at
                remaining = goal.deadline - now
                if total > 0 and remaining / total < (1 - self._config.deadline_warning_ratio):
                    warnings.append((goal.id, "deadline_approaching"))

        return warnings

    def detect_stalled_goals(self) -> list[int]:
        """Detect active goals with no significant progress.

        Returns:
            List of stalled goal IDs.
        """
        stalled: list[int] = []
        cfg = self._config
        for goal in self._goals:
            if goal.status != GoalStatus.ACTIVE:
                continue
            hist = goal.progress_history
            if len(hist) < cfg.stall_steps:
                continue
            recent = hist[-cfg.stall_steps:]
            if max(recent) - min(recent) < cfg.stall_threshold:
                stalled.append(goal.id)
        return stalled

    def detect_opportunities(self, current_state: torch.Tensor) -> list[int]:
        """Detect paused or low-priority goals relevant to current context.

        Args:
            current_state: (d_model,) current embedding.

        Returns:
            List of opportunity goal IDs.
        """
        opportunities: list[int] = []
        for goal in self._goals:
            if goal.status not in (GoalStatus.PAUSED, GoalStatus.BLOCKED):
                continue
            emb = goal.embedding
            if emb.device != current_state.device:
                emb = emb.to(current_state.device)
            sim = F.cosine_similarity(
                current_state.unsqueeze(0), emb.unsqueeze(0)
            ).item()
            if sim > 0.5:
                opportunities.append(goal.id)
        return opportunities

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save_goals(self) -> list[dict]:
        """Serialize all goals to a list of dicts for persistence.

        Returns:
            List of serializable goal dicts.
        """
        now = time.monotonic()
        result: list[dict] = []
        for goal in self._goals:
            if goal.status in (GoalStatus.COMPLETED, GoalStatus.FAILED, GoalStatus.EXPIRED):
                continue
            elapsed = now - goal.created_at
            remaining_ttl = max(0.0, goal.ttl - elapsed)
            result.append({
                "id": goal.id,
                "embedding": goal.embedding.cpu().tolist(),
                "satisfaction_embedding": goal.satisfaction_embedding.cpu().tolist(),
                "description_hash": goal.description_hash.hex(),
                "priority": goal.priority,
                "urgency": goal.urgency,
                "importance": goal.importance,
                "progress": goal.progress,
                "status": goal.status.name,
                "parent_id": goal.parent_id,
                "child_ids": goal.child_ids,
                "dependency_ids": goal.dependency_ids,
                "remaining_ttl": remaining_ttl,
                "deadline": goal.deadline,
            })
        return result

    def load_goals(self, data: list[dict], device: torch.device | None = None) -> None:
        """Restore goals from serialized dicts.

        Args:
            data: List of goal dicts from save_goals().
            device: Device to place embeddings on.
        """
        dev = device or next(self.parameters()).device
        now = time.monotonic()
        self._goals.clear()
        for d in data:
            goal = Goal(
                id=d["id"],
                embedding=torch.tensor(d["embedding"], device=dev),
                satisfaction_embedding=torch.tensor(
                    d["satisfaction_embedding"], device=dev
                ),
                description_hash=bytes.fromhex(d["description_hash"]),
                priority=d["priority"],
                urgency=d["urgency"],
                importance=d["importance"],
                progress=d["progress"],
                status=GoalStatus[d["status"]],
                parent_id=d["parent_id"],
                child_ids=d["child_ids"],
                dependency_ids=d["dependency_ids"],
                created_at=now,
                deadline=d.get("deadline"),
                ttl=d.get("remaining_ttl", self._config.ttl_default),
                last_progress_update=now,
                progress_history=[],
            )
            self._goals.append(goal)
            self._next_goal_id = max(self._next_goal_id, goal.id + 1)

        logger.info("Loaded %d goals from serialized data", len(self._goals))

    # ------------------------------------------------------------------
    # Goal management
    # ------------------------------------------------------------------

    def remove_goal(self, goal_id: int) -> bool:
        """Remove a goal and unlink from parent/children.

        Args:
            goal_id: ID of the goal to remove.

        Returns:
            True if removed, False if not found.
        """
        goal = self._find_goal(goal_id)
        if goal is None:
            return False

        # Unlink from parent
        if goal.parent_id is not None:
            parent = self._find_goal(goal.parent_id)
            if parent is not None and goal_id in parent.child_ids:
                parent.child_ids.remove(goal_id)

        # Orphan children (set parent to None)
        for cid in goal.child_ids:
            child = self._find_goal(cid)
            if child is not None:
                child.parent_id = None

        self._goals = [g for g in self._goals if g.id != goal_id]
        return True

    def pause_goal(self, goal_id: int) -> None:
        """Pause an active goal."""
        goal = self._find_goal(goal_id)
        if goal is not None and goal.status == GoalStatus.ACTIVE:
            goal.status = GoalStatus.PAUSED

    def resume_goal(self, goal_id: int) -> None:
        """Resume a paused goal."""
        goal = self._find_goal(goal_id)
        if goal is not None and goal.status == GoalStatus.PAUSED:
            goal.status = GoalStatus.ACTIVE

    @property
    def active_goals(self) -> list[Goal]:
        """Return all goals with ACTIVE status."""
        return [g for g in self._goals if g.status == GoalStatus.ACTIVE]

    @property
    def all_goals(self) -> list[Goal]:
        """Return all goals regardless of status."""
        return list(self._goals)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_goal(self, goal_id: int) -> Goal | None:
        """Find a goal by ID."""
        for g in self._goals:
            if g.id == goal_id:
                return g
        return None

    def _complete_goal(self, goal: Goal) -> None:
        """Mark a goal as completed and propagate to parent."""
        goal.status = GoalStatus.COMPLETED
        logger.info("Goal %d completed (progress=%.3f)", goal.id, goal.progress)

        # Unblock dependent goals
        for other in self._goals:
            if goal.id in other.dependency_ids:
                other.dependency_ids.remove(goal.id)
                if not other.dependency_ids and other.status == GoalStatus.BLOCKED:
                    other.status = GoalStatus.ACTIVE
                    logger.debug("Goal %d unblocked", other.id)

        # Check if parent should complete (all children done)
        if goal.parent_id is not None:
            parent = self._find_goal(goal.parent_id)
            if parent is not None:
                children_done = all(
                    self._find_goal(cid) is None
                    or self._find_goal(cid).status == GoalStatus.COMPLETED
                    for cid in parent.child_ids
                )
                if children_done and parent.child_ids:
                    self._complete_goal(parent)

    def _evict_one(self) -> None:
        """Evict one goal to make room, preferring completed/expired/failed."""
        # Priority: completed > expired > failed > lowest-priority active
        for status in (GoalStatus.COMPLETED, GoalStatus.EXPIRED, GoalStatus.FAILED):
            for g in self._goals:
                if g.status == status:
                    self.remove_goal(g.id)
                    return

        # Evict lowest-priority active goal
        active = [g for g in self._goals if g.status == GoalStatus.ACTIVE]
        if active:
            worst = min(active, key=lambda g: g.priority)
            self.remove_goal(worst.id)
