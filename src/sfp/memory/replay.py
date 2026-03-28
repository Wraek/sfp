"""Generative Replay — synthetic episode generation during consolidation.

Operates off the critical inference path, producing synthetic episodes via
three strategies: basin interpolation, chain dreaming, and boundary
exploration.  Synthetic episodes are validated (embedding health, manifold
proximity, Tier 3 consistency, diversity, backbone coherence) and then
injected into the standard consolidation pipeline at reduced weight.

The engine uses **zero persistent parameters** — it exploits the backbone
transformer, Tier 2 basin geometry, and transition graph as implicit
generative models.  Total persistent footprint is ~64 KB (drift buffers);
transient VRAM peaks at ~7 MB during generation.
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F

from sfp.config import GenerativeReplayConfig
from sfp.types import ReplayStrategy, SyntheticEpisode
from sfp.utils.logging import get_logger

logger = get_logger("memory.replay")


class GenerativeReplay:
    """Off-critical-path synthetic episode generation.

    This is a plain class (**not** ``nn.Module``) with zero persistent
    learnable parameters.  All generation uses existing structures
    (backbone, Tier 2 keys/values, transition graph) and produces only
    transient tensors freed after each cycle.

    Three generation strategies:

    - **Basin interpolation** (~40%): Blend moderate-similarity basin pairs
      to explore the space between known concepts.
    - **Chain dreaming** (~30%): Traverse the transition graph offline
      (random walk, goal-directed, or counterfactual) to discover novel
      paths and implicit connections.
    - **Boundary exploration** (~30%): Probe decision boundaries between
      neighbouring basins with midpoint + perpendicular noise to sharpen
      concept boundaries.

    Args:
        config: GenerativeReplayConfig with ratios, thresholds, scheduling.
        d_model: Embedding dimensionality (default 512).
    """

    def __init__(
        self,
        config: GenerativeReplayConfig | None = None,
        d_model: int = 512,
    ) -> None:
        cfg = config or GenerativeReplayConfig()
        self._config = cfg
        self._d_model = d_model

        # --- Scheduling state ---
        self._total_episodes_seen: int = 0
        self._cycle_count: int = 0
        self._last_inference_time: float = time.monotonic()

        # --- Drift monitoring per-basin (EMA of key displacement) ---
        self._drift_ema: dict[int, float] = defaultdict(float)
        self._drift_baseline: dict[int, float] = defaultdict(float)
        self._drift_counts: dict[int, int] = defaultdict(int)

        # --- Generation statistics ---
        self._total_generated: int = 0
        self._total_validated: int = 0
        self._strategy_counts: dict[str, int] = defaultdict(int)
        self._recent_synthetics: list[torch.Tensor] = []  # diversity buffer

        logger.info(
            "GenerativeReplay initialized: warmup=%d, middle=%d, "
            "max_per_cycle=%d, d_model=%d",
            cfg.warmup_episodes,
            cfg.middle_episodes,
            cfg.max_synthetics_per_cycle,
            d_model,
        )

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def should_generate(
        self,
        cycle_count: int,
        total_episodes: int,
        idle_seconds: float = 0.0,
    ) -> tuple[bool, int]:
        """Determine whether to generate synthetic episodes this cycle.

        Scheduling phases:
          - **Warmup** (< ``warmup_episodes``): disabled.
          - **Middle** (warmup … ``middle_episodes``): every
            ``middle_cycle_interval``-th cycle, ``middle_synthetics``.
          - **Mature** (> ``middle_episodes``): every
            ``mature_cycle_interval``-th cycle, ``mature_synthetics``.
          - **Idle** (> ``idle_timeout_seconds`` since last inference):
            daydream with the budget appropriate to maturity.

        Returns:
            ``(should_run, n_synthetics)``
        """
        cfg = self._config
        self._cycle_count = cycle_count
        self._total_episodes_seen = total_episodes

        # Warmup: no replay — system must build a stable real knowledge base
        if total_episodes < cfg.warmup_episodes:
            return False, 0

        # Idle daydreaming (checked before schedule so we don't wait)
        if idle_seconds > cfg.idle_timeout_seconds:
            budget = (
                cfg.mature_synthetics
                if total_episodes > cfg.middle_episodes
                else cfg.middle_synthetics
            )
            return True, budget

        # Mature operation
        if total_episodes > cfg.middle_episodes:
            if cycle_count % cfg.mature_cycle_interval == 0:
                return True, cfg.mature_synthetics
        else:
            # Middle operation
            if cycle_count % cfg.middle_cycle_interval == 0:
                return True, cfg.middle_synthetics

        return False, 0

    def record_inference(self) -> None:
        """Called when the system processes a real inference request."""
        self._last_inference_time = time.monotonic()

    # ------------------------------------------------------------------
    # Batch generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        n: int,
        tier2: object,
        transitions: object | None = None,
        backbone: object | None = None,
        tier3: object | None = None,
        valence_system: object | None = None,
    ) -> list[SyntheticEpisode]:
        """Generate a batch of synthetic episodes using all three strategies.

        Budget allocation is approximately 40 % interpolation, 30 % dreaming,
        30 % boundary probing.  Each candidate is validated, weighted, and
        wrapped as a :class:`SyntheticEpisode`.

        Args:
            n: Target number of synthetics to generate.
            tier2: EssentialMemory (duck-typed, needs ``keys``, ``n_active``,
                   ``confidence``).
            transitions: TransitionStructure (duck-typed, needs
                         ``get_outgoing``).  Optional.
            backbone: BackboneTransformer (duck-typed, needs ``forward``).
                      Optional — used for coherence check.
            tier3: CoreMemory (duck-typed, needs ``keys``, ``values``,
                   ``n_active``).  Optional — used for axiom check.
            valence_system: ValenceSystem (duck-typed).  Optional, reserved
                           for future valence-weighted sampling.

        Returns:
            List of validated :class:`SyntheticEpisode` objects.
        """
        if n <= 0:
            return []

        excluded_basins = self.get_excluded_basins()

        # Allocate budget: ~40 % interp, ~30 % dream, ~30 % boundary
        n_interp = max(1, int(n * 0.4))
        n_dream = max(1, int(n * 0.3))
        n_boundary = n - n_interp - n_dream

        # raw candidates: (embedding, source_basins, strategy)
        all_raw: list[tuple[torch.Tensor, list[int], ReplayStrategy]] = []

        # Strategy 1: Basin interpolation
        for _ in range(n_interp):
            result = self.generate_interpolation(tier2, excluded_basins)
            if result is not None:
                emb, basins = result
                all_raw.append(
                    (emb, basins, ReplayStrategy.BASIN_INTERPOLATION)
                )

        # Strategy 2: Chain dreaming
        for _ in range(n_dream):
            result = self.generate_chain_dream(
                tier2, transitions, excluded_basins=excluded_basins,
            )
            if result is not None:
                emb, basins = result
                all_raw.append(
                    (emb, basins, ReplayStrategy.CHAIN_DREAMING)
                )

        # Strategy 3: Boundary exploration
        for _ in range(n_boundary):
            result = self.generate_boundary_probe(tier2, excluded_basins)
            if result is not None:
                emb, basins = result
                all_raw.append(
                    (emb, basins, ReplayStrategy.BOUNDARY_EXPLORATION)
                )

        self._total_generated += len(all_raw)
        for _, _, strat in all_raw:
            self._strategy_counts[strat.name] += 1

        # Validate and wrap
        validated: list[SyntheticEpisode] = []
        for emb, basins, strategy in all_raw:
            if self.validate_synthetic(emb, tier2, tier3, backbone):
                weight = self.compute_synthetic_weight(emb, tier2)
                episode = SyntheticEpisode(
                    embedding=emb.detach(),
                    source_basins=basins,
                    strategy=strategy,
                    validation_passed=True,
                    weight=weight,
                )
                validated.append(episode)

        self._total_validated += len(validated)

        # Update diversity buffer (keep last 200)
        for ep in validated:
            self._recent_synthetics.append(ep.embedding)
        if len(self._recent_synthetics) > 200:
            self._recent_synthetics = self._recent_synthetics[-200:]

        logger.debug(
            "Generated %d raw, %d validated "
            "(interp=%d, dream=%d, boundary=%d)",
            len(all_raw),
            len(validated),
            n_interp,
            n_dream,
            n_boundary,
        )

        return validated

    # ------------------------------------------------------------------
    # Strategy 1: Basin interpolation
    # ------------------------------------------------------------------

    def generate_interpolation(
        self,
        tier2: object,
        excluded_basins: set[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Blend two moderate-similarity basin keys to produce a novel embedding.

        Selects pairs where cosine similarity is in
        [``basin_similarity_min``, ``basin_similarity_max``] (default
        [0.2, 0.7]) and interpolates with a random α in
        [``interpolation_alpha_min``, ``interpolation_alpha_max``].

        Args:
            tier2: EssentialMemory (duck-typed).
            excluded_basins: Basin IDs to skip (excessive drift).

        Returns:
            ``(interpolated_embedding, [basin_a, basin_b])`` or ``None``.
        """
        cfg = self._config
        n_active = getattr(tier2, "n_active", 0)
        if n_active < 10:
            return None

        keys = getattr(tier2, "keys", None)
        if keys is None:
            return None

        active_keys = keys[:n_active]
        excluded = excluded_basins or set()

        # Candidate pool: all non-excluded active basins
        candidates = [
            i for i in range(n_active) if i not in excluded
        ]
        if len(candidates) < 10:
            return None

        # Attempt to find a valid pair with moderate similarity
        max_attempts = 20
        for _ in range(max_attempts):
            a, b = random.sample(candidates, 2)
            sim = F.cosine_similarity(
                active_keys[a].unsqueeze(0),
                active_keys[b].unsqueeze(0),
            ).item()
            if cfg.basin_similarity_min <= sim <= cfg.basin_similarity_max:
                break
        else:
            return None  # no valid pair found

        alpha = random.uniform(
            cfg.interpolation_alpha_min, cfg.interpolation_alpha_max,
        )
        interpolated = (1.0 - alpha) * active_keys[a] + alpha * active_keys[b]

        # Renormalize to match average key norm of the pair
        mean_norm = (active_keys[a].norm() + active_keys[b].norm()) / 2.0
        if mean_norm > 1e-8:
            interpolated = F.normalize(interpolated, dim=-1) * mean_norm

        return interpolated.detach(), [a, b]

    # ------------------------------------------------------------------
    # Strategy 2: Chain dreaming
    # ------------------------------------------------------------------

    def generate_chain_dream(
        self,
        tier2: object,
        transitions: object | None = None,
        goal_context: torch.Tensor | None = None,
        excluded_basins: set[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Traverse the transition graph or key-similarity graph offline.

        Three dreaming modes (selected stochastically):
          - **Random walk** (50 %): follow random outgoing edges with
            exploration noise.
          - **Goal-directed** (30 %): aim for a semantically related but
            transitionally disconnected basin.
          - **Counterfactual** (20 %): perturb a hop in the walk to explore
            alternative paths.

        The dream embedding is the normalized mean of all visited basin keys.

        Args:
            tier2: EssentialMemory (duck-typed).
            transitions: TransitionStructure (duck-typed, optional).
            goal_context: Optional (d_model,) tensor for goal-directed mode.
            excluded_basins: Basin IDs to skip (excessive drift).

        Returns:
            ``(dream_embedding, visited_basins)`` or ``None``.
        """
        n_active = getattr(tier2, "n_active", 0)
        if n_active < 5:
            return None

        keys = getattr(tier2, "keys", None)
        if keys is None:
            return None

        active_keys = keys[:n_active]
        excluded = excluded_basins or set()
        n_hops = random.randint(3, 5)

        if transitions is not None and hasattr(transitions, "get_outgoing"):
            return self._dream_with_transitions(
                tier2, transitions, active_keys, n_active,
                n_hops, excluded,
            )
        else:
            return self._dream_fallback(
                active_keys, n_active, n_hops, excluded,
            )

    def _dream_with_transitions(
        self,
        tier2: object,
        transitions: object,
        active_keys: torch.Tensor,
        n_active: int,
        n_hops: int,
        excluded: set[int],
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Chain dreaming via the transition structure.

        Uses confidence-weighted start selection and noisy softmax over
        outgoing edge weights for exploration.
        """
        confidence = getattr(tier2, "confidence", None)

        # Select start basin (confidence-weighted, excluding drifting basins)
        if confidence is not None:
            conf = confidence[:n_active].float().clone()
            for bid in excluded:
                if bid < n_active:
                    conf[bid] = 0.0
            total = conf.sum()
            if total < 1e-8:
                return None
            weights = conf / total
            start = torch.multinomial(weights, 1).item()
        else:
            candidates = [i for i in range(n_active) if i not in excluded]
            if not candidates:
                return None
            start = random.choice(candidates)

        visited: list[int] = [start]
        current = start
        state = active_keys[start].clone()

        for _ in range(n_hops):
            try:
                targets, edge_weights, _relations = transitions.get_outgoing(
                    current,
                )
            except Exception:
                break

            if len(targets) == 0:
                break

            # Add exploration noise (temperature=2.0)
            noisy = edge_weights.float() + torch.randn_like(
                edge_weights.float(),
            ) * 0.1
            probs = F.softmax(noisy / 2.0, dim=-1)
            idx = torch.multinomial(probs, 1).item()
            next_basin = targets[idx].item()

            if (
                next_basin >= n_active
                or next_basin in visited
                or next_basin in excluded
            ):
                break

            visited.append(next_basin)
            hop_key = active_keys[next_basin]
            state = 0.7 * hop_key + 0.3 * state
            current = next_basin

        if len(visited) < 2:
            return None

        # Dream embedding = normalized mean of visited basin keys
        visited_keys = torch.stack([active_keys[b] for b in visited])
        dream_emb = visited_keys.mean(dim=0)
        ref_norm = active_keys[visited[0]].norm()
        if ref_norm > 1e-8:
            dream_emb = F.normalize(dream_emb, dim=-1) * ref_norm

        return dream_emb.detach(), visited

    def _dream_fallback(
        self,
        active_keys: torch.Tensor,
        n_active: int,
        n_hops: int,
        excluded: set[int],
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Cosine-similarity-based random walk (no transition structure).

        At each hop, picks from the top-5 most similar unvisited basins
        with softmax noise.
        """
        candidates = [i for i in range(n_active) if i not in excluded]
        if len(candidates) < 5:
            return None

        start = random.choice(candidates)
        visited: list[int] = [start]
        current = start

        for _ in range(n_hops):
            sims = F.cosine_similarity(
                active_keys[current].unsqueeze(0),
                active_keys,
                dim=-1,
            )
            # Mask already visited and excluded
            for v in visited:
                sims[v] = -2.0
            for e in excluded:
                if e < n_active:
                    sims[e] = -2.0

            remaining = (sims > -1.5).sum().item()
            top_k = min(5, int(remaining))
            if top_k <= 0:
                break

            values, indices = sims.topk(top_k)
            probs = F.softmax(values / 0.5, dim=-1)
            pick = torch.multinomial(probs, 1).item()
            next_basin = indices[pick].item()
            visited.append(next_basin)
            current = next_basin

        if len(visited) < 2:
            return None

        visited_keys = torch.stack([active_keys[b] for b in visited])
        dream_emb = visited_keys.mean(dim=0)
        ref_norm = active_keys[visited[0]].norm()
        if ref_norm > 1e-8:
            dream_emb = F.normalize(dream_emb, dim=-1) * ref_norm

        return dream_emb.detach(), visited

    # ------------------------------------------------------------------
    # Strategy 3: Boundary exploration
    # ------------------------------------------------------------------

    def generate_boundary_probe(
        self,
        tier2: object,
        excluded_basins: set[int] | None = None,
    ) -> tuple[torch.Tensor, list[int]] | None:
        """Probe the decision boundary between two neighbouring basins.

        Generates an embedding near the midpoint of two close basins, with
        a small perpendicular perturbation to explore boundary width.

        The interpolation factor is drawn from ``Beta(3, 3)`` (peaked at
        0.5 — boundary midpoint), and perpendicular noise is 5 % of the
        inter-basin direction norm.

        Args:
            tier2: EssentialMemory (duck-typed).
            excluded_basins: Basin IDs to skip (excessive drift).

        Returns:
            ``(probe_embedding, [basin_a, basin_b])`` or ``None``.
        """
        n_active = getattr(tier2, "n_active", 0)
        if n_active < 5:
            return None

        keys = getattr(tier2, "keys", None)
        if keys is None:
            return None

        active_keys = keys[:n_active]
        excluded = excluded_basins or set()

        candidates = [i for i in range(n_active) if i not in excluded]
        if len(candidates) < 5:
            return None

        # Pick a random anchor basin and find its nearest neighbour
        anchor = random.choice(candidates)
        sims = F.cosine_similarity(
            active_keys[anchor].unsqueeze(0),
            active_keys,
            dim=-1,
        )
        sims[anchor] = -2.0  # mask self
        for e in excluded:
            if e < n_active:
                sims[e] = -2.0
        neighbour = sims.argmax().item()

        key_a = active_keys[anchor]
        key_b = active_keys[neighbour]

        # Beta(3, 3) distribution peaks at 0.5 — stay near midpoint
        alpha = random.betavariate(3, 3)
        probe = (1.0 - alpha) * key_a + alpha * key_b

        # Add perpendicular noise (5 % of inter-basin distance)
        direction = key_b - key_a
        dir_norm_sq = (direction * direction).sum() + 1e-8
        perp = torch.randn_like(direction)
        dot = (perp * direction).sum()
        perp = perp - (dot / dir_norm_sq) * direction
        perp_scale = direction.norm() * 0.05
        if perp.norm() > 1e-8:
            perp = F.normalize(perp, dim=-1) * perp_scale
        probe = probe + perp

        # Renormalize to match anchor norm
        anchor_norm = key_a.norm()
        if anchor_norm > 1e-8:
            probe = F.normalize(probe, dim=-1) * anchor_norm

        return probe.detach(), [anchor, neighbour]

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_synthetic(
        self,
        embedding: torch.Tensor,
        tier2: object,
        tier3: object | None = None,
        backbone: object | None = None,
    ) -> bool:
        """Apply 5 validation checks to a candidate synthetic embedding.

        1. **Embedding health**: not NaN / Inf, reasonable norm.
        2. **Manifold proximity**: close enough to an existing real basin
           (cosine distance < ``manifold_proximity_threshold``).
        3. **Tier 3 consistency**: does not contradict core axioms.
        4. **Diversity**: not too similar to recent synthetics
           (cosine sim < ``diversity_threshold``).
        5. **Backbone coherence**: backbone output is non-degenerate
           (no NaN, not zero-norm, not max-entropy).

        Args:
            embedding: (d_model,) candidate synthetic.
            tier2: EssentialMemory (duck-typed).
            tier3: CoreMemory (duck-typed, optional).
            backbone: BackboneTransformer (duck-typed, optional).

        Returns:
            ``True`` if all checks pass.
        """
        cfg = self._config

        # ----- Check 1: embedding health -----
        if torch.isnan(embedding).any() or torch.isinf(embedding).any():
            return False
        norm = embedding.norm().item()
        if norm < 1e-3 or norm > 1e4:
            return False

        # ----- Check 2: manifold proximity -----
        n_active = getattr(tier2, "n_active", 0)
        keys = getattr(tier2, "keys", None)
        if keys is not None and n_active > 0:
            active_keys = keys[:n_active]
            sims = F.cosine_similarity(
                embedding.unsqueeze(0), active_keys, dim=-1,
            )
            max_sim = sims.max().item()
            distance = 1.0 - max_sim
            if distance > cfg.manifold_proximity_threshold:
                return False

        # ----- Check 3: Tier 3 consistency -----
        if tier3 is not None:
            t3_n = getattr(tier3, "n_active", 0)
            t3_keys = getattr(tier3, "keys", None)
            t3_values = getattr(tier3, "values", None)
            if (
                t3_keys is not None
                and t3_values is not None
                and t3_n > 0
            ):
                t3_sims = F.cosine_similarity(
                    embedding.unsqueeze(0), t3_keys[:t3_n], dim=-1,
                )
                if t3_sims.max().item() > 0.7:
                    # Strongly activates a core basin — verify coherence
                    t3_id = t3_sims.argmax().item()
                    t3_val = t3_values[t3_id]
                    coherence = F.cosine_similarity(
                        embedding.unsqueeze(0), t3_val.unsqueeze(0),
                    ).item()
                    if coherence < 0.0:
                        return False  # contradicts core axiom

        # ----- Check 4: diversity vs recent synthetics -----
        check_window = self._recent_synthetics[-50:]
        for recent_emb in check_window:
            if recent_emb.device != embedding.device:
                recent_emb = recent_emb.to(embedding.device)
            sim = F.cosine_similarity(
                embedding.unsqueeze(0), recent_emb.unsqueeze(0),
            ).item()
            if sim > cfg.diversity_threshold:
                return False  # too similar to a recent synthetic

        # ----- Check 5: backbone coherence -----
        if backbone is not None and hasattr(backbone, "forward"):
            try:
                with torch.no_grad():
                    # Backbone expects (batch, seq, d_model) in most cases
                    inp = embedding.unsqueeze(0).unsqueeze(0)
                    output = backbone(inp)
                    if isinstance(output, torch.Tensor):
                        out = output.squeeze()
                        if torch.isnan(out).any() or torch.isinf(out).any():
                            return False
                        if out.norm().item() < 1e-6:
                            return False
                        # Entropy check (only for 1-D logit-like outputs)
                        if out.dim() == 1 and out.numel() > 1:
                            probs = F.softmax(out, dim=-1)
                            entropy = -(
                                probs * torch.log(probs + 1e-8)
                            ).sum().item()
                            max_entropy = math.log(out.numel())
                            if (
                                max_entropy > 0
                                and entropy / max_entropy
                                > cfg.backbone_coherence_threshold
                            ):
                                return False
            except Exception:
                pass  # backbone not compatible — skip check

        return True

    # ------------------------------------------------------------------
    # Synthetic weight
    # ------------------------------------------------------------------

    def compute_synthetic_weight(
        self,
        embedding: torch.Tensor,
        tier2: object,
    ) -> float:
        """Assign weight to a validated synthetic based on basin proximity.

        Closer to an existing basin → **lower** weight (less novel).
        Further away (but still valid) → **higher** weight (more
        informative for generalization).

        Returns:
            Weight in [``synthetic_weight_min``, ``synthetic_weight_max``].
        """
        cfg = self._config
        n_active = getattr(tier2, "n_active", 0)
        keys = getattr(tier2, "keys", None)
        if keys is None or n_active == 0:
            return cfg.synthetic_weight_min

        active_keys = keys[:n_active]
        sims = F.cosine_similarity(
            embedding.unsqueeze(0), active_keys, dim=-1,
        )
        max_sim = sims.max().item()
        # Invert: high similarity → low weight, low similarity → high weight
        t = 1.0 - max(0.0, min(1.0, max_sim))
        weight = cfg.synthetic_weight_min + t * (
            cfg.synthetic_weight_max - cfg.synthetic_weight_min
        )
        return weight

    # ------------------------------------------------------------------
    # Drift monitoring
    # ------------------------------------------------------------------

    def update_drift_monitoring(
        self,
        basin_id: int,
        old_key: torch.Tensor,
        new_key: torch.Tensor,
    ) -> None:
        """Track per-basin key drift via EMA.

        Called by the consolidation engine after basin key updates.  The
        *fast* EMA (``drift_ema_decay``) tracks recent drift rate; the
        *slow* EMA (0.99) tracks the long-term baseline.

        Args:
            basin_id: Basin that was updated.
            old_key: (d_model,) key before update.
            new_key: (d_model,) key after update.
        """
        cfg = self._config
        drift = (new_key - old_key).norm().item()

        # Fast EMA — recent drift rate
        mom = cfg.drift_ema_decay
        prev = self._drift_ema.get(basin_id, drift)
        self._drift_ema[basin_id] = mom * prev + (1.0 - mom) * drift

        self._drift_counts[basin_id] = self._drift_counts.get(basin_id, 0) + 1

        # Slow EMA — long-term baseline
        base_mom = 0.99
        prev_base = self._drift_baseline.get(basin_id, drift)
        self._drift_baseline[basin_id] = (
            base_mom * prev_base + (1.0 - base_mom) * drift
        )

    def is_drift_excessive(self, basin_id: int) -> bool:
        """Check whether a basin's drift has exceeded safe limits.

        Returns ``True`` when current drift > ``drift_throttle_multiplier``
        × baseline (default 2 ×), indicating generative replay may be
        amplifying errors for this basin.

        At least 20 observations are required before a verdict.
        """
        cfg = self._config
        if self._drift_counts.get(basin_id, 0) < 20:
            return False

        current = self._drift_ema.get(basin_id, 0.0)
        baseline = self._drift_baseline.get(basin_id, 0.0)
        if baseline < 1e-8:
            return False

        return current > cfg.drift_throttle_multiplier * baseline

    def get_excluded_basins(self) -> set[int]:
        """Return basin IDs where drift is excessive (replay should skip).

        Replay generation functions use this set to avoid sourcing
        synthetic episodes from basins that appear to be drifting.
        """
        excluded: set[int] = set()
        for basin_id in self._drift_ema:
            if self.is_drift_excessive(basin_id):
                excluded.add(basin_id)
        return excluded

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_generation_stats(self) -> dict:
        """Return a summary of generation activity."""
        return {
            "total_generated": self._total_generated,
            "total_validated": self._total_validated,
            "validation_rate": (
                self._total_validated / max(1, self._total_generated)
            ),
            "cycle_count": self._cycle_count,
            "total_episodes_seen": self._total_episodes_seen,
            "strategy_counts": dict(self._strategy_counts),
            "drift_monitored_basins": len(self._drift_ema),
            "excluded_basins": len(self.get_excluded_basins()),
        }

    def reset_stats(self) -> None:
        """Reset generation statistics (but preserve drift monitoring)."""
        self._total_generated = 0
        self._total_validated = 0
        self._strategy_counts.clear()
        self._recent_synthetics.clear()
