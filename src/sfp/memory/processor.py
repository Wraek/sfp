"""HierarchicalMemoryProcessor — top-level orchestrator composing all tiers.

Wires together Perceiver IO, backbone transformer, the four memory tiers,
consolidation engine, reasoning system, defense framework, and six optional
cognitive modules (world model, goals, metacognition, valence, salience gate,
generative replay) into a single coherent processing pipeline.
"""

from __future__ import annotations

import time

import torch

from sfp.config import (
    BackboneConfig,
    ConsolidationConfig,
    DefenseConfig,
    EWCConfig,
    FieldConfig,
    GenerativeReplayConfig,
    GoalPersistenceConfig,
    LoRAConfig,
    MetacognitionConfig,
    PerceiverConfig,
    ReasoningChainConfig,
    SelectiveAttentionConfig,
    StreamingConfig,
    Tier0Config,
    Tier1Config,
    Tier2Config,
    Tier3Config,
    TransitionConfig,
    ValenceConfig,
    WorldModelConfig,
)
from sfp.core.backbone import BackboneTransformer
from sfp.core.field import SemanticFieldProcessor
from sfp.core.perceiver import PerceiverIO
from sfp.core.streaming import StreamingProcessor
from sfp.defense.anchor_verification import AnchorVerifier
from sfp.defense.gradient_bounds import AdaptiveGradientClipper, UpdateBudget
from sfp.defense.input_validation import EmbeddingAnomalyDetector, InputSanitizer
from sfp.defense.topology_monitor import ManifoldIntegrityMonitor
from sfp.memory.consolidation import ConsolidationEngine
from sfp.memory.core import CoreMemory
from sfp.memory.episodic import EpisodicMemory
from sfp.memory.essential import EssentialMemory
from sfp.memory.events import CriteriaAuthorizationHandler, PromotionEventEmitter
from sfp.memory.integrity import compute_episode_hash, compute_weight_hash
from sfp.reasoning.chain import AssociativeReasoningChain
from sfp.reasoning.learning import ChainShortcutLearner, TransitionLearner
from sfp.reasoning.router import ReasoningRouter
from sfp.reasoning.transitions import TransitionStructure
from sfp.types import (
    ConsolidationMode,
    Episode,
    ProcessingLevel,
    ReasoningResult,
    SurpriseMetric,
    ValenceSignal,
    WorldModelState,
)
from sfp.utils.device import resolve_device
from sfp.utils.logging import get_logger

logger = get_logger("memory.processor")


class HierarchicalMemoryProcessor:
    """Top-level processor composing the full SFP architecture.

    Pipeline on process():
      0. (NEW) Salience gate → SKIP / SKIM / FULL
      1. Input sanitization (defense layer 1)
      2. Perceiver IO (compress multi-modal to fixed latents)
      3. Backbone transformer (contextual processing)
      4. Embedding anomaly detection (defense layer 2)
      4.5. (NEW) World model train_step
      4.6. (NEW) Goal progress update + priority recomputation
      5. Tier 3 retrieve (core axioms)
      6. Tier 2 retrieve / reasoning chain via router
      6.5. (NEW) Goal + valence reasoning bias, annotate visited basins
      6.6. (NEW) Metacognition uncertainty estimation
      7. Tier 0 forward + surprise + gated update (enhanced with WM surprise + valence modifier)
      8. Maybe store episode in Tier 1 (enhanced with valence annotation)
      9. Maybe trigger consolidation (enhanced with generative replay)
      10. (NEW) Goal deadline/stall monitoring

    Pipeline on query():
      Steps 1-6 (read-only, no weight updates)

    Args:
        field_config: Configuration for the MLP manifold (Tier 0).
        perceiver_config: Configuration for Perceiver IO.
        backbone_config: Configuration for the backbone transformer.
        tier0_config: Tier 0 hardening parameters.
        tier1_config: Tier 1 episodic memory parameters.
        tier2_config: Tier 2 essential memory parameters.
        tier3_config: Tier 3 core memory parameters.
        consolidation_config: Consolidation engine scheduling.
        transition_config: Transition structure parameters.
        reasoning_config: Reasoning chain parameters.
        defense_config: Defense framework parameters.
        streaming_config: Streaming update parameters.
        lora_config: LoRA adapter parameters.
        ewc_config: EWC forgetting protection parameters.
        world_model_config: Predictive world model config (None = disabled).
        goal_config: Goal persistence config (None = disabled).
        metacognition_config: Metacognition engine config (None = disabled).
        valence_config: Valence and affect config (None = disabled).
        attention_config: Selective attention / salience gate config (None = disabled).
        replay_config: Generative replay config (None = disabled).
        device: Device string ("auto", "cpu", "cuda", "mps").
    """

    def __init__(
        self,
        field_config: FieldConfig | None = None,
        perceiver_config: PerceiverConfig | None = None,
        backbone_config: BackboneConfig | None = None,
        tier0_config: Tier0Config | None = None,
        tier1_config: Tier1Config | None = None,
        tier2_config: Tier2Config | None = None,
        tier3_config: Tier3Config | None = None,
        consolidation_config: ConsolidationConfig | None = None,
        transition_config: TransitionConfig | None = None,
        reasoning_config: ReasoningChainConfig | None = None,
        defense_config: DefenseConfig | None = None,
        streaming_config: StreamingConfig | None = None,
        lora_config: LoRAConfig | None = None,
        ewc_config: EWCConfig | None = None,
        world_model_config: WorldModelConfig | None = None,
        goal_config: GoalPersistenceConfig | None = None,
        metacognition_config: MetacognitionConfig | None = None,
        valence_config: ValenceConfig | None = None,
        attention_config: SelectiveAttentionConfig | None = None,
        replay_config: GenerativeReplayConfig | None = None,
        device: str = "auto",
    ) -> None:
        # Resolve device
        self._device = resolve_device(device)

        # Store configs (defaults if not provided)
        field_cfg = field_config or FieldConfig()
        perceiver_cfg = perceiver_config or PerceiverConfig(d_input=field_cfg.dim, d_latent=field_cfg.dim)
        backbone_cfg = backbone_config or BackboneConfig(d_model=field_cfg.dim)
        t0_cfg = tier0_config or Tier0Config()
        t1_cfg = tier1_config or Tier1Config()
        t2_cfg = tier2_config or Tier2Config(d_value=field_cfg.dim)
        t3_cfg = tier3_config or Tier3Config(d_value=field_cfg.dim)
        consolidation_cfg = consolidation_config or ConsolidationConfig()
        transition_cfg = transition_config or TransitionConfig()
        reasoning_cfg = reasoning_config or ReasoningChainConfig()
        defense_cfg = defense_config or DefenseConfig()
        streaming_cfg = streaming_config or StreamingConfig()

        d_model = field_cfg.dim

        # --- Neural architecture ---
        self._perceiver = PerceiverIO(perceiver_cfg).to(self._device)
        self._backbone = BackboneTransformer(backbone_cfg).to(self._device)

        # --- Memory tiers ---
        field = SemanticFieldProcessor(field_cfg).to(self._device)

        # Tier 2 (essential memory) — created before Tier 0 so we can wire consistency checker
        self._tier2 = EssentialMemory(t2_cfg, d_model=d_model).to(self._device)

        # Tier 0 (working memory = StreamingProcessor)
        self._tier0 = StreamingProcessor(
            field,
            streaming_config=streaming_cfg,
            lora_config=lora_config,
            ewc_config=ewc_config,
            tier0_config=t0_cfg,
            consistency_checker=self._tier2,
        )

        # Tier 1 (episodic memory)
        self._tier1 = EpisodicMemory(t1_cfg, d_model=d_model)

        # Tier 3 (core memory) with event-based authorization
        self._event_emitter = PromotionEventEmitter(default_approve=False)
        criteria_handler = CriteriaAuthorizationHandler(
            min_confidence=t3_cfg.min_confidence,
            min_episode_count=t3_cfg.min_episode_count,
            min_modalities=t3_cfg.min_modalities,
            min_age_days=t3_cfg.min_age_days,
        )
        self._event_emitter.register(criteria_handler)
        self._tier3 = CoreMemory(t3_cfg, d_model=d_model, event_emitter=self._event_emitter).to(self._device)

        # --- Reasoning system ---
        self._transitions = TransitionStructure(transition_cfg, d_model=d_model).to(self._device)
        self._chain = AssociativeReasoningChain(self._tier2, self._transitions, reasoning_cfg)
        self._router = ReasoningRouter(self._tier2, self._transitions, self._chain)
        self._transition_learner = TransitionLearner(self._tier2, self._transitions)
        self._shortcut_learner = ChainShortcutLearner(self._transitions, reasoning_cfg)

        # --- Consolidation engine ---
        self._consolidation = ConsolidationEngine(
            config=consolidation_cfg,
            tier0=self._tier0,
            tier1=self._tier1,
            tier2=self._tier2,
            tier3=self._tier3,
        )

        # --- Defense framework ---
        self._defense_config = defense_cfg
        self._sanitizer = InputSanitizer()
        self._anomaly_detector = EmbeddingAnomalyDetector(d_model=d_model, threshold=defense_cfg.embedding_anomaly_threshold)
        self._gradient_clipper = AdaptiveGradientClipper(field)
        self._update_budget = UpdateBudget(field)
        self._topology_monitor = ManifoldIntegrityMonitor(defense_cfg)
        self._anchor_verifier: AnchorVerifier | None = None

        # --- Cognitive modules (optional, None = disabled) ---

        self._world_model = None
        if world_model_config is not None:
            from sfp.prediction.world_model import PredictiveWorldModel
            self._world_model = PredictiveWorldModel(
                world_model_config, d_model=d_model,
            ).to(self._device)

        self._goals = None
        if goal_config is not None:
            from sfp.goals.persistence import GoalRegister
            self._goals = GoalRegister(
                goal_config, d_model=d_model,
            ).to(self._device)

        self._metacognition = None
        if metacognition_config is not None:
            from sfp.metacognition.uncertainty import MetacognitionEngine
            self._metacognition = MetacognitionEngine(
                metacognition_config, d_model=d_model,
            ).to(self._device)

        self._valence = None
        if valence_config is not None:
            from sfp.affect.valence import ValenceSystem
            self._valence = ValenceSystem(
                valence_config, d_model=d_model,
            ).to(self._device)

        # Co-adaptation: wire valence into consolidation for weighted sampling
        if self._valence is not None:
            self._consolidation.set_valence_system(self._valence)

        self._salience_gate = None
        if attention_config is not None:
            from sfp.attention.salience import SalienceGate
            self._salience_gate = SalienceGate(
                attention_config, d_model=d_model,
            ).to(self._device)

        self._replay = None
        if replay_config is not None:
            from sfp.memory.replay import GenerativeReplay
            self._replay = GenerativeReplay(
                replay_config, d_model=d_model,
            )

        # Scene graph for spatial reasoning (always available, activated by metadata)
        from sfp.reasoning.scene_graph import SceneGraph
        self._scene_graph = SceneGraph(d_model=d_model).to(self._device)

        # G1: Topology tracker (optional — requires giotto-tda)
        self._topology_tracker = None
        try:
            from sfp.topology.homology import PersistentHomologyTracker
            self._topology_tracker = PersistentHomologyTracker()
        except ImportError:
            pass

        # Cognitive module state
        self._last_wm_state: WorldModelState | None = None
        self._last_wm_reward: float = 0.0
        self._last_valence: ValenceSignal | None = None
        self._last_continue_prob: float = 1.0
        self._spatial_prediction_error_ema: float = 1.0

        # Chain valence bias from previous step (F4)
        self._last_chain_valence_bias: dict[int, float] = {}

        # Cognitive signal histories for LoRA merge context (C)
        self._uncertainty_history: list[float] = []
        self._mood_history: list[float] = []

        # Step counter
        self._step_count: int = 0
        self._salience_stats: dict[str, int] = {"skip": 0, "skim": 0, "full": 0}

        enabled_modules = [
            name for name, obj in [
                ("world_model", self._world_model),
                ("goals", self._goals),
                ("metacognition", self._metacognition),
                ("valence", self._valence),
                ("salience_gate", self._salience_gate),
                ("replay", self._replay),
            ] if obj is not None
        ]
        logger.info(
            "HierarchicalMemoryProcessor initialized: dim=%d, device=%s, "
            "cognitive_modules=%s",
            d_model, self._device, enabled_modules or "none",
        )

    def process(
        self,
        x: torch.Tensor,
        modality: str = "tensor",
        target: torch.Tensor | None = None,
        metadata: dict | None = None,
    ) -> SurpriseMetric:
        """Process a single input through the full hierarchical pipeline.

        Args:
            x: Input tensor. Shape depends on whether Perceiver is used:
               - (d_model,) or (B, d_model) for direct field input
               - (B, N, d_input) for Perceiver IO input
            modality: Input modality identifier.
            target: Optional target for supervised updates.
            metadata: Optional bridge-provided metadata dict.  Supported
                keys: ``"entity_positions"`` (N, 3), ``"entity_embeddings"``
                (N, d_model).

        Returns:
            SurpriseMetric from the Tier 0 update.
        """
        # Always increment step count so process() calls are tracked
        # even when the salience gate filters the observation.
        self._step_count += 1

        # Salience combined score (set if gate runs and reaches FULL)
        _salience_combined: float | None = None

        # ----------------------------------------------------------------
        # Step 0: Salience gate — evaluate → SKIP / SKIM / FULL
        # ----------------------------------------------------------------
        if self._salience_gate is not None:
            with torch.no_grad():
                # Build modality input for gate (pool to d_model if needed)
                gate_emb = self._pool_to_vector(x).to(self._device)
                gate_inputs = {modality: gate_emb}

                # Gather context signals
                goal_ctx = (
                    self._goals.get_goal_context().to(self._device)
                    if self._goals is not None
                    else None
                )
                wm_pred = None
                if self._world_model is not None and self._last_wm_state is not None:
                    wm_pred = self._last_wm_state.deterministic.to(self._device)

                salience_result = self._salience_gate.evaluate(
                    gate_inputs,
                    goal_context=goal_ctx,
                    world_model_prediction=wm_pred,
                )

                if salience_result.level == ProcessingLevel.SKIP:
                    self._salience_stats["skip"] += 1
                    return SurpriseMetric(
                        grad_norm=0.0, loss=0.0, updated=False,
                    )

                if salience_result.level == ProcessingLevel.SKIM:
                    self._salience_stats["skim"] += 1
                    self._salience_gate.skim_process(
                        gate_emb, modality, self._tier0,
                    )
                    # 2B: Hindsight label — SKIM inputs are not useful for field learning
                    self._salience_gate.train_hindsight(
                        salience_result.combined_salience, False, gate_emb,
                    )
                    return SurpriseMetric(
                        grad_norm=0.0, loss=0.0, updated=False,
                    )

                self._salience_stats["full"] += 1
                _salience_combined = getattr(salience_result, 'combined_salience', None)

        # ----------------------------------------------------------------
        # Step 1: Input sanitization
        # ----------------------------------------------------------------
        x_clean = self._sanitizer.sanitize(x, modality)

        # ----------------------------------------------------------------
        # Steps 2-3: Perceiver IO + Backbone
        # ----------------------------------------------------------------
        if x_clean.dim() == 3:
            x_clean = x_clean.to(self._device)
            with torch.amp.autocast(self._device.type, enabled=self._device.type == "cuda"):
                latents = self._perceiver(x_clean)
                contextualized = self._backbone(latents)
            embedding = contextualized.float().mean(dim=1)
        elif x_clean.dim() == 2:
            embedding = x_clean.to(self._device)
        else:
            embedding = x_clean.unsqueeze(0).to(self._device)

        # ----------------------------------------------------------------
        # Step 4: Embedding anomaly detection
        # ----------------------------------------------------------------
        self._anomaly_detector.update_statistics(embedding, modality)
        is_anomalous = self._anomaly_detector.is_anomalous(embedding, modality)
        if is_anomalous:
            logger.warning(
                "Anomalous input detected (modality=%s) — proceeding with caution",
                modality,
            )

        # Flatten to (d_model,) for downstream steps
        query_vec = (
            embedding.squeeze(0)
            if embedding.dim() == 2 and embedding.shape[0] == 1
            else embedding
        )
        if query_vec.dim() > 1:
            query_vec = query_vec[0]

        # ----------------------------------------------------------------
        # Step 4.5 (NEW): World model train_step
        # ----------------------------------------------------------------
        if self._world_model is not None:
            spatial_pos = metadata.get("spatial_position") if metadata else None
            wm_losses = self._world_model.train_step(
                query_vec, spatial_position=spatial_pos,
            )
            self._last_wm_state = WorldModelState(
                deterministic=self._world_model._h.detach().clone(),
                stochastic=self._world_model._z.detach().clone(),
                prediction_error=wm_losses.get("prediction_error", 0.0),
                kl_divergence=wm_losses.get("kl_loss", 0.0),
                reconstruction_error=wm_losses.get("reconstruction_loss", 0.0),
            )

            # 2D: Compute predicted reward for valence system
            with torch.no_grad():
                wm_latent = torch.cat([
                    self._last_wm_state.deterministic,
                    self._last_wm_state.stochastic,
                ])
                self._last_wm_reward = self._world_model.reward_head(wm_latent).item()

        # ----------------------------------------------------------------
        # Step 4.51: continue_head → episode boundary detection
        # ----------------------------------------------------------------
        episode_boundary = False
        if self._world_model is not None:
            continue_prob = wm_losses.get("continue_probability", 1.0)
            self._last_continue_prob = continue_prob
            wm_cfg = self._world_model._config

            if continue_prob < wm_cfg.continue_threshold:
                episode_boundary = True
                logger.info(
                    "Episode boundary detected: P(continue)=%.3f < %.3f",
                    continue_prob, wm_cfg.continue_threshold,
                )

        # ----------------------------------------------------------------
        # Step 4.52: spatial_predictor → spatial prediction error as surprise
        # ----------------------------------------------------------------
        _spatial_surprise_boost: float = 0.0
        if (
            self._world_model is not None
            and self._world_model._config.spatial_surprise_enabled
            and metadata is not None
        ):
            spatial_pos = metadata.get("spatial_position")
            if spatial_pos is not None:
                spatial_error = self._world_model.compute_spatial_prediction_error(
                    spatial_pos,
                )
                if spatial_error is not None:
                    decay = self._world_model._config.spatial_surprise_ema_decay
                    self._spatial_prediction_error_ema = (
                        decay * self._spatial_prediction_error_ema
                        + (1 - decay) * max(spatial_error, 1e-8)
                    )
                    normalized_error = spatial_error / max(
                        self._spatial_prediction_error_ema, 1e-8,
                    )
                    if normalized_error > 1.5:
                        _spatial_surprise_boost = (
                            self._world_model._config.spatial_surprise_weight
                            * (normalized_error - 1.0)
                        )

        # ----------------------------------------------------------------
        # Step 4.6 (NEW): Goal progress update + priority recomputation
        # ----------------------------------------------------------------
        if self._goals is not None:
            for goal in self._goals._goals:
                if goal.status.name == "ACTIVE":
                    self._goals.update_progress(goal.id, query_vec)
            self._goals.compute_priorities()

        # ----------------------------------------------------------------
        # Step 4.7: Scene graph spatial reasoning (if metadata provides entities)
        # ----------------------------------------------------------------
        spatial_relations = []
        if metadata is not None:
            entity_positions = metadata.get("entity_positions")
            entity_embeddings = metadata.get("entity_embeddings")
            if entity_positions is not None and entity_embeddings is not None:
                with torch.no_grad():
                    spatial_relations = self._scene_graph.update(
                        entity_embeddings.to(self._device),
                        entity_positions.to(self._device),
                    )

        # ----------------------------------------------------------------
        # Step 5: Tier 3 retrieve (core axioms)
        # ----------------------------------------------------------------
        with torch.no_grad():
            core_knowledge = self._tier3.retrieve(query_vec)

        # A2: Extract axiom anchor for field learning (used in step 7)
        axiom_anchor: torch.Tensor | None = None
        if core_knowledge.norm().item() > 1e-6:
            axiom_anchor = core_knowledge.detach().clone()

        # ----------------------------------------------------------------
        # Step 5.5: Spatial episodic retrieval (A3)
        # ----------------------------------------------------------------
        spatial_episodic_bias: dict[int, float] | None = None
        if metadata is not None:
            spatial_pos = metadata.get("spatial_position")
            if spatial_pos is not None:
                nearby_episodes = self._tier1.retrieve_by_location(
                    spatial_pos, embedding=query_vec, max_results=5,
                )
                if nearby_episodes:
                    spatial_episodic_bias = {}
                    for ep, score in nearby_episodes:
                        bid = ep.attractor_basin_id
                        if bid >= 0 and bid < self._tier2.n_active:
                            spatial_episodic_bias[bid] = (
                                spatial_episodic_bias.get(bid, 0.0) + score * 0.2
                            )

        # ----------------------------------------------------------------
        # Step 6: Tier 2 retrieve / reasoning chain via router
        # ----------------------------------------------------------------
        # Compute goal + valence + spatial reasoning bias for chain steering
        target_bias: dict[int, float] | None = None
        if self._goals is not None or self._valence is not None:
            target_bias = {}
            if self._goals is not None:
                goal_bias = self._goals.get_reasoning_bias(
                    self._tier2.keys, self._tier2.active_indices,
                )
                for basin_idx, bias_value in goal_bias.items():
                    if bias_value != 0.0:
                        target_bias[basin_idx] = bias_value
            if self._valence is not None and self._last_valence is not None:
                mode = self._valence.get_reasoning_mode(self._last_valence)
                if mode != "neutral":
                    val_bias = self._valence.get_reasoning_valence_bias(
                        mode, list(range(self._tier2.n_active)),
                    )
                    for i in range(min(val_bias.shape[0], self._tier2.n_active)):
                        if val_bias[i].item() != 0.0:
                            target_bias[i] = target_bias.get(i, 0.0) + val_bias[i].item()
            if not target_bias:
                target_bias = None

        # Merge spatial bias from scene graph
        if metadata is not None and spatial_relations:
            entity_positions = metadata.get("entity_positions")
            entity_embeddings = metadata.get("entity_embeddings")
            if entity_positions is not None and entity_embeddings is not None:
                spatial_bias = self._scene_graph.compute_spatial_bias(
                    query_vec,
                    entity_embeddings.to(self._device),
                    entity_positions.to(self._device),
                )
                if spatial_bias:
                    if target_bias is None:
                        target_bias = {}
                    for idx, val in spatial_bias.items():
                        target_bias[idx] = target_bias.get(idx, 0.0) + val

                # Inject spatial edges into transition structure
                self._scene_graph.inject_into_transitions(
                    self._transitions, spatial_relations,
                )

        # Merge spatial episodic bias into target_bias
        if spatial_episodic_bias:
            if target_bias is None:
                target_bias = {}
            for bid, val in spatial_episodic_bias.items():
                target_bias[bid] = target_bias.get(bid, 0.0) + val

        # Merge chain valence bias from previous step (F4)
        if self._last_chain_valence_bias:
            if target_bias is None:
                target_bias = {}
            for bid, val in self._last_chain_valence_bias.items():
                target_bias[bid] = target_bias.get(bid, 0.0) + val
            self._last_chain_valence_bias = {}

        reasoning_result = self._router.route(
            query_vec, target_bias=target_bias,
        )

        # A1: Extract tier2 guidance (top basin key) for field learning
        tier2_guidance: torch.Tensor | None = None
        if reasoning_result.visited_basins:
            top_basin = reasoning_result.visited_basins[0]
            if (
                top_basin >= 0
                and top_basin < self._tier2.n_active
                and self._tier2.active_mask[top_basin]
            ):
                tier2_guidance = self._tier2.keys[top_basin].detach().clone()

        # ----------------------------------------------------------------
        # Step 6.5 (NEW): Annotate visited basins + edges with valence
        # ----------------------------------------------------------------
        if self._valence is not None and reasoning_result.visited_basins:
            visited = reasoning_result.visited_basins
            for basin_id in visited:
                if basin_id >= 0 and basin_id < self._tier2.n_active:
                    basin_key = self._tier2.keys[basin_id]
                    valence_signal = self._valence.compute_valence(basin_key)
                    self._valence.annotate_basin(
                        basin_id,
                        valence_signal.scalar_valence,
                        valence_signal.valence_embedding,
                    )

            # 2E: Annotate transition edges between consecutive visited basins
            for i in range(len(visited) - 1):
                src, dst = visited[i], visited[i + 1]
                edge_idx = self._transitions._find_edge(src, dst)
                if edge_idx is not None:
                    self._valence.annotate_edge(
                        edge_idx, valence_signal.scalar_valence,
                    )

        # ----------------------------------------------------------------
        # Step 6.6 (NEW): Metacognition uncertainty estimation
        # ----------------------------------------------------------------
        _uncertainty = None
        if self._metacognition is not None:
            # Gather inputs for the 4 uncertainty sources
            r_unc = self._metacognition.estimate_retrieval_uncertainty(
                attn_weights=None,
                basin_confidence=(
                    self._tier2.confidence[reasoning_result.visited_basins[0]].item()
                    if reasoning_result.visited_basins
                    and reasoning_result.visited_basins[0] < self._tier2.n_active
                    else 0.5
                ),
                n_active=self._tier2.n_active,
            )
            c_unc = self._metacognition.estimate_chain_uncertainty(
                reasoning_result.trace,
            )
            p_unc = (
                self._metacognition.estimate_prediction_uncertainty(
                    self._last_wm_state,
                )
                if self._last_wm_state is not None
                else 0.5
            )
            k_unc = self._metacognition.estimate_knowledge_uncertainty(
                confidence=(
                    self._tier2.confidence[reasoning_result.visited_basins[0]].item()
                    if reasoning_result.visited_basins
                    and reasoning_result.visited_basins[0] < self._tier2.n_active
                    else 0.5
                ),
                maturity=min(1.0, self._step_count / 10000.0),
                modality_coverage=0.5,
            )
            _uncertainty = self._metacognition.compose_uncertainty(
                r_unc, c_unc, p_unc, k_unc, query_vec,
            )

        # Co-adaptation: extract metacognition confidence for field learning
        metacognition_confidence: float | None = None
        if _uncertainty is not None:
            metacognition_confidence = _uncertainty.scalar_confidence

        # F1: Info-seeking → exploratory goal creation
        if (
            self._metacognition is not None
            and _uncertainty is not None
            and self._goals is not None
            and self._metacognition._config.metacognition_goal_generation
        ):
            suggestions = self._metacognition.suggest_information_seeking(
                _uncertainty,
                basin_keys=self._tier2.keys,
                basin_confidence=self._tier2.confidence,
                n_active=self._tier2.n_active,
            )
            if suggestions:
                top = suggestions[0]
                explore_emb = query_vec.detach().clone()
                explore_emb = explore_emb + torch.randn_like(explore_emb) * 0.1
                self._goals.create_goal(
                    explore_emb,
                    importance=0.3,
                    urgency=0.2,
                    ttl=600.0,
                    description=f"explore:{top['source']}",
                )

        # F4: Store chain valence bias for next step's routing
        if self._valence is not None and reasoning_result.visited_basins:
            chain_val = self._valence.compute_chain_valence(
                reasoning_result.visited_basins,
            )
            if abs(chain_val) > 0.1:
                self._last_chain_valence_bias = {}
                for bid in reasoning_result.visited_basins:
                    if bid >= 0 and bid < self._tier2.n_active and chain_val < 0:
                        # Penalize negative-valence chain basins next step
                        self._last_chain_valence_bias[bid] = chain_val * 0.1

        # C: Build LoRA merge context from cognitive signals
        from sfp.core.lora import LoRAMergeContext
        merge_ctx = LoRAMergeContext()
        if self._metacognition is not None and _uncertainty is not None:
            self._uncertainty_history.append(1.0 - _uncertainty.scalar_confidence)
            merge_ctx.prediction_uncertainty_history = self._uncertainty_history
        if self._valence is not None and self._last_valence is not None:
            self._mood_history.append(self._last_valence.composite_mood)
            merge_ctx.mood_history = self._mood_history
        if self._goals is not None:
            merge_ctx.goal_progress_history = {
                g.id: g.progress_history
                for g in self._goals._goals
                if g.status.name == "ACTIVE"
            }
        self._tier0.set_merge_context(merge_ctx)

        # Co-adaptation: compute WM predicted observation for auxiliary field loss
        wm_predicted_obs: torch.Tensor | None = None
        if self._world_model is not None and self._last_wm_state is not None:
            with torch.no_grad():
                wm_latent = torch.cat([
                    self._last_wm_state.deterministic,
                    self._last_wm_state.stochastic,
                ])
                wm_predicted_obs = self._world_model.decoder(wm_latent).detach()

        # Co-adaptation: gather active goal satisfaction embeddings for field loss
        active_goal_embeddings: list[torch.Tensor] | None = None
        active_goals: list = []
        if self._goals is not None:
            active_goals = [
                g for g in self._goals._goals if g.status.name == "ACTIVE"
            ]
            if active_goals:
                active_goal_embeddings = [
                    g.satisfaction_embedding for g in active_goals
                ]

        # E1: Satisfaction hindsight training
        if (
            self._goals is not None
            and self._goals._config.satisfaction_hindsight_enabled
        ):
            for goal in active_goals:
                if goal.progress > self._goals._config.satisfaction_hindsight_threshold:
                    self._goals.train_satisfaction_hindsight(goal.id, query_vec)

        # E3: Goal deadline urgency → external LR scale
        goal_urgency_lr_scale: float = 1.0
        if (
            self._goals is not None
            and self._goals._config.goal_urgency_lr_enabled
            and active_goals
        ):
            now = time.monotonic()
            for goal in active_goals:
                if goal.deadline is not None:
                    total = goal.deadline - goal.created_at
                    remaining = goal.deadline - now
                    if total > 0 and remaining > 0:
                        remaining_ratio = remaining / total
                        if remaining_ratio < 0.5:
                            urgency_factor = 1.0 + (
                                (1.0 - remaining_ratio * 2)
                                * (self._goals._config.goal_urgency_max_multiplier - 1.0)
                            )
                            goal_urgency_lr_scale = max(
                                goal_urgency_lr_scale, urgency_factor,
                            )

        # F2: Risk tolerance → LR modulation
        valence_lr_scale: float = 1.0
        if (
            self._valence is not None
            and self._last_valence is not None
            and self._valence._config.valence_lr_modulation
        ):
            rt = self._last_valence.risk_tolerance  # [0.2, 0.8]
            lo, hi = self._valence._config.valence_lr_scale_range
            t = (rt - 0.2) / 0.6
            valence_lr_scale = lo + t * (hi - lo)

        # Combined external LR scale
        combined_lr_scale: float | None = None
        raw = valence_lr_scale * goal_urgency_lr_scale
        if raw != 1.0:
            combined_lr_scale = raw

        # 1E: Anomaly rejection scaling — reduce LR for anomalous inputs
        if is_anomalous:
            anomaly_scale = self._defense_config.anomaly_lr_scale
            if combined_lr_scale is None:
                combined_lr_scale = anomaly_scale
            else:
                combined_lr_scale *= anomaly_scale

        # Compute latent distance for dual-path surprise verification
        latent_distance: float | None = None
        if reasoning_result.visited_basins:
            nearest_basin = reasoning_result.visited_basins[0]
            if self._tier2.active_mask[nearest_basin]:
                basin_key = self._tier2.keys[nearest_basin]
                latent_distance = (query_vec - basin_key).norm().item()

        # D1: Extract salience score for gradient scaling
        salience_score_val: float | None = None
        if self._salience_gate is not None and _salience_combined is not None:
            salience_score_val = _salience_combined

        # ----------------------------------------------------------------
        # Step 7: Tier 0 forward + surprise + consistency check + gated update
        # ----------------------------------------------------------------
        field_input = (
            embedding.squeeze(0)
            if embedding.dim() == 2 and embedding.shape[0] == 1
            else embedding
        )

        # Step 7.1 (NEW): Compute WM enhanced surprise + valence threshold modifier.
        # These modulate the effective surprise score by scaling the grad norm
        # the Tier 0 processor sees via a synthetic latent distance adjustment.
        surprise_boost: float = 0.0

        if self._world_model is not None and self._last_wm_state is not None:
            wm_surprise = self._world_model.compute_enhanced_surprise(
                self._last_wm_state,
            )
            # Blend: high WM surprise → boost effective surprise
            if wm_surprise > 0.5:
                surprise_boost += wm_surprise * 0.3

        # Step 7.2 (NEW): Valence threshold modifier — high |valence| lowers
        # effective threshold (equivalent to boosting effective surprise).
        if self._valence is not None and self._last_valence is not None:
            val_modifier = self._valence.get_surprise_threshold_modifier(
                self._last_valence,
            )
            # val_modifier ∈ [0.7, 1.0]; invert so high |valence| gives a boost
            surprise_boost += (1.0 - val_modifier) * 0.5

        # F3: Vigilance → surprise threshold modulation
        if (
            self._valence is not None
            and self._last_valence is not None
            and self._valence._config.vigilance_surprise_modulation
        ):
            vigilance = self._last_valence.vigilance
            if vigilance > 0.5:
                surprise_boost += (vigilance - 0.5) * 0.6

        # Episode boundary boost (continue_head)
        if episode_boundary and self._world_model is not None:
            surprise_boost += self._world_model._config.continue_surprise_boost

        # Spatial prediction error boost
        if _spatial_surprise_boost > 0.0:
            surprise_boost += _spatial_surprise_boost

        # Lower latent_distance → higher effective surprise (Tier 0 dual-path)
        if latent_distance is not None and surprise_boost > 0.0:
            latent_distance = max(0.0, latent_distance - surprise_boost)

        metric = self._tier0.process(
            field_input,
            target=target,
            latent_distance=latent_distance,
            wm_prediction=wm_predicted_obs,
            confidence=metacognition_confidence,
            goal_embeddings=active_goal_embeddings,
            tier2_guidance=tier2_guidance,
            axiom_anchor=axiom_anchor,
            salience_score=salience_score_val,
            external_lr_scale=combined_lr_scale,
        )

        # Apply gradient clipping and update budget enforcement
        if metric.updated:
            self._gradient_clipper.clip(self._tier0.field)
            self._update_budget.enforce(self._tier0.field)

        # 2B: Hindsight label — FULL path, useful if it triggered a weight update
        if self._salience_gate is not None and _salience_combined is not None:
            self._salience_gate.train_hindsight(
                _salience_combined, metric.updated, query_vec,
            )

        # ----------------------------------------------------------------
        # Step 8: Maybe store episode in Tier 1
        # ----------------------------------------------------------------
        # Step 8.1 (NEW): Compute valence for this input
        if self._valence is not None:
            self._last_valence = self._valence.compute_valence(
                query_vec,
                reward=self._last_wm_reward,
                goal_alignment=(
                    self._goals.update_progress(
                        self._goals._goals[0].id, query_vec,
                    )
                    if self._goals is not None and self._goals._goals
                    else 0.0
                ),
            )

        if metric.updated:
            self._maybe_store_episode(
                embedding, metric, modality, reasoning_result,
                metadata=metadata,
            )
        elif (
            episode_boundary
            and self._world_model is not None
            and self._world_model._config.continue_force_store
        ):
            # Force-store on episode boundary even if surprise gate didn't fire
            self._maybe_store_episode(
                embedding, metric, modality, reasoning_result,
                metadata=metadata,
                force=True,
            )

        # Observe chain for shortcut learning
        if reasoning_result.routing == "multi_hop" and reasoning_result.n_hops > 0:
            quality = 1.0 if metric.updated else 0.5
            self._shortcut_learner.observe_chain(
                reasoning_result.visited_basins, quality,
            )

        # ----------------------------------------------------------------
        # Step 9: Maybe trigger consolidation
        # ----------------------------------------------------------------
        # (step_count already incremented at top of process())

        # Record inference for replay scheduling
        if self._replay is not None:
            self._replay.record_inference()

        mode = self._consolidation.should_consolidate(self._step_count)
        if mode is not None:
            self._consolidation.consolidate(mode, self._step_count)

            # 1D: Suppress rate limiting post-consolidation to avoid
            # replay bursts triggering defensive throttling
            if self._tier0._hardener is not None:
                cooldown = self._tier0._tier0_config.rate_limit_cooldown_steps if self._tier0._tier0_config else 20
                self._tier0._hardener.suppress_rate_limiting(cooldown)

            # Run transition learning during standard consolidation
            if mode in (ConsolidationMode.STANDARD, ConsolidationMode.DEEP):
                episodes = self._tier1.sample_for_replay(32)
                if episodes:
                    self._transition_learner.learn_from_episodes(episodes)
                self._transition_learner.learn_compositional_relations()
                self._shortcut_learner.create_shortcuts()

                # Step 9.1 (NEW): Generative replay during consolidation
                if self._replay is not None:
                    idle_secs = (
                        time.monotonic() - self._replay._last_inference_time
                    )
                    should_run, n_synth = self._replay.should_generate(
                        self._step_count,
                        self._tier1.total_count,
                        idle_secs,
                    )
                    if should_run and n_synth > 0:
                        synthetics = self._replay.generate_batch(
                            n=n_synth,
                            tier2=self._tier2,
                            transitions=self._transitions,
                            backbone=self._backbone,
                            tier3=self._tier3,
                            valence_system=self._valence,
                        )
                        # Step 9.2: Feed valid synthetics into Tier 2 with reduced weight
                        for syn in synthetics:
                            if self._tier2.n_active > 0:
                                emb = syn.embedding.to(self._device)
                                with torch.no_grad():
                                    _, basin_id, _ = self._tier2.retrieve(emb)
                                    bid = (
                                        basin_id.item()
                                        if basin_id.dim() == 0
                                        else basin_id[0].item()
                                    )
                                    if bid >= 0 and self._tier2.active_mask[bid]:
                                        # Micro-update basin key with synthetic weight
                                        old_key = self._tier2.keys[bid].clone()
                                        momentum = 1.0 - syn.weight * 0.01
                                        self._tier2.keys[bid] = (
                                            momentum * self._tier2.keys[bid]
                                            + (1.0 - momentum) * emb
                                        )
                                        # Track drift
                                        self._replay.update_drift_monitoring(
                                            bid, old_key, self._tier2.keys[bid],
                                        )

                        if synthetics:
                            logger.debug(
                                "Generative replay: %d synthetics applied",
                                len(synthetics),
                            )

                # D2: Skim buffer replay during consolidation
                if (
                    self._salience_gate is not None
                    and self._consolidation._config.skim_replay_enabled
                ):
                    self._consolidation.replay_skim_buffer(
                        self._salience_gate, self._tier0,
                    )

                # Salience gate hindsight training during consolidation
                if self._salience_gate is not None:
                    self._salience_gate.run_hindsight_training()

                # 2A: Metacognition health → consolidation action
                if self._metacognition is not None:
                    health = self._metacognition.monitor_memory_health(
                        self._tier2.keys, self._tier2.confidence,
                        self._tier2.n_active,
                    )
                    # Reduce confidence of dormant basins so consolidation can reclaim
                    for bid in health.get("dormant_basins", []):
                        if bid < self._tier2.n_active:
                            self._tier2.confidence[bid] *= 0.9
                    # Log declining basins for visibility
                    for info in health.get("declining_basins", []):
                        logger.info(
                            "Basin %d declining: conf %.3f (down %.3f)",
                            info["basin_id"], info["current"], info["decline"],
                        )

            # Run topology monitoring during deep consolidation
            if mode == ConsolidationMode.DEEP:
                alerts = self._topology_monitor.check_basin_integrity(
                    self._tier2,
                )
                alerts += self._topology_monitor.check_transition_integrity(
                    self._transitions, self._tier2,
                )
                loops = self._topology_monitor.detect_reasoning_loops(
                    self._transitions,
                )
                if alerts:
                    logger.warning("Topology alerts: %s", alerts)
                if loops:
                    logger.warning(
                        "Reasoning loops detected: %d cycles", len(loops),
                    )

                # G1/G2: Topology snapshot + structural change detection
                if self._topology_tracker is not None:
                    try:
                        self._topology_tracker.snapshot(self._tier0.field)
                        events = self._topology_tracker.detect_changes()

                        # G3: Update Betti B0 for consolidation urgency
                        if self._topology_tracker._history:
                            b0 = self._topology_tracker._history[-1].betti_numbers[0]
                            self._consolidation.set_topology_urgency(b0)

                        # G2: Significant structural change → LoRA merge
                        if events:
                            significant = [
                                e for e in events if e.significance > 0.3
                            ]
                            if significant and self._tier0.lora_manager is not None:
                                logger.info(
                                    "Topology structural change: %d significant events — merging LoRA",
                                    len(significant),
                                )
                                self._tier0.lora_manager.merge_all()
                                self._tier0.reset_optimizer()
                    except Exception:
                        pass  # giotto-tda may fail on edge cases

                # Anchor verification if configured
                if self._anchor_verifier is not None:
                    violations = self._anchor_verifier.verify(self._tier2)
                    if violations:
                        logger.warning("Anchor violations: %s", violations)

            # A4: Notify Tier 0 of newly consolidated basins
            if (
                self._consolidation._config.consolidation_notify_tier0
                and hasattr(self._consolidation, '_last_new_basin_keys')
                and self._consolidation._last_new_basin_keys
            ):
                self._tier0.register_consolidated_concepts(
                    self._consolidation._last_new_basin_keys,
                )
                self._consolidation._last_new_basin_keys = []

        # ----------------------------------------------------------------
        # Step 10 (NEW): Goal deadline/stall monitoring
        # ----------------------------------------------------------------
        if self._goals is not None:
            warnings = self._goals.check_deadlines()
            for goal_id, msg in warnings:
                logger.info("Goal %d: %s", goal_id, msg)
            stalled = self._goals.detect_stalled_goals()
            if stalled:
                logger.info("Stalled goals detected: %s", stalled)

                # E2: Goal stall → forced consolidation
                for gid in stalled:
                    goal = self._goals._find_goal(gid)
                    if goal is not None and len(goal.progress_history) >= (
                        self._consolidation._config.goal_stall_consolidation_steps
                    ):
                        logger.info(
                            "Goal %d stalled for %d+ steps — forcing consolidation",
                            gid,
                            self._consolidation._config.goal_stall_consolidation_steps,
                        )
                        self._consolidation.consolidate(
                            ConsolidationMode.STANDARD, self._step_count,
                        )
                        break  # One forced consolidation per step is enough

        return metric

    def query(self, x: torch.Tensor, return_trace: bool = False) -> ReasoningResult:
        """Query the memory system without updating weights.

        Args:
            x: Query tensor, shape (d_model,) or (B, d_model) or (B, N, d_input).
            return_trace: Whether to include reasoning trace.

        Returns:
            ReasoningResult with accumulated knowledge.
        """
        with torch.no_grad():
            # Process through Perceiver + Backbone if needed
            if x.dim() == 3:
                x = x.to(self._device)
                latents = self._perceiver(x)
                contextualized = self._backbone(latents)
                query_vec = contextualized.mean(dim=1)
            elif x.dim() == 2:
                query_vec = x.to(self._device)
            else:
                query_vec = x.unsqueeze(0).to(self._device)

            if query_vec.dim() > 1:
                query_vec = query_vec[0]

            # Pre-activation cache check (world model shortcut)
            if self._world_model is not None:
                cached_val, match_score = self._world_model.check_cache(
                    query_vec,
                )
                if cached_val is not None:
                    logger.debug(
                        "Cache hit (score=%.3f), returning cached retrieval",
                        match_score,
                    )
                    return ReasoningResult(
                        knowledge=cached_val,
                        n_hops=0,
                        visited_basins=[],
                        terminated_reason="cache_hit",
                        routing="single_hop",
                        chain_weight=0.0,
                    )

            # Route through reasoning system
            result = self._router.route(
                query_vec, return_trace=return_trace,
            )

            # Also retrieve from core memory
            core_knowledge = self._tier3.retrieve(query_vec)

            # Blend with core knowledge (core has highest priority)
            if core_knowledge.norm().item() > 1e-6:
                result = ReasoningResult(
                    knowledge=0.7 * result.knowledge + 0.3 * core_knowledge,
                    n_hops=result.n_hops,
                    visited_basins=result.visited_basins,
                    terminated_reason=result.terminated_reason,
                    routing=result.routing,
                    chain_weight=result.chain_weight,
                    trace=result.trace,
                )

        return result

    def consolidate(self, force_mode: ConsolidationMode | None = None) -> None:
        """Manually trigger consolidation.

        Args:
            force_mode: If specified, run this consolidation mode regardless of schedule.
        """
        mode = force_mode or self._consolidation.should_consolidate(self._step_count)
        if mode is not None:
            self._consolidation.consolidate(mode, self._step_count)

    def reset_session(self) -> None:
        """Reset Tier 0 working memory (session volatility)."""
        self._tier0.reset_working_memory()
        logger.info("Session reset: Tier 0 working memory cleared")

    def set_anchor_verifier(self, verifier: AnchorVerifier) -> None:
        """Set the anchor verifier for periodic integrity checks."""
        self._anchor_verifier = verifier

    def health_report(self) -> dict:
        """Generate a comprehensive health report across all tiers and modules."""
        report = {
            "step_count": self._step_count,
            "tier0": {
                "param_count": self._tier0.field.param_count,
                "surprise_history_len": len(self._tier0.surprise_history),
            },
            "tier1": {
                "hot_count": self._tier1.hot_count,
                "cold_count": self._tier1.cold_count,
                "total_count": self._tier1.total_count,
                "spatial_count": sum(
                    1 for ep in self._tier1._hot + self._tier1._cold
                    if ep.spatial_position is not None
                ),
                "basin_distribution": self._tier1.basin_distribution,
            },
            "tier2": {
                "n_active": self._tier2.n_active,
            },
            "tier3": {
                "n_active": self._tier3.n_active,
                "integrity_failures": len(self._tier3.verify_integrity()),
            },
            "reasoning": {
                "n_edges": self._transitions.n_active_edges,
            },
            "consolidation": self._consolidation.stats,
        }

        # Cognitive module health (included when module is enabled)
        if self._world_model is not None:
            wm_info: dict = {"enabled": True}
            if self._last_wm_state is not None:
                wm_info["prediction_error"] = self._last_wm_state.prediction_error
                wm_info["kl_divergence"] = self._last_wm_state.kl_divergence
                wm_info["reconstruction_error"] = self._last_wm_state.reconstruction_error
            wm_info["spatial_loss"] = self._world_model._spatial_loss_ema
            wm_info["continue_probability"] = self._last_continue_prob
            wm_info["spatial_prediction_error_ema"] = self._spatial_prediction_error_ema
            report["world_model"] = wm_info

        # Scene graph
        if hasattr(self, "_scene_graph"):
            sg = self._scene_graph
            n_nodes = sg._prev_positions.shape[0] if sg._prev_positions is not None else 0
            report["scene_graph"] = {"n_nodes": n_nodes}
        if self._goals is not None:
            active = [g for g in self._goals._goals if g.status.name == "ACTIVE"]
            report["goals"] = {
                "active_count": len(active),
                "total_count": len(self._goals._goals),
            }
        if self._metacognition is not None:
            report["metacognition"] = {
                "ece": self._metacognition.get_ece(),
            }
        if self._valence is not None:
            val_info: dict = {"enabled": True}
            if self._last_valence is not None:
                val_info["composite_mood"] = self._last_valence.composite_mood
                val_info["risk_tolerance"] = self._last_valence.risk_tolerance
                val_info["vigilance"] = self._last_valence.vigilance
            report["valence"] = val_info
        if self._replay is not None:
            report["replay"] = self._replay.get_generation_stats()

        return report

    def memory_footprint(self) -> dict[str, int]:
        """Estimate VRAM usage per component in bytes."""

        def _module_bytes(m: torch.nn.Module) -> int:
            return sum(p.numel() * p.element_size() for p in m.parameters()) + sum(
                b.numel() * b.element_size() for b in m.buffers()
            )

        footprint = {
            "perceiver": _module_bytes(self._perceiver),
            "backbone": _module_bytes(self._backbone),
            "tier0_field": _module_bytes(self._tier0.field),
            "tier2_essential": _module_bytes(self._tier2),
            "tier3_core": _module_bytes(self._tier3),
            "transitions": _module_bytes(self._transitions),
        }

        # Cognitive modules
        if self._world_model is not None:
            footprint["world_model"] = _module_bytes(self._world_model)
        if self._goals is not None:
            footprint["goals"] = _module_bytes(self._goals)
        if self._metacognition is not None:
            footprint["metacognition"] = _module_bytes(self._metacognition)
        if self._valence is not None:
            footprint["valence"] = _module_bytes(self._valence)
        if self._salience_gate is not None:
            footprint["salience_gate"] = _module_bytes(self._salience_gate)

        footprint["total"] = sum(footprint.values())
        return footprint

    def _maybe_store_episode(
        self,
        embedding: torch.Tensor,
        metric: SurpriseMetric,
        modality: str,
        reasoning_result: ReasoningResult,
        *,
        metadata: dict | None = None,
        force: bool = False,
    ) -> None:
        """Create and attempt to store an episode in Tier 1.

        Args:
            force: If True, use force_store (bypasses surprise threshold
                but still checks dedup and integrity).  Used for episode
                boundary storage.
        """
        field = self._tier0.field
        weight_summary = field.get_weight_summary()
        weight_hash = compute_weight_hash(field)

        emb = embedding.squeeze(0) if embedding.dim() == 2 and embedding.shape[0] == 1 else embedding
        if emb.dim() > 1:
            emb = emb[0]

        logit_snapshot = field(emb).detach()
        integrity_hash = compute_episode_hash(emb.detach(), logit_snapshot, weight_hash)

        basin_id = reasoning_result.visited_basins[0] if reasoning_result.visited_basins else -1
        basin_distance = 0.0
        if basin_id >= 0 and self._tier2.active_mask[basin_id]:
            basin_distance = (emb - self._tier2.keys[basin_id]).norm().item()

        # Extract spatial metadata from bridge (generic — any bridge can provide these)
        spatial_position = None
        spatial_orientation = None
        if metadata is not None:
            spatial_position = metadata.get("spatial_position")
            spatial_orientation = metadata.get("spatial_orientation")

        episode = Episode(
            id=self._tier1.allocate_id(),
            timestamp=time.monotonic(),
            modality=modality,
            provenance_hash=weight_hash[:16],
            input_embedding=emb.detach().cpu(),
            working_memory_state=weight_summary.detach().cpu(),
            logit_snapshot=logit_snapshot.detach().cpu(),
            surprise_at_storage=metric.grad_norm,
            attractor_basin_id=basin_id,
            attractor_distance=basin_distance,
            preceding_episode_id=self._tier1.last_episode_id,
            following_episode_id=None,
            integrity_hash=integrity_hash,
            weight_hash_at_storage=weight_hash,
            spatial_position=spatial_position,
            spatial_orientation=spatial_orientation,
        )

        if force:
            self._tier1.force_store(episode)
        else:
            self._tier1.maybe_store(episode)

    # --- Internal helpers ---

    @staticmethod
    def _pool_to_vector(x: torch.Tensor) -> torch.Tensor:
        """Pool an input tensor of any shape to a single 1-D vector."""
        if x.dim() == 1:
            return x
        if x.dim() == 2:
            return x.mean(dim=0) if x.shape[0] > 1 else x.squeeze(0)
        if x.dim() == 3:
            return x.mean(dim=(0, 1))
        return x.flatten()

    # --- Properties for direct tier access ---

    @property
    def tier0(self) -> StreamingProcessor:
        return self._tier0

    @property
    def tier1(self) -> EpisodicMemory:
        return self._tier1

    @property
    def tier2(self) -> EssentialMemory:
        return self._tier2

    @property
    def tier3(self) -> CoreMemory:
        return self._tier3

    @property
    def transitions(self) -> TransitionStructure:
        return self._transitions

    @property
    def perceiver(self) -> PerceiverIO:
        return self._perceiver

    @property
    def backbone(self) -> BackboneTransformer:
        return self._backbone

    @property
    def event_emitter(self) -> PromotionEventEmitter:
        return self._event_emitter

    # --- Cognitive module properties ---

    @property
    def world_model(self):
        """PredictiveWorldModel or None if disabled."""
        return self._world_model

    @property
    def goals(self):
        """GoalRegister or None if disabled."""
        return self._goals

    @property
    def metacognition(self):
        """MetacognitionEngine or None if disabled."""
        return self._metacognition

    @property
    def valence(self):
        """ValenceSystem or None if disabled."""
        return self._valence

    @property
    def salience_gate(self):
        """SalienceGate or None if disabled."""
        return self._salience_gate

    @property
    def replay(self):
        """GenerativeReplay or None if disabled."""
        return self._replay
