"""Hardware profiles and VRAM budget estimation.

Provides presets for common consumer GPUs and functions to recommend
configurations and estimate memory usage for the hierarchical memory system.
"""

from __future__ import annotations

from sfp.config import (
    BackboneConfig,
    ConsolidationConfig,
    DefenseConfig,
    EWCConfig,
    FieldConfig,
    FieldSize,
    GenerativeReplayConfig,
    GoalPersistenceConfig,
    HardwareProfile,
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


class HardwareProfiles:
    """Predefined hardware profiles for common consumer GPUs."""

    RTX_3060 = HardwareProfile(
        name="rtx3060",
        vram_budget_mb=12_000,
        target_latency_ms=100.0,
        fp16_capable=True,
    )
    RTX_4090 = HardwareProfile(
        name="rtx4090",
        vram_budget_mb=24_000,
        target_latency_ms=50.0,
        fp16_capable=True,
    )
    M4_MAX = HardwareProfile(
        name="m4_max",
        vram_budget_mb=128_000,
        target_latency_ms=200.0,
        fp16_capable=True,
    )
    CPU_ONLY = HardwareProfile(
        name="cpu",
        vram_budget_mb=0,
        target_latency_ms=500.0,
        fp16_capable=False,
    )


def recommend_config(profile: HardwareProfile) -> dict:
    """Recommend a full configuration based on hardware profile.

    Returns a dict of config objects suitable for passing to
    HierarchicalMemoryProcessor or create_field(hierarchical=True).

    Args:
        profile: Hardware profile to optimize for.

    Returns:
        Dict mapping config names to config objects.
    """
    if profile.vram_budget_mb >= 24_000:
        # RTX 4090 or better: can afford larger models
        field_size = FieldSize.MEDIUM
        field_config = FieldConfig.from_preset(field_size)
        d = field_config.dim  # 1024
        n_latents = 512
        backbone_layers = 12
        t2_slots = 8192
        t3_slots = 1024
        max_edges = 81920
        hot_capacity = 4000
        cold_capacity = 16000
    elif profile.vram_budget_mb >= 12_000:
        # RTX 3060: standard config
        field_size = FieldSize.SMALL
        field_config = FieldConfig.from_preset(field_size)
        d = field_config.dim  # 512
        n_latents = 256
        backbone_layers = 8
        t2_slots = 4096
        t3_slots = 512
        max_edges = 40960
        hot_capacity = 2000
        cold_capacity = 8000
    elif profile.vram_budget_mb >= 4_000:
        # Low-VRAM GPU
        field_size = FieldSize.TINY
        field_config = FieldConfig.from_preset(field_size)
        d = field_config.dim  # 256
        n_latents = 128
        backbone_layers = 4
        t2_slots = 1024
        t3_slots = 128
        max_edges = 10240
        hot_capacity = 500
        cold_capacity = 2000
    else:
        # CPU only: minimal config
        field_size = FieldSize.TINY
        field_config = FieldConfig.from_preset(field_size)
        d = field_config.dim  # 256
        n_latents = 64
        backbone_layers = 2
        t2_slots = 512
        t3_slots = 64
        max_edges = 5120
        hot_capacity = 200
        cold_capacity = 800

    result = {
        "field_config": field_config,
        "perceiver_config": PerceiverConfig(d_input=d, d_latent=d, n_latents=n_latents),
        "backbone_config": BackboneConfig(d_model=d, n_layers=backbone_layers),
        "tier0_config": Tier0Config(),
        "tier1_config": Tier1Config(hot_capacity=hot_capacity, cold_capacity=cold_capacity),
        "tier2_config": Tier2Config(n_slots=t2_slots, d_value=d),
        "tier3_config": Tier3Config(n_slots=t3_slots, d_value=d),
        "consolidation_config": ConsolidationConfig(),
        "transition_config": TransitionConfig(max_edges=max_edges),
        "reasoning_config": ReasoningChainConfig(),
        "defense_config": DefenseConfig(),
        "streaming_config": StreamingConfig(),
        "lora_config": LoRAConfig(enabled=True),
        "ewc_config": EWCConfig(enabled=True),
    }

    # Cognitive modules: RTX 4090+ enables all 6; RTX 3060 enables world model + goals
    if profile.vram_budget_mb >= 24_000:
        result["world_model_config"] = WorldModelConfig(d_observation=d)
        result["goal_config"] = GoalPersistenceConfig(d_goal=d, d_satisfaction=d)
        result["metacognition_config"] = MetacognitionConfig()
        result["valence_config"] = ValenceConfig()
        result["attention_config"] = SelectiveAttentionConfig()
        result["replay_config"] = GenerativeReplayConfig()
    elif profile.vram_budget_mb >= 12_000:
        result["world_model_config"] = WorldModelConfig(d_observation=d)
        result["goal_config"] = GoalPersistenceConfig(d_goal=d, d_satisfaction=d)

    return result


def estimate_vram(
    field_config: FieldConfig | None = None,
    perceiver_config: PerceiverConfig | None = None,
    backbone_config: BackboneConfig | None = None,
    tier2_config: Tier2Config | None = None,
    tier3_config: Tier3Config | None = None,
    tier1_config: Tier1Config | None = None,
    transition_config: TransitionConfig | None = None,
    world_model_config: WorldModelConfig | None = None,
    goal_config: GoalPersistenceConfig | None = None,
    metacognition_config: MetacognitionConfig | None = None,
    valence_config: ValenceConfig | None = None,
    attention_config: SelectiveAttentionConfig | None = None,
    replay_config: GenerativeReplayConfig | None = None,
    dtype_bytes: int = 4,
) -> dict[str, int]:
    """Estimate VRAM usage per component in bytes.

    This is a static estimate based on parameter counts, not actual allocation.

    Args:
        field_config: MLP manifold config.
        perceiver_config: Perceiver IO config.
        backbone_config: Backbone transformer config.
        tier2_config: Tier 2 essential memory config.
        tier3_config: Tier 3 core memory config.
        tier1_config: Tier 1 episodic memory config.
        transition_config: Transition structure config.
        world_model_config: Predictive world model config (optional).
        goal_config: Goal persistence config (optional).
        metacognition_config: Metacognition engine config (optional).
        valence_config: Valence system config (optional).
        attention_config: Selective attention / salience gate config (optional).
        replay_config: Generative replay config (optional).
        dtype_bytes: Bytes per element (4 for FP32, 2 for FP16).

    Returns:
        Dict mapping component names to estimated bytes.
    """
    fc = field_config or FieldConfig()
    pc = perceiver_config or PerceiverConfig()
    bc = backbone_config or BackboneConfig()
    t2c = tier2_config or Tier2Config()
    t3c = tier3_config or Tier3Config()
    t1c = tier1_config or Tier1Config()
    tc = transition_config or TransitionConfig()

    d = fc.dim

    # Perceiver IO: latents + input_proj + cross_attn layers + self_attn layers
    perceiver_params = (
        pc.n_latents * pc.d_latent  # latent array
        + pc.d_input * pc.d_latent  # input projection
        + pc.n_cross_attn_layers * (4 * pc.d_latent * pc.d_latent + 2 * pc.d_latent)  # cross-attn
        + pc.n_self_attn_layers * (4 * pc.d_latent * pc.d_latent + 2 * 4 * pc.d_latent * pc.d_latent)  # self-attn + FFN
    )

    # Backbone: n_layers * (self_attn + SwiGLU FFN + norms)
    backbone_params = bc.n_layers * (
        4 * bc.d_model * bc.d_model  # self-attn (Q, K, V, O projections)
        + 3 * bc.d_model * bc.d_ff  # SwiGLU (gate, up, down)
        + 4 * bc.d_model  # RMSNorm weights (2 per layer)
    )

    # Tier 0: MLP manifold
    tier0_params = fc.n_layers * d * d + fc.n_layers * d  # weights + biases

    # Tier 1: Hot buffer (embedding + working_state + logits per episode)
    # Approximate working_state as d * n_layers * 2 (mean + std per layer)
    tier1_bytes = t1c.hot_capacity * (d + fc.n_layers * 2 + d) * dtype_bytes

    # Tier 2: keys + values + projection layers + metadata buffers
    tier2_params = (
        t2c.n_slots * d  # keys
        + t2c.n_slots * t2c.d_value  # values
        + 4 * d * d  # projection layers (Q, K, V, O)
    )
    tier2_metadata = t2c.n_slots * (4 + 8 + 8 + 8 + 8 + 4 + 1)  # confidence, counts, timestamps, etc.

    # Tier 3: keys + values + metadata
    tier3_params = t3c.n_slots * d + t3c.n_slots * t3c.d_value
    tier3_metadata = t3c.n_slots * (4 + 8 + 8 + 1 + 8)

    # Transition structure: edge storage + relation embeddings + scoring network
    transition_params = (
        tc.max_edges * tc.d_relation  # per-edge embeddings
        + tc.n_relation_types * tc.d_relation  # prototypes
        + (d + tc.d_relation) * 1  # scoring linear
    )
    transition_buffers = tc.max_edges * (8 + 8 + 4 + 4 + 8 + 8 + 1)  # source, target, weight, conf, etc.

    # Optimizer states (AdamW): 2x parameter size for first and second moments
    optimizer_bytes = tier0_params * dtype_bytes * 2

    components = {
        "perceiver": perceiver_params * dtype_bytes,
        "backbone": backbone_params * dtype_bytes,
        "tier0_field": tier0_params * dtype_bytes,
        "tier1_hot_buffer": tier1_bytes,
        "tier2_params": tier2_params * dtype_bytes,
        "tier2_metadata": tier2_metadata,
        "tier3_params": tier3_params * dtype_bytes,
        "tier3_metadata": tier3_metadata,
        "transitions_params": transition_params * dtype_bytes,
        "transitions_buffers": transition_buffers,
        "optimizer_states": optimizer_bytes,
    }

    # --- Cognitive modules (only included when config is provided) ---

    if world_model_config is not None:
        wm = world_model_config
        d_stoch = wm.d_stochastic_categories * wm.d_stochastic_classes
        # GRU + prior + posterior + decoder + reward + continue + subspace + cache + optimizer
        wm_params = (
            3 * (d_stoch + wm.d_observation) * wm.d_deterministic  # GRU (3 gates)
            + 3 * wm.d_deterministic  # GRU bias
            + wm.d_deterministic * wm.d_deterministic + wm.d_deterministic * d_stoch  # prior
            + (wm.d_deterministic + wm.d_observation) * wm.d_deterministic + wm.d_deterministic * d_stoch  # posterior
            + (wm.d_deterministic + d_stoch) * wm.d_deterministic + wm.d_deterministic * wm.d_observation  # decoder
            + (wm.d_deterministic + d_stoch) * 256 + 256  # reward head
            + (wm.d_deterministic + d_stoch) * 256 + 256  # continue head
            + wm.n_subspace_projections * wm.d_observation * (wm.d_observation // wm.n_subspace_projections)  # subspace
        )
        wm_optimizer = wm_params * dtype_bytes * 2  # Adam moments
        components["world_model"] = wm_params * dtype_bytes + wm_optimizer

    if goal_config is not None:
        gc = goal_config
        # Encoders + decomposition + priority MLP
        goal_params = (
            gc.d_goal * gc.d_goal * 2 + gc.d_goal * 2  # goal encoder
            + gc.d_satisfaction * gc.d_satisfaction * 2 + gc.d_satisfaction * 2  # satisfaction encoder
            + gc.d_goal * gc.d_goal * 2 * gc.max_subgoals  # decomposition
            + (gc.d_goal + 4) * gc.priority_mlp_hidden + gc.priority_mlp_hidden  # priority MLP
        )
        components["goals"] = goal_params * dtype_bytes

    if metacognition_config is not None:
        mc = metacognition_config
        # 4 estimators + composer + scalar head + calibration buffers
        meta_params = (
            4 * (3 * mc.estimator_hidden + mc.estimator_hidden + mc.estimator_hidden * 1 + 1)  # 4 estimators
            + (4 + d) * 128 + 128 + 128 * mc.d_uncertainty_embedding  # composer
            + mc.d_uncertainty_embedding * 32 + 32 + 32 * 1 + 1  # scalar head
        )
        meta_buffers = mc.n_calibration_bins * 2 * 8  # calibration counts
        components["metacognition"] = meta_params * dtype_bytes + meta_buffers

    if valence_config is not None:
        vc = valence_config
        # Projections + learned valence + basin/edge buffers
        val_params = (
            d * vc.d_valence_embedding  # valence_proj
            + vc.d_valence_embedding * d  # context_proj
            + d * 128 + 128 + 128 * 1 + 1  # learned_valence
        )
        val_buffers = (
            4096 * dtype_bytes  # basin_valence_scalar
            + 4096 * vc.d_valence_embedding * dtype_bytes  # basin_valence_embedding
            + 4096 * 8  # basin_valence_count
            + 40960 * dtype_bytes  # edge_valence
            + 40960 * 8  # edge_valence_count
        )
        components["valence"] = val_params * dtype_bytes + val_buffers

    if attention_config is not None:
        ac = attention_config
        # Modality estimators + change detectors + context agg + combiner
        attn_params = (
            ac.n_modalities * (d * ac.d_salience + ac.d_salience + ac.d_salience * 1 + 1)  # estimators
            + ac.n_modalities * (d * 2 * ac.d_salience + ac.d_salience + ac.d_salience * 1 + 1)  # change detectors
            + d * 3 * ac.d_context + ac.d_context  # context agg
            + (3 + ac.d_context) * 64 + 64 + 64 * 1 + 1  # combiner
        )
        components["salience_gate"] = attn_params * dtype_bytes

    if replay_config is not None:
        # Zero persistent params; ~64 KB drift buffers + ~7 MB transient
        components["replay"] = 64 * 1024  # drift monitoring buffers only

    components["total"] = sum(components.values())

    return components
