"""Configuration dataclasses for all SFP components."""

from __future__ import annotations

from dataclasses import dataclass

from sfp.types import CommLayer, FieldSize

_SIZE_MAP: dict[FieldSize, tuple[int, int]] = {
    FieldSize.TINY: (4, 256),
    FieldSize.SMALL: (6, 512),
    FieldSize.MEDIUM: (8, 1024),
    FieldSize.LARGE: (8, 2048),
}


@dataclass(frozen=True)
class FieldConfig:
    """Configuration for the SemanticFieldProcessor MLP manifold."""

    dim: int = 512
    n_layers: int = 6
    activation: str = "gelu"
    use_layernorm: bool = True
    residual: bool = False

    @classmethod
    def from_preset(cls, size: FieldSize) -> FieldConfig:
        n_layers, dim = _SIZE_MAP[size]
        return cls(dim=dim, n_layers=n_layers)


@dataclass(frozen=True)
class StreamingConfig:
    """Configuration for the StreamingProcessor surprise-gated updates."""

    lr: float = 1e-4
    weight_decay: float = 0.001
    surprise_threshold: float = 0.1
    momentum: float = 0.9
    adaptive_surprise: bool = True
    surprise_percentile: float = 0.9
    loss_fn: str = "mse"
    # Co-adaptation: world model prediction error as auxiliary field loss
    auxiliary_loss_weight: float = 0.1
    # Co-adaptation: metacognition confidence gates field update magnitude
    confidence_modulation_enabled: bool = True
    confidence_low_threshold: float = 0.5
    confidence_high_threshold: float = 0.8
    # Co-adaptation: goal satisfaction regularization in field loss
    goal_loss_weight: float = 0.05
    # Memory tier feedback: Tier 2 basin key as field guidance loss
    tier2_guidance_weight: float = 0.05
    # Memory tier feedback: Tier 3 axiom as field anchor loss
    axiom_anchor_weight: float = 0.02
    # Learning dynamics: LR warmup
    warmup_steps: int = 100
    warmup_start_factor: float = 0.1
    # Salience gate modulates gradient magnitude
    salience_gradient_scaling: bool = True
    # Soft surprise gate (replaces binary threshold with sigmoid)
    soft_gate_enabled: bool = True
    soft_gate_steepness: float = 10.0
    soft_gate_floor: float = 0.05
    # Composite importance weights (shift sigmoid center)
    importance_consistency_weight: float = 0.4
    importance_confidence_weight: float = 0.3
    importance_surprise_weight: float = 0.3
    # Post-warmup LR decay (cosine annealing)
    lr_decay_enabled: bool = True
    lr_decay_steps: int = 50000
    lr_decay_min_factor: float = 0.01
    # External LR scale bounds
    external_lr_scale_range: tuple[float, float] = (0.1, 5.0)
    # Gradient accumulation
    gradient_accumulation_steps: int = 1
    # Gradient conflict detection
    gradient_conflict_detection: bool = True
    gradient_conflict_ema_decay: float = 0.95
    gradient_conflict_warn_ratio: float = 0.5
    # Gradient conflict mitigation (adaptive loss scaling)
    gradient_conflict_mitigation: bool = True
    gradient_conflict_damping: float = 0.8
    gradient_conflict_score_ema_decay: float = 0.95
    gradient_conflict_score_threshold: float = 0.3
    gradient_conflict_weight_floor: float = 0.1
    # Curriculum / difficulty scheduling
    curriculum_enabled: bool = False
    curriculum_competence_ema_decay: float = 0.99
    curriculum_too_easy_threshold: float = 0.2
    curriculum_too_easy_scale: float = 0.3
    curriculum_too_hard_ratio: float = 5.0
    curriculum_too_hard_scale: float = 0.3
    curriculum_warmup_steps: int = 100


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for Online-LoRA adapters."""

    rank: int = 4
    alpha: float = 1.0
    merge_threshold: float = 0.5
    enabled: bool = True
    # Multi-signal merge triggers
    uncertainty_merge_threshold: float = 0.7
    uncertainty_merge_window: int = 50
    mood_merge_threshold: float = -0.3
    mood_merge_window: int = 50
    goal_stall_merge_steps: int = 100


@dataclass(frozen=True)
class EWCConfig:
    """Configuration for Elastic Weight Consolidation."""

    lambda_: float = 100.0
    decay: float = 0.999
    enabled: bool = True


@dataclass(frozen=True)
class AttractorConfig:
    """Configuration for attractor-based querying."""

    max_iterations: int = 20
    step_size: float = 0.7
    tolerance: float = 1e-4
    return_trajectory: bool = False


@dataclass(frozen=True)
class QuantizationConfig:
    """Configuration for weight quantization."""

    weight_bits: int = 8
    activation_bits: int = 16
    per_channel: bool = True


@dataclass(frozen=True)
class CommConfig:
    """Configuration for the communication protocol."""

    preferred_layer: CommLayer = CommLayer.L2_MANIFOLD_COORD
    surprise_threshold: float = 0.1
    compression_method: str = "topk"
    compression_density: float = 0.01


# ---------------------------------------------------------------------------
# Hardware & neural architecture configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """Hardware-specific VRAM and latency budget constraints."""

    name: str = "rtx3060"
    vram_budget_mb: int = 12_000
    target_latency_ms: float = 100.0
    fp16_capable: bool = True


@dataclass(frozen=True)
class PerceiverConfig:
    """Configuration for the Perceiver IO multi-modal bottleneck."""

    n_latents: int = 256
    d_latent: int = 512
    d_input: int = 512
    n_cross_attn_layers: int = 2
    n_self_attn_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0


@dataclass(frozen=True)
class BackboneConfig:
    """Configuration for the shared backbone transformer."""

    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1


# ---------------------------------------------------------------------------
# Hierarchical memory configs (Tiers 0-3 + consolidation)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tier0Config:
    """Working memory (Titans-style LMM) hardening parameters."""

    surprise_momentum: float = 0.9
    max_surprise_ratio: float = 3.0
    rate_limit_threshold: float = 0.3
    rate_limit_window: int = 100
    clip_multiplier: float = 2.0
    consolidation_threshold: float = 1.5
    rate_limit_cooldown_steps: int = 20


@dataclass(frozen=True)
class Tier1Config:
    """Episodic memory buffer configuration."""

    hot_capacity: int = 2000
    cold_capacity: int = 8000
    surprise_threshold: float = 0.5
    dedup_threshold: float = 0.95
    min_per_basin: int = 5
    eviction_batch_size: int = 100


@dataclass(frozen=True)
class Tier2Config:
    """Essential memory (Hopfield retrieval) configuration."""

    n_slots: int = 4096
    d_value: int = 512
    n_heads: int = 8
    temperature: float = 1.0
    consistency_threshold: float = 0.3


@dataclass(frozen=True)
class Tier3Config:
    """Core/axiomatic memory configuration."""

    n_slots: int = 512
    d_value: int = 512
    min_confidence: float = 0.9
    min_episode_count: int = 1000
    min_modalities: int = 2
    min_age_days: float = 7.0


@dataclass(frozen=True)
class ConsolidationConfig:
    """Consolidation engine scheduling and thresholds."""

    mini_interval: int = 100
    standard_interval: int = 1000
    deep_interval: int = 10000
    replay_batch_size: int = 32
    n_replay_steps: int = 100
    representation_threshold: float = 0.1
    new_concept_threshold: int = 5
    distillation_threshold: float = 0.5
    # Co-adaptation: replay episodes through field during consolidation
    replay_through_field_enabled: bool = True
    replay_through_field_batch_size: int = 8
    replay_lr_scale: float = 0.5
    # Co-adaptation: bias consolidation sampling toward high-|valence| basins
    valence_weighted_sampling: bool = True
    # Notify Tier 0 of newly consolidated basins
    consolidation_notify_tier0: bool = True
    # Skim buffer replay during consolidation
    skim_replay_enabled: bool = True
    skim_replay_lr_scale: float = 0.25
    # Goal stall triggers forced consolidation
    goal_stall_consolidation_steps: int = 200
    # Topology-informed consolidation urgency
    betti_b0_consolidation_threshold: int = 20
    betti_consolidation_interval_reduction: float = 0.5


# ---------------------------------------------------------------------------
# Reasoning chain configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionConfig:
    """Transition structure (sparse concept graph) configuration."""

    d_relation: int = 64
    max_edges: int = 40960
    n_relation_types: int = 10


@dataclass(frozen=True)
class ReasoningChainConfig:
    """Associative reasoning chain traversal configuration."""

    max_hops: int = 7
    convergence_threshold: float = 0.01
    context_decay: float = 0.85
    query_retention: float = 0.3
    branch_threshold: float = 0.4
    max_branches: int = 3
    shortcut_min_traversals: int = 10


# ---------------------------------------------------------------------------
# Defense config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefenseConfig:
    """Poisoning defense configuration."""

    security_level: str = "normal"  # "high", "normal", "accelerated"
    embedding_anomaly_threshold: float = 3.0
    dual_path_verification: bool = True
    topology_check_interval: int = 1000
    anchor_verification: bool = True
    merge_alarm_threshold: int = 5
    topology_change_threshold: int = 3
    anomaly_lr_scale: float = 0.1


# ---------------------------------------------------------------------------
# Cognitive module configs (world model, goals, metacognition, valence,
# selective attention, generative replay)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for the RSSM-based predictive world model."""

    d_deterministic: int = 512
    d_stochastic_categories: int = 32
    d_stochastic_classes: int = 32
    d_observation: int = 512
    d_reward: int = 1
    n_projection_steps: int = 4
    kl_free_nats: float = 1.0
    kl_weight: float = 0.1
    reconstruction_weight: float = 1.0
    prediction_error_weight: float = 0.4
    kl_divergence_weight: float = 0.3
    reconstruction_error_weight: float = 0.3
    n_subspace_projections: int = 8
    cache_size: int = 8
    cache_match_threshold: float = 0.8
    symlog_epsilon: float = 1e-3
    # Configurable gradient clipping (was hardcoded 100.0)
    grad_clip_norm: float = 10.0
    # continue_head consumption (episode boundary detection)
    continue_threshold: float = 0.3
    continue_force_store: bool = True
    continue_surprise_boost: float = 0.3
    # spatial_predictor consumption (spatial prediction error as surprise)
    spatial_surprise_enabled: bool = True
    spatial_surprise_weight: float = 0.2
    spatial_surprise_ema_decay: float = 0.99


@dataclass(frozen=True)
class GoalPersistenceConfig:
    """Configuration for the goal persistence register."""

    max_goals: int = 32
    d_goal: int = 512
    d_satisfaction: int = 512
    max_subgoals: int = 4
    progress_ema_decay: float = 0.95
    satisfaction_threshold: float = 0.8
    stall_threshold: float = 0.01
    stall_steps: int = 50
    deadline_warning_ratio: float = 0.8
    reasoning_bias: float = 0.3
    salience_boost: float = 0.2
    ttl_default: float = 3600.0
    priority_mlp_hidden: int = 64
    # Satisfaction hindsight training
    satisfaction_hindsight_enabled: bool = True
    satisfaction_hindsight_threshold: float = 0.8
    # Goal urgency LR scaling
    goal_urgency_lr_enabled: bool = True
    goal_urgency_max_multiplier: float = 2.0


@dataclass(frozen=True)
class MetacognitionConfig:
    """Configuration for the metacognition and uncertainty system."""

    d_uncertainty_embedding: int = 64
    n_calibration_bins: int = 10
    estimator_hidden: int = 32
    confidence_threshold_high: float = 0.8
    confidence_threshold_low: float = 0.3
    seeking_max_alternatives: int = 3
    health_dormant_threshold: float = 0.01
    health_decline_window: int = 100
    # Auto-create exploratory goals from info-seeking suggestions
    metacognition_goal_generation: bool = True


@dataclass(frozen=True)
class ValenceConfig:
    """Configuration for the valence and affective state system."""

    d_valence_embedding: int = 32
    rl_value_weight: float = 0.5
    user_feedback_weight: float = 0.25
    goal_alignment_weight: float = 0.15
    prediction_satisfaction_weight: float = 0.10
    learned_blend: float = 0.4
    immediate_tau: float = 0.5
    short_term_tau: float = 0.99
    baseline_tau: float = 0.9999
    mood_weights: tuple[float, float, float] = (0.3, 0.5, 0.2)
    approach_weight: float = 0.2
    avoidance_weight: float = 0.4
    basin_valence_ema_decay: float = 0.95
    edge_valence_ema_decay: float = 0.95
    # Risk tolerance modulates field LR
    valence_lr_modulation: bool = True
    valence_lr_scale_range: tuple[float, float] = (0.5, 1.5)
    # Vigilance modulates surprise threshold
    vigilance_surprise_modulation: bool = True


@dataclass(frozen=True)
class SelectiveAttentionConfig:
    """Configuration for the selective attention / salience gate."""

    n_modalities: int = 5
    modality_names: tuple[str, ...] = (
        "text",
        "vision",
        "audio",
        "sensor",
        "structured",
    )
    skip_threshold: float = 0.1
    skim_threshold: float = 0.4
    d_salience: int = 64
    d_context: int = 512
    skim_buffer_size: int = 100
    skim_lr_scale: float = 0.01
    threshold_ema_decay: float = 0.99
    alarm_basin_threshold: float = 0.9
    accumulation_window: int = 5
    accumulation_threshold: float = 0.3
    cross_modal_threshold: float = 0.5
    hindsight_buffer_size: int = 500
    hindsight_lr: float = 1e-3


@dataclass(frozen=True)
class GenerativeReplayConfig:
    """Configuration for the generative replay module."""

    real_to_synthetic_ratio: float = 3.0
    max_synthetics_per_cycle: int = 16
    synthetic_weight_min: float = 0.2
    synthetic_weight_max: float = 0.5
    interpolation_alpha_min: float = 0.2
    interpolation_alpha_max: float = 0.8
    basin_similarity_min: float = 0.2
    basin_similarity_max: float = 0.7
    manifold_proximity_threshold: float = 3.0
    diversity_threshold: float = 0.7
    drift_ema_decay: float = 0.95
    drift_throttle_multiplier: float = 2.0
    warmup_episodes: int = 1000
    middle_episodes: int = 10000
    middle_cycle_interval: int = 5
    middle_synthetics: int = 8
    mature_cycle_interval: int = 3
    mature_synthetics: int = 16
    idle_timeout_seconds: float = 30.0
    backbone_coherence_threshold: float = 0.5
