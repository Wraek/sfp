# SFP Feature Reference

Complete reference for all configurable features in the Semantic Field Processor framework. Features are grouped by the subsystem they belong to and the config dataclass that controls them.

---

## Table of Contents

- [Surprise Gating (StreamingConfig)](#surprise-gating)
- [Co-Adaptation Losses (StreamingConfig)](#co-adaptation-losses)
- [Gradient Modulation (StreamingConfig)](#gradient-modulation)
- [Memory Tier Feedback (StreamingConfig)](#memory-tier-feedback)
- [LoRA Multi-Signal Merge (LoRAConfig)](#lora-multi-signal-merge)
- [Consolidation Enhancements (ConsolidationConfig)](#consolidation-enhancements)
- [Goal System Enhancements (GoalPersistenceConfig)](#goal-system-enhancements)
- [Metacognition Enhancements (MetacognitionConfig)](#metacognition-enhancements)
- [Valence / Affect Enhancements (ValenceConfig)](#valence-and-affect-enhancements)
- [Topology-Informed Learning](#topology-informed-learning)

---

## Surprise Gating

Controls how the system decides whether an input is "surprising enough" to warrant a weight update. Located in `StreamingConfig` and implemented in `src/sfp/core/streaming.py`.

### Soft Surprise Gate

Replaces the binary surprise gate (`updated = surprise > threshold`) with a continuous sigmoid that produces a gradient scale factor in [0, 1]. Eliminates the cliff-edge instability where surprise at 99% of threshold produces zero update while 101% produces a full update.

**How it works:**

1. A composite **importance score** is computed from consistency, confidence, and surprise magnitude. This score shifts the sigmoid center — high importance lowers the center (easier to update), low importance raises it.
2. A **sigmoid gate** `gate_scale = sigmoid(steepness * (surprise - center))` produces a continuous value.
3. If `gate_scale >= floor`, gradients are scaled by `gate_scale` before the optimizer step. If below the floor, the update is skipped entirely (noise rejection).

The total gradient scaling chain is: `gate_scale * confidence_scale * salience_scale * external_lr_scale`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `soft_gate_enabled` | `True` | Enable soft sigmoid gate. `False` reverts to binary. |
| `soft_gate_steepness` | `10.0` | Sigmoid slope. Higher = sharper transition around threshold. |
| `soft_gate_floor` | `0.05` | Minimum gate scale to proceed with update. Below this, skip. |
| `importance_consistency_weight` | `0.4` | Weight of Tier 2 consistency in importance score. |
| `importance_confidence_weight` | `0.3` | Weight of metacognition confidence in importance score. |
| `importance_surprise_weight` | `0.3` | Weight of surprise magnitude ratio in importance score. |

### Adaptive Surprise Threshold

Instead of a fixed threshold, uses the 90th percentile of recent gradient norms as a moving threshold. This lets the system automatically calibrate to the current data distribution.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `adaptive_surprise` | `True` | Enable percentile-based adaptive threshold. |
| `surprise_percentile` | `0.9` | Percentile of recent history to use as threshold. |
| `surprise_threshold` | `0.1` | Fixed threshold (used when adaptive is off, or as fallback for first 10 steps). |

### LR Warmup

Linear learning rate warmup ramps from `start_factor * lr` to `lr` over the warmup period. Prevents large early updates when the manifold is randomly initialized.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `warmup_steps` | `100` | Number of steps over which to linearly ramp LR. 0 = disabled. |
| `warmup_start_factor` | `0.1` | Starting LR as fraction of configured LR. |

---

## Co-Adaptation Losses

Auxiliary loss terms added to the primary reconstruction loss during Tier 0 field updates. These create bidirectional coupling between the field and other cognitive modules.

### World Model Prediction Error

Adds MSE between field output and the world model's predicted observation as an auxiliary loss. This pulls the field representation toward what the world model expects, creating co-adaptation between the two systems.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `auxiliary_loss_weight` | `0.1` | Weight of `MSE(field_output, wm_prediction)` in total loss. |

**Source:** The world model's deterministic + stochastic state is decoded to produce `wm_predicted_obs`, passed to `StreamingProcessor.process()` as `wm_prediction`.

### Goal Satisfaction Regularization

Adds negative cosine similarity between field output and active goal satisfaction embeddings. This biases the manifold toward states that satisfy current goals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal_loss_weight` | `0.05` | Weight of `-cosine_sim(field_output, goal_embeddings)` in total loss. |

---

## Memory Tier Feedback

Higher memory tiers (Tier 2 essential, Tier 3 core) feed knowledge back into Tier 0 field learning as auxiliary loss targets.

### Tier 2 Basin Guidance

Steers field output toward the nearest established Tier 2 basin key via MSE loss. This encourages the field to organize around known concepts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tier2_guidance_weight` | `0.05` | Weight of `MSE(field_output, nearest_basin_key)` in total loss. |

### Tier 3 Axiom Anchoring

Anchors field output toward Tier 3 core knowledge (axioms) via MSE loss. This provides long-term stability — the field drifts slowly near established axioms.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `axiom_anchor_weight` | `0.02` | Weight of `MSE(field_output, axiom_knowledge)` in total loss. |

### Consolidation Notification

After consolidation creates new Tier 2 basins, their keys are registered with Tier 0 for guidance during subsequent updates.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `consolidation_notify_tier0` | `True` | Notify Tier 0 of newly consolidated basin keys. |

---

## Gradient Modulation

Post-gate gradient scaling factors that modulate how strongly each update affects the manifold. These stack multiplicatively with the soft gate scale.

### Confidence Modulation

Metacognition confidence gates update magnitude. Low confidence = cautious (small updates), high confidence = full updates.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_modulation_enabled` | `True` | Enable confidence-based gradient scaling. |
| `confidence_low_threshold` | `0.5` | Below this: scale gradients to 0.1 (very cautious). |
| `confidence_high_threshold` | `0.8` | Above this: full gradient scale (1.0). Linear ramp between. |

### Salience Gradient Scaling

The salience gate's combined score scales gradient magnitude. High-salience inputs get larger updates; low-salience inputs get smaller ones.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `salience_gradient_scaling` | `True` | Enable salience-based gradient scaling. |

The scale formula is `clamp(salience_score * 2.0, 0.1, 2.0)`.

### Valence LR Modulation

The affect system's risk tolerance modulates learning rate. High risk tolerance (approach mode) allows faster learning; low tolerance (avoidance mode) slows it.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `valence_lr_modulation` | `True` | Enable valence risk tolerance LR scaling. |
| `valence_lr_scale_range` | `(0.5, 1.5)` | (min, max) LR multiplier mapped from risk tolerance [0.2, 0.8]. |

### Vigilance Surprise Modulation

The affect system's vigilance level boosts effective surprise. High vigilance (threat state) lowers the effective surprise threshold, making the system more reactive.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `vigilance_surprise_modulation` | `True` | Enable vigilance-driven surprise threshold reduction. |

When vigilance > 0.5, `surprise_boost += (vigilance - 0.5) * 0.6`, which lowers the latent distance used for dual-path surprise verification.

---

## LoRA Multi-Signal Merge

Online-LoRA adapters accumulate low-rank updates that are periodically merged into the base field weights. Beyond the original gradient-norm-based trigger, merge can now be triggered by sustained cognitive signals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `uncertainty_merge_threshold` | `0.7` | Merge if mean uncertainty > threshold over window. |
| `uncertainty_merge_window` | `50` | Number of recent steps to average uncertainty over. |
| `mood_merge_threshold` | `-0.3` | Merge if mean mood < threshold over window (sustained negative affect). |
| `mood_merge_window` | `50` | Number of recent steps to average mood over. |
| `goal_stall_merge_steps` | `100` | Merge if any active goal has stalled for this many steps. |

The merge context is assembled by the orchestrator (`processor.py`) from metacognition uncertainty history, valence mood history, and goal progress history, then passed to `OnlineLoRAManager.check_and_merge()`.

---

## Consolidation Enhancements

Extensions to the consolidation engine that improve how knowledge flows between memory tiers. Located in `ConsolidationConfig` and implemented in `src/sfp/memory/consolidation.py`.

### Replay Through Field

During consolidation, sample episodes from Tier 1 and replay them through the field with reduced LR. This reinforces important patterns without overwriting recent online learning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `replay_through_field_enabled` | `True` | Enable episode replay through field during consolidation. |
| `replay_through_field_batch_size` | `8` | Number of episodes to replay per consolidation cycle. |
| `replay_lr_scale` | `0.5` | LR multiplier for replay updates (lower = more conservative). |

### Valence-Weighted Sampling

Bias consolidation sampling toward high-|valence| basins. Emotionally significant memories are consolidated more frequently.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `valence_weighted_sampling` | `True` | Bias sampling toward high absolute valence basins. |

Requires the valence system to be enabled and wired via `consolidation.set_valence_system()`.

### Skim Buffer Replay

During consolidation, replay observations that the salience gate previously classified as SKIM (stored in the skim buffer at reduced weight). This ensures borderline-relevant inputs still contribute to learning.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `skim_replay_enabled` | `True` | Enable skim buffer replay during consolidation. |
| `skim_replay_lr_scale` | `0.25` | LR multiplier for skim replay (lower than normal replay). |

### Goal Stall Forced Consolidation

When a goal stalls for an extended period, force a standard consolidation cycle. This can reorganize the manifold to unblock progress.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal_stall_consolidation_steps` | `200` | Force consolidation after goal stalls for this many steps. |

### Topology-Informed Consolidation Urgency

When the manifold's Betti B0 number (number of disconnected components) exceeds a threshold, reduce the consolidation interval. A highly fragmented manifold needs more frequent consolidation to form coherent concepts.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `betti_b0_consolidation_threshold` | `20` | B0 above this triggers interval reduction. |
| `betti_consolidation_interval_reduction` | `0.5` | Multiplier applied to standard consolidation interval when B0 is high. |

Requires `giotto-tda` to be installed for persistent homology computation.

---

## Goal System Enhancements

Extensions to the goal persistence register. Located in `GoalPersistenceConfig` and implemented in `src/sfp/goals/persistence.py`.

### Satisfaction Hindsight Training

When a goal is progressing well (progress > threshold), train the satisfaction encoder using the current observation as target. This improves the system's ability to predict what "goal satisfied" looks like, creating a self-improving satisfaction signal.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `satisfaction_hindsight_enabled` | `True` | Enable hindsight training of satisfaction encoder. |
| `satisfaction_hindsight_threshold` | `0.8` | Minimum progress to trigger hindsight training. |

### Goal Urgency LR Scaling

When a goal approaches its deadline (remaining time < 50% of total), scale up the learning rate proportionally. Creates time pressure that accelerates learning for urgent goals.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goal_urgency_lr_enabled` | `True` | Enable deadline-based LR scaling. |
| `goal_urgency_max_multiplier` | `2.0` | Maximum LR multiplier at deadline. Linear ramp from 1.0 at 50% remaining. |

---

## Metacognition Enhancements

Extensions to the metacognition and uncertainty system. Located in `MetacognitionConfig` and implemented in `src/sfp/metacognition/uncertainty.py`.

### Exploratory Goal Generation

When the metacognition system detects high uncertainty and suggests information-seeking actions, automatically create exploratory goals. This drives the system to actively seek out information in areas of high uncertainty.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metacognition_goal_generation` | `True` | Auto-create exploratory goals from info-seeking suggestions. |

Goals are created with low importance (0.3), low urgency (0.2), and 10-minute TTL, prefixed with `explore:` and the uncertainty source.

---

## Valence and Affect Enhancements

Extensions to the valence and affective state system. Located in `ValenceConfig` and implemented in `src/sfp/affect/valence.py`.

### Risk Tolerance LR Modulation

The affect system maintains a risk tolerance score derived from mood and vigilance. This modulates the learning rate — the system learns faster when in an exploratory (approach) state and slower when cautious (avoidance).

See [Valence LR Modulation](#valence-lr-modulation) in the Gradient Modulation section.

### Vigilance Surprise Modulation

The affect system's vigilance level (reflecting perceived threat) boosts effective surprise, making the system more reactive to environmental changes during heightened alertness.

See [Vigilance Surprise Modulation](#vigilance-surprise-modulation) in the Gradient Modulation section.

### Chain Valence Bias

After reasoning chain traversal, the valence of visited basins is computed. Negative-valence chains cause those basins to receive a penalty bias on the next routing step, making the system avoid recently negative reasoning paths.

This feature is always active when the valence system is enabled; no separate config toggle.

### Basin Valence Annotation

After reasoning, visited basins are annotated with their current valence signal. This builds up a valence map across the concept graph that influences future routing decisions.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `basin_valence_ema_decay` | `0.95` | EMA decay for basin valence tracking. |
| `edge_valence_ema_decay` | `0.95` | EMA decay for transition edge valence tracking. |

---

## Topology-Informed Learning

Uses persistent homology (Betti numbers) to monitor the manifold's topological structure and inform learning decisions. Requires the optional `giotto-tda` dependency.

### Topology Snapshot + Change Detection

During deep consolidation, the topology tracker takes a snapshot of the field's weight manifold and computes Betti numbers. Significant structural changes (e.g., components merging or splitting) are detected by comparing consecutive snapshots.

### Structural Change LoRA Merge

When significant topological events are detected (significance > 0.3), all LoRA adapters are immediately merged into the base weights. This ensures the base model captures the structural reorganization rather than keeping it isolated in adapters.

### Betti B0 Consolidation Urgency

See [Topology-Informed Consolidation Urgency](#topology-informed-consolidation-urgency) in the Consolidation section.

---

## World Model

The RSSM-based predictive world model has one configurable parameter relevant to the structural improvements:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grad_clip_norm` | `10.0` | Maximum gradient norm for world model parameters (was hardcoded at 100.0). |

---

## Feature Interaction Summary

The features form a multi-signal feedback network. Here is the flow during a single `process()` call:

```
Input
  |
  v
Salience Gate -----> SKIP (no compute) / SKIM (reduced update) / FULL
  |
  v
Perceiver IO + Backbone --> embedding
  |
  v
World Model train_step --> prediction error (-> auxiliary field loss)
Goal progress update --> satisfaction (-> field regularization loss)
  |
  v
Tier 3 retrieve --> axiom anchor loss
Tier 2 retrieve / reasoning chain --> basin guidance loss
  |
  v
Metacognition uncertainty --> confidence (-> gradient scaling)
                          --> info-seeking (-> exploratory goal creation)
  |
  v
Valence --> risk tolerance (-> LR scaling)
        --> vigilance (-> surprise threshold modulation)
        --> chain valence bias (-> next-step routing penalty)
  |
  v
Tier 0 field update:
  Loss = primary + auxiliary_wm + goal_satisfaction + tier2_guidance + axiom_anchor + EWC
  Gradients *= soft_gate_scale * confidence_scale * salience_scale * external_lr_scale
  Optimizer step (with LR warmup)
  |
  v
LoRA merge check (gradient norm + uncertainty + mood + goal stall triggers)
  |
  v
Episode storage (Tier 1) with valence annotation
Consolidation check (with topology urgency + goal stall forcing)
  Replay: field episodes + skim buffer + generative synthetics
  Valence-weighted sampling
  Topology snapshot + structural change detection
  Basin key notification to Tier 0
```

All features default to enabled. Each can be disabled independently via its config parameter without affecting the others.

---

## Stability Features

### EWC Loss Normalization

The EWC penalty is normalized by total parameter count, preventing the penalty from dominating the primary loss regardless of model size. Default `lambda_` reduced from 1000 to 100.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lambda_` | `100.0` | EWC penalty weight (applied after per-parameter normalization). |

### Post-Warmup LR Decay

After warmup completes, learning rate decays via cosine annealing. Prevents unnecessary jitter in the mature manifold.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr_decay_enabled` | `True` | Enable cosine annealing after warmup. |
| `lr_decay_steps` | `50000` | Number of steps for cosine anneal to reach minimum. |
| `lr_decay_min_factor` | `0.01` | Minimum LR as fraction of peak LR. |

### External LR Scale Bounds

Clamps the combined external LR scale (from valence + goal urgency) to prevent extreme values from upstream bugs.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `external_lr_scale_range` | `(0.1, 5.0)` | (min, max) for external LR scale clamping. |

### Rate Limit Cooldown

After consolidation completes, rate limiting is suppressed for N steps to prevent replay bursts from triggering defensive throttling.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `rate_limit_cooldown_steps` | `20` | Steps to suppress rate limiting post-consolidation. |

### Anomaly Rejection Scaling

Anomalous inputs (detected by the embedding anomaly detector) receive a scaled-down learning rate instead of full processing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `anomaly_lr_scale` | `0.1` | LR multiplier for anomalous inputs (lower = more cautious). |

### Loss Component Monitoring

Each loss component (primary, EWC, WM auxiliary, goal, Tier 2 guidance, axiom anchor) is tracked individually in `SurpriseMetric.loss_components`. A warning is logged when any component exceeds 10x the primary loss.

### Gradient Accumulation

Accumulates gradients over N steps before calling the optimizer, reducing noise in single-sample streaming updates.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gradient_accumulation_steps` | `1` | Steps to accumulate before optimizer.step(). 1 = no accumulation. |

---

## Feedback Loop Closures

### Metacognition Health → Consolidation

During standard/deep consolidation, `monitor_memory_health()` is called. Dormant basins (low activation count) have their confidence reduced by 10%, making them candidates for overwrite during the next consolidation cycle.

### Salience Hindsight Ground Truth

The salience gate's hindsight buffer is populated with labeled data during processing:
- **FULL path + weight update**: labeled as `useful=True`
- **FULL path + no update**: labeled as `useful=False`
- **SKIM path**: labeled as `useful=False`

This data is consumed by `run_hindsight_training()` during consolidation to improve salience predictions.

### Surprise-Weighted Replay

Episode replay sampling now weights by `surprise_at_storage` within each basin. High-surprise episodes (which carry the strongest learning signal) are replayed more frequently.

### World Model Reward → Valence

The world model's `reward_head` output is computed after each train step and passed to the valence system as the `reward` parameter. This connects learned environmental reward predictions to the affective system.

### Edge Valence Annotation

During reasoning chain traversal, transition edges between consecutive visited basins are annotated with valence. This populates the 40,960-slot edge valence buffer that was previously allocated but unused.
