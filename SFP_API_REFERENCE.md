# SFP API Reference

Complete API reference for the Semantic Field Processing system. All classes, methods, configurations, and types needed to connect external processes.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [SFPInterface](#2-sfpinterface) — Thread-safe external facade
3. [HierarchicalMemoryProcessor](#3-hierarchicalmemoryprocessor) — Main orchestrator
4. [Core: Tier 0](#4-core-tier-0) — Working memory
5. [Memory: Tiers 1–3](#5-memory-tiers-13) — Episodic, Essential, Core
6. [Consolidation](#6-consolidation) — Tier promotion engine
7. [Reasoning](#7-reasoning) — Transitions, chains, routing, learning
8. [Cognitive Modules](#8-cognitive-modules) — World model, goals, metacognition, valence, salience, replay
9. [Defense](#9-defense) — Input validation, gradient bounds, topology monitoring
10. [Architecture](#10-architecture) — Perceiver IO, Backbone Transformer
11. [Configuration](#11-configuration) — All config dataclasses
12. [Types & Enums](#12-types--enums) — Data structures, protocols, enumerations
13. [Integrity](#13-integrity) — Cryptographic hashing utilities

---

## 1. Quick Start

### Minimal pipeline (Tier 0 only)

```python
import torch
from sfp.config import FieldConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.core.streaming import StreamingProcessor

field = SemanticFieldProcessor(FieldConfig(dim=128, n_layers=4))
processor = StreamingProcessor(field, StreamingConfig(surprise_threshold=0.1))

x = torch.randn(128)
metric = processor.process(x)
# metric.grad_norm, metric.loss, metric.updated
```

### Full pipeline (all tiers + cognitive modules)

```python
from sfp.config import *
from sfp.memory.processor import HierarchicalMemoryProcessor

processor = HierarchicalMemoryProcessor(
    field_config=FieldConfig(dim=512, n_layers=6),
    tier1_config=Tier1Config(hot_capacity=2000),
    tier2_config=Tier2Config(n_slots=4096),
    tier3_config=Tier3Config(n_slots=512),
    world_model_config=WorldModelConfig(d_deterministic=512, d_observation=512),
    valence_config=ValenceConfig(),
    device="cuda",
)

for x in data_stream:
    result = processor.process(x, modality="tensor")
    if result.updated:
        print(f"Learned: loss={result.loss:.4f} surprise={result.grad_norm:.4f}")
```

### Query without updating

```python
result = processor.query(query_embedding, return_trace=True)
# result.knowledge — retrieved knowledge vector
# result.n_hops — reasoning chain length
# result.visited_basins — basin IDs traversed
# result.trace — list of ChainTrace events
```

### External integration (recommended)

For external processes (bridges, adapters, GUIs), use [`SFPInterface`](#2-sfpinterface) — the thread-safe facade:

```python
import sfp

processor = sfp.create_field("small", hierarchical=True, valence=True, goals=True)
interface = sfp.SFPInterface(processor)

# Thread-safe from any thread
metric = interface.process(obs_tokens, modality="environment", metadata=metadata)
result = interface.query(obs_tokens)
interface.inject_valence(obs_tensor, reward=0.5)
interface.create_goal(instruction_embedding, importance=0.8)
nearby = interface.retrieve_by_location((10.0, 64.0, -20.0), radius=30.0)
```

---

## 2. SFPInterface

**Module:** `sfp.interface`

Thread-safe external facade for the SFP system. Wraps a `HierarchicalMemoryProcessor` with `threading.RLock` serialization, providing a unified API for external integrations (bridges, adapters, GUIs) to interact with the memory system concurrently without corruption.

**Design intent:** SFPInterface is the single point of access for all external interaction. Both GUI threads and bridge threads share one instance — the RLock ensures safe concurrent access without IPC.

### Constructor

```python
SFPInterface(processor: HierarchicalMemoryProcessor)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `processor` | `HierarchicalMemoryProcessor` | Fully initialized processor (via `sfp.create_field(..., hierarchical=True)`) |

Raises `TypeError` if `processor` is not a `HierarchicalMemoryProcessor`.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `processor` | `HierarchicalMemoryProcessor` | Direct access to underlying processor. **Bypasses thread-safety.** |
| `lock` | `threading.RLock` | The interface's RLock — for callers that need to hold the lock across multiple operations. |
| `d_model` | `int` | Embedding dimensionality. Use this to create correctly shaped input tensors. |
| `step_count` | `int` | Number of `process()` calls completed. |
| `is_valence_enabled` | `bool` | Whether the valence/affect module is active. |
| `is_goals_enabled` | `bool` | Whether the goal persistence module is active. |

### Core Processing

#### `process(x, modality="tensor", target=None, metadata=None) → SurpriseMetric`

Submit an observation through the full hierarchical pipeline. Updates weights, stores episodes, runs consolidation checks.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | — | Input tensor: `(d_model,)`, `(B, d_model)`, or `(1, N, d_model)` for multi-token observations |
| `modality` | `str` | `"tensor"` | Modality identifier (e.g. `"environment"`, `"text"`) |
| `target` | `torch.Tensor \| None` | `None` | Optional target for supervised updates |
| `metadata` | `dict \| None` | `None` | Bridge-provided metadata. Supported keys: `"entity_positions"` `(N, 3)` tensor, `"entity_embeddings"` `(N, d_model)` tensor, `"spatial_position"` `(x, y, z)` tuple, `"spatial_orientation"` `(yaw, pitch)` tuple |

**Returns:** `SurpriseMetric` with `.grad_norm`, `.loss`, `.updated`.

```python
# Single vector observation
metric = interface.process(obs_tensor, modality="text")

# Multi-token observation with spatial metadata (from a bridge)
metadata = {
    "spatial_position": (10.0, 64.0, -20.0),
    "spatial_orientation": (45.0, 0.0),
    "entity_positions": entity_pos_tensor,   # (N, 3)
    "entity_embeddings": entity_emb_tensor,  # (N, d_model)
}
metric = interface.process(obs_tokens, modality="environment", metadata=metadata)
```

#### `query(x, return_trace=False) → ReasoningResult`

Query the memory system without updating weights. Read-only — does not increment step count or trigger consolidation.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `x` | `torch.Tensor` | — | Query tensor (same shape rules as `process`) |
| `return_trace` | `bool` | `False` | Include per-hop reasoning trace in result |

**Returns:** `ReasoningResult` with `.knowledge` (retrieved vector), `.n_hops`, `.visited_basins`, `.trace`.

```python
result = interface.query(obs_tensor)
action_embedding = result.knowledge  # shape: (d_model,)
```

#### `consolidate(force_mode=None) → None`

Trigger memory consolidation (episode promotion across tiers).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `force_mode` | `ConsolidationMode \| None` | `None` | Force a specific mode (`MINI`, `STANDARD`, `DEEP`), or `None` to let the engine decide |

```python
interface.consolidate()  # auto-schedule
interface.consolidate(force_mode=ConsolidationMode.DEEP)  # force deep
```

### Valence Injection

#### `inject_valence(embedding, *, reward=0.0, user_feedback=0.0, goal_alignment=0.0, prediction_satisfaction=0.0) → ValenceSignal | None`

Inject an external valence signal into the affect system. Updates the processor's internal mood state so subsequent processing steps reflect the injected signal.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding` | `torch.Tensor` | — | `(d_model,)` embedding to associate valence with |
| `reward` | `float` | `0.0` | Raw RL reward (normalized internally) |
| `user_feedback` | `float` | `0.0` | Explicit human feedback `[-1, 1]` |
| `goal_alignment` | `float` | `0.0` | Goal alignment score `[-1, 1]` |
| `prediction_satisfaction` | `float` | `0.0` | Prediction confirmation `[-1, 1]` |

**Returns:** `ValenceSignal` if the valence module is enabled, `None` otherwise.

```python
signal = interface.inject_valence(
    obs_tensor,
    reward=0.5,
    user_feedback=0.8,
    goal_alignment=0.3,
)
if signal is not None:
    print(f"Mood: {signal.composite_mood:.2f}, Valence: {signal.scalar_valence:.2f}")
```

### Goal Management

#### `create_goal(instruction_embedding, *, importance=0.5, urgency=0.5, deadline=None, ttl=None) → Goal | None`

Create a goal from an instruction embedding.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruction_embedding` | `torch.Tensor` | — | `(d_model,)` embedding of the goal instruction |
| `importance` | `float` | `0.5` | Static importance `[0, 1]` |
| `urgency` | `float` | `0.5` | Initial urgency `[0, 1]` |
| `deadline` | `float \| None` | `None` | Absolute monotonic-time deadline |
| `ttl` | `float \| None` | `None` | Time-to-live in seconds (default from config) |

**Returns:** `Goal` if goals module is enabled, `None` otherwise.

```python
goal = interface.create_goal(instruction_embedding, importance=0.8, urgency=0.6)
if goal is not None:
    print(f"Goal {goal.id} created, status={goal.status.name}")
```

#### `remove_goal(goal_id) → bool`

Remove a goal by ID.

**Returns:** `True` if removed, `False` if goals are disabled or goal not found.

#### `list_goals() → list[dict]`

Return all goals as serializable dicts.

Each dict contains: `id`, `status`, `priority`, `progress`, `importance`, `urgency`.

```python
for g in interface.list_goals():
    print(f"Goal {g['id']}: {g['status']} priority={g['priority']:.2f}")
```

### Episode Storage

#### `store_episode(input_embedding, *, modality="external", surprise=1.0, valence=0.0) → bool`

Store an externally-created episode in Tier 1. Handles all internal complexity: provenance hashing, weight summaries, logit snapshots, integrity hashing, and basin assignment. The caller only provides the embedding and metadata.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_embedding` | `torch.Tensor` | — | `(d_model,)` embedding to store |
| `modality` | `str` | `"external"` | Modality tag (e.g. `"minecraft"`, `"demo"`) |
| `surprise` | `float` | `1.0` | Surprise score — must exceed Tier 1's threshold for admission |
| `valence` | `float` | `0.0` | Valence annotation for the episode |

**Returns:** `True` if episode was admitted, `False` if rejected (surprise threshold, deduplication, or integrity check).

```python
stored = interface.store_episode(
    demo_embedding,
    modality="demo",
    surprise=1.0,
    valence=0.3,
)
```

### Status and Health

#### `health_report() → dict`

Comprehensive health report across all tiers and modules. Returns a nested dict with per-tier and per-module metrics.

#### `memory_footprint() → dict[str, int]`

Estimate VRAM usage per component in bytes. Returns dict mapping component name to byte count, plus `"total"`.

#### `status() → dict`

Quick status summary for overlay rendering or monitoring. Lighter than `health_report()`.

**Returns:** Dict with keys:
- `step_count` — number of `process()` calls
- `tier1_episodes` — total episodes in Tier 1
- `tier2_basins` — active Tier 2 basins
- `tier3_axioms` — active Tier 3 axioms
- `active_goals` — (only if goals enabled) number of active goals
- `mood` — (only if valence enabled and signal exists) composite mood
- `valence` — (only if valence enabled and signal exists) scalar valence

```python
s = interface.status()
print(f"Step {s['step_count']}: {s['tier1_episodes']} episodes, {s['tier2_basins']} basins")
```

**`health_report()` additional keys** (from the 3D perception pipeline):

| Key Path | Type | Description |
|----------|------|-------------|
| `tier1.spatial_count` | `int` | Episodes with `spatial_position != None` |
| `world_model.spatial_loss` | `float` | Spatial prediction EMA loss (0 if no spatial data) |
| `scene_graph.n_nodes` | `int` | Current entity count tracked by scene graph |

### Spatial Memory

#### `retrieve_by_location(position, *, radius=50.0, max_results=10, embedding=None, spatial_weight=0.7) → list[dict]`

Retrieve episodes stored near a spatial position. "I was here before" — finds episodes with nearby `spatial_position`, optionally blended with semantic similarity.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `position` | `tuple[float, float, float]` | — | Query position `(x, y, z)` |
| `radius` | `float` | `50.0` | Maximum Euclidean distance |
| `max_results` | `int` | `10` | Maximum episodes to return |
| `embedding` | `torch.Tensor \| None` | `None` | Optional `(d_model,)` query for semantic blending |
| `spatial_weight` | `float` | `0.7` | Weight for spatial proximity vs semantic similarity |

**Returns:** List of dicts sorted by descending score:
```python
[
    {"id": 42, "score": 0.95, "distance": 2.5, "position": (10.0, 64.0, -20.0), ...},
    {"id": 37, "score": 0.72, "distance": 14.0, "position": (15.0, 64.0, -25.0), ...},
]
```

```python
nearby = interface.retrieve_by_location((10.0, 64.0, -20.0), radius=30.0)
for ep in nearby:
    print(f"Episode {ep['id']} at distance {ep['distance']:.1f}, score {ep['score']:.2f}")
```

### Session Management

#### `reset_session() → None`

Reset Tier 0 working memory (session volatility). Does not affect long-term memories in Tiers 1–3.

### Thread-Safety

All public methods acquire the interface's `threading.RLock` before accessing the processor. This means:

- **Safe for concurrent access** from GUI threads, bridge threads, and background workers.
- **Reentrant** — internal calls that re-acquire the lock will not deadlock.
- **No IPC needed** — GUI and bridge run in the same Python process and share one `SFPInterface` instance.

For operations that need to hold the lock across multiple calls, use the `lock` property:

```python
with interface.lock:
    status = interface.status()
    if status["tier1_episodes"] > 1000:
        interface.consolidate(force_mode=ConsolidationMode.STANDARD)
```

### Integration Pattern

The intended usage for a GUI + bridge architecture:

```python
import sfp

# 1. GUI creates the processor
processor = sfp.create_field("small", hierarchical=True, valence=True, goals=True)

# 2. Wrap in thread-safe interface
interface = sfp.SFPInterface(processor)

# 3. GUI uses the interface for its own operations
status = interface.status()
interface.consolidate()

# 4. Bridge receives the same interface instance
def start_bridge(interface: sfp.SFPInterface):
    """Runs in a separate thread."""
    d = interface.d_model
    while running:
        obs_tokens, metadata = tokenize_observation(d)  # (1, N, d_model) + metadata
        metric = interface.process(obs_tokens, modality="environment", metadata=metadata)
        result = interface.query(obs_tokens)
        interface.inject_valence(obs_tokens.mean(dim=1).squeeze(0), reward=compute_reward())

# 5. Both threads safely share the interface
bridge_thread = threading.Thread(target=start_bridge, args=(interface,))
bridge_thread.start()
```

### Adapter Mapping

For projects like SFPMC, the SFPInterface methods map directly to adapter needs:

| Adapter Operation | SFPInterface Method |
|-------------------|---------------------|
| Submit multi-token observation at 20Hz | `interface.process(obs_tokens, modality="environment", metadata=meta)` |
| Query for action (multi-token) | `interface.query(obs_tokens)` |
| Inject reward/feedback | `interface.inject_valence(emb, reward=r, user_feedback=f)` |
| Set a goal | `interface.create_goal(emb, importance=p)` |
| Store demo episode | `interface.store_episode(emb, modality="demo", surprise=1.0)` |
| Retrieve nearby episodes | `interface.retrieve_by_location((x, y, z), radius=50.0)` |
| Trigger consolidation | `interface.consolidate()` |
| Check system status | `interface.status()` |
| Get embedding dimension | `interface.d_model` |

---

## 3. HierarchicalMemoryProcessor

**Module:** `sfp.memory.processor`

Primary orchestrator composing all tiers and cognitive modules into a single processing pipeline.

### Constructor

```python
HierarchicalMemoryProcessor(
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
)
```

All config parameters are optional. Pass `None` to disable a module. Pass a config to enable and configure it.

### Methods

#### `process`
```python
def process(
    x: torch.Tensor,
    modality: str = "tensor",
    target: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> SurpriseMetric
```
Process input through the full pipeline:
1. Salience gate (Skip/Skim/Full)
2. Input sanitization + anomaly detection
3. Perceiver IO → Backbone (if configured)
4. World model train step (including spatial prediction if `metadata` has `spatial_position`)
4a. Scene graph update (spatial relations from entity positions in `metadata`)
5. Goal progress update
6. Tier 3 retrieval (core axioms)
7. Tier 2 retrieval + reasoning chain (via router, with spatial bias from scene graph)
8. Valence computation + basin annotation
9. Metacognition uncertainty estimation
10. Tier 0 surprise-gated weight update
11. Episode storage in Tier 1 (with `spatial_position`/`spatial_orientation` from `metadata`)
12. Automatic consolidation check

**Parameters:**
- `x` — Input tensor. Shape `(dim,)`, `(batch, dim)`, or `(1, N, dim)` for multi-token.
- `modality` — Modality label for provenance tracking and salience gating.
- `target` — Optional supervision target. If `None`, uses autoassociative loss.
- `metadata` — Optional bridge metadata dict (entity positions, spatial position, etc.).

**Returns:** `SurpriseMetric` with `grad_norm`, `loss`, `updated`.

---

#### `query`
```python
def query(
    x: torch.Tensor,
    return_trace: bool = False,
) -> ReasoningResult
```
Query memory without updating weights. Retrieves from Tier 2/3 and runs reasoning chain.

**Returns:** `ReasoningResult` with `knowledge`, `n_hops`, `visited_basins`, `trace`.

---

#### `consolidate`
```python
def consolidate(force_mode: ConsolidationMode | None = None) -> None
```
Manually trigger consolidation. If `force_mode` is `None`, determines which mode is due based on step count and intervals.

---

#### `health_report`
```python
def health_report() -> dict
```
Returns comprehensive health report including:
- `tier0`: param_count, surprise_history_len
- `tier1`: hot_count, cold_count, total_count
- `tier2`: n_active, mean_confidence
- `tier3`: n_active, integrity_failures
- `consolidation`: total_mini, total_standard, total_deep
- `transitions`: n_active_edges
- `reasoning`: (chain stats)
- `defense`: anomaly stats

---

#### `memory_footprint`
```python
def memory_footprint() -> dict[str, int]
```
Returns estimated memory usage in bytes per component.

---

#### `reset_session`
```python
def reset_session() -> None
```
Reset Tier 0 working memory (session volatility). Tiers 1–3 are preserved.

---

#### `set_anchor_verifier`
```python
def set_anchor_verifier(verifier: AnchorVerifier) -> None
```
Set an anchor verifier for ongoing integrity checks against known concept anchors.

---

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `tier0` | `StreamingProcessor` | Tier 0 working memory |
| `tier1` | `EpisodicMemory` | Tier 1 episodic buffer |
| `tier2` | `EssentialMemory` | Tier 2 concept basins |
| `tier3` | `CoreMemory` | Tier 3 core axioms |
| `transitions` | `TransitionStructure` | Concept graph |
| `perceiver` | `PerceiverIO` | Multi-modal bottleneck |
| `backbone` | `BackboneTransformer` | Contextual processor |
| `event_emitter` | `PromotionEventEmitter` | Tier 2→3 promotion events |
| `world_model` | `PredictiveWorldModel \| None` | RSSM world model |
| `goals` | `GoalRegister \| None` | Goal management |
| `metacognition` | `MetacognitionEngine \| None` | Uncertainty estimation |
| `valence` | `ValenceSystem \| None` | Hedonic system |
| `salience_gate` | `SalienceGate \| None` | Attention gating |
| `replay` | `GenerativeReplay \| None` | Generative replay |

---

## 4. Core: Tier 0

### SemanticFieldProcessor

**Module:** `sfp.core.field`

MLP whose weight geometry defines a concept manifold. Fixed-point attractors in this manifold correspond to learned concepts.

```python
SemanticFieldProcessor(config: FieldConfig)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `forward` | `(x: Tensor) -> Tensor` | `(*, dim)` | Forward pass through manifold |
| `jacobian` | `(x: Tensor) -> Tensor` | `(dim, dim)` or `(B, dim, dim)` | Jacobian matrix dF/dx |
| `linear_layers` | `() -> list[nn.Linear]` | List of Linear layers | For LoRA wrapping, EWC |
| `get_weight_summary` | `() -> Tensor` | `(dim,)` vector | Weight state fingerprint |
| `memory_bytes` | `(dtype=float32) -> int` | Byte count | Memory estimation |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `param_count` | `int` | Total parameter count |
| `config` | `FieldConfig` | Configuration |

---

### StreamingProcessor

**Module:** `sfp.core.streaming`

Surprise-gated online learning with LoRA, EWC, and defense hardening.

```python
StreamingProcessor(
    field: SemanticFieldProcessor,
    streaming_config: StreamingConfig | None = None,
    lora_config: LoRAConfig | None = None,
    ewc_config: EWCConfig | None = None,
    tier0_config: Tier0Config | None = None,
    consistency_checker: ConsistencyChecker | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `process` | `(x, target=None, latent_distance=None) -> SurpriseMetric` | SurpriseMetric | Process single input with gated update |
| `process_stream` | `(stream, callback=None) -> list[SurpriseMetric]` | Metric list | Process iterable stream |
| `query` | `(x, config=None) -> AttractorResult` | AttractorResult | Query nearest attractor |
| `reset_working_memory` | `() -> None` | — | Reset weights and state |
| `set_consistency_checker` | `(checker) -> None` | — | Set Tier 2 consistency checker |
| `reset_optimizer` | `() -> None` | — | Rebuild optimizer |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `surprise_history` | `list[SurpriseMetric]` | Full update history |
| `field` | `SemanticFieldProcessor` | The MLP manifold |
| `lora_manager` | `OnlineLoRAManager \| None` | LoRA adapter manager |
| `ewc_strategy` | `EWCStrategy \| None` | EWC forgetting strategy |
| `config` | `StreamingConfig` | Configuration |

---

### AttractorQuery

**Module:** `sfp.core.attractors`

Content-addressable memory via iterative fixed-point convergence on the manifold.

```python
AttractorQuery(field: SemanticFieldProcessor, config: AttractorConfig | None = None)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `query` | `(x: Tensor) -> AttractorResult` | AttractorResult | Converge single input to nearest attractor |
| `query_batch` | `(xs: Tensor) -> list[AttractorResult]` | List of results | Converge batch |
| `discover_attractors` | `(n_probes=1000, merge_radius=0.1) -> Tensor` | `(N, dim)` | Discover unique attractors via random probing |
| `map_basins` | `(grid_points: Tensor) -> tuple[Tensor, Tensor]` | `(basin_ids, converged)` | Map points to basin IDs |

---

### OnlineLoRAManager

**Module:** `sfp.core.lora`

Low-rank adaptation for continual learning with merge-on-shift.

```python
OnlineLoRAManager(field: SemanticFieldProcessor, config: LoRAConfig)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `wrap_field` | `() -> None` | — | Replace Linear layers with LoRA |
| `trainable_parameters` | `() -> Iterator[Parameter]` | Parameters | LoRA params only |
| `total_lora_params` | `() -> int` | Count | Total LoRA parameter count |
| `check_and_merge` | `(surprise_history) -> bool` | Merged? | Merge if distribution shift detected |
| `merge_all` | `() -> None` | — | Merge all LoRA → base |

---

### EWCStrategy

**Module:** `sfp.core.forgetting`

Elastic Weight Consolidation for catastrophic forgetting prevention. Implements `ForgetStrategy` protocol.

```python
EWCStrategy(field: SemanticFieldProcessor, config: EWCConfig)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `penalty` | `(model: nn.Module) -> Tensor` | Scalar loss | EWC penalty term |
| `update_importance` | `(model: nn.Module) -> None` | — | Update Fisher estimates |
| `update_anchors` | `(model: nn.Module) -> None` | — | Snapshot current weights as anchors |

---

## 5. Memory: Tiers 1–3

### EpisodicMemory (Tier 1)

**Module:** `sfp.memory.episodic`

Structured hot/cold episodic buffer with surprise-gated admission, deduplication, and integrity verification.

```python
EpisodicMemory(config: Tier1Config | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `maybe_store` | `(episode: Episode) -> bool` | Admitted? | Attempt to store (3 gates: surprise, dedup, integrity) |
| `sample_for_replay` | `(batch_size=32) -> list[Episode]` | Episode list | Stratified sampling by basin |
| `validate_integrity` | `() -> list[Episode]` | Failed episodes | Verify all episode integrity hashes |
| `promote_to_hot` | `(episode_ids) -> int` | Count | Move cold → hot |
| `allocate_id` | `() -> int` | Episode ID | Next sequential ID |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `hot_count` | `int` | Episodes in hot buffer |
| `cold_count` | `int` | Episodes in cold buffer |
| `total_count` | `int` | Total episodes |
| `last_episode_id` | `int \| None` | Most recent ID |
| `basin_distribution` | `dict[int, int]` | Episodes per basin |

---

### EssentialMemory (Tier 2)

**Module:** `sfp.memory.essential`

Key-value associative memory with Hopfield-style retrieval. Basins correspond to learned concepts. Also implements `ConsistencyChecker` protocol.

```python
EssentialMemory(config: Tier2Config | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `retrieve` | `(query, top_k=0) -> (output, basin_ids, attn_weights)` | Tuple of 3 tensors | Hopfield retrieval |
| `allocate_slot` | `(key, value) -> int` | Slot index | Allocate new basin |
| `update_slot` | `(slot, key_delta, value_delta, confidence_update, ...) -> None` | — | EMA update existing basin |
| `check_consistency` | `(input_embedding, proposed_update) -> Tensor` | Scores `[0,1]` | ConsistencyChecker protocol |

#### Key Parameters

| Parameter | Type | Shape | Description |
|-----------|------|-------|-------------|
| `keys` | `nn.Parameter` | `(n_slots, d_model)` | Basin key embeddings |
| `values` | `nn.Parameter` | `(n_slots, d_value)` | Basin value embeddings |
| `confidence` | `Tensor` | `(n_slots,)` | Per-basin confidence |
| `active_mask` | `Tensor` | `(n_slots,)` | Boolean active mask |
| `importance` | `Tensor` | `(n_slots,)` | Importance scores |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_active` | `int` | Number of active basins |
| `active_indices` | `Tensor` | Indices of active slots |
| `active_keys_tensor` | `Tensor` | `(n_active, d_model)` active keys |
| `config` | `Tier2Config` | Configuration |

---

### CoreMemory (Tier 3)

**Module:** `sfp.memory.core`

Near-frozen axiomatic memory with SHA-256 cryptographic integrity. Promotion requires authorization through the event system.

```python
CoreMemory(
    config: Tier3Config | None = None,
    d_model: int = 512,
    event_emitter: PromotionEventEmitter | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `retrieve` | `(query: Tensor) -> Tensor` | `(d_model,)` or `(B, d_model)` | Retrieve axiom knowledge |
| `promote_from_tier2` | `(tier2, slot_id) -> bool` | Success? | Promote Tier 2 basin to core |
| `verify_integrity` | `() -> list[int]` | Failed slot indices | Verify all SHA-256 hashes |
| `write_slot` | `(slot, key, value) -> None` | — | Direct write (for initialization) |

#### Properties

Same structure as EssentialMemory: `keys`, `values`, `confidence`, `active_mask`, `n_active`.

---

### PromotionEventEmitter

**Module:** `sfp.memory.events`

Event system for Tier 2→3 promotion authorization.

```python
PromotionEventEmitter(default_policy: str = "deny")
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `register` | `(handler, priority=0) -> None` | — | Register authorization handler |
| `unregister` | `(handler) -> None` | — | Remove handler |
| `emit` | `(request: PromotionRequest) -> bool` | Approved? | Request promotion authorization |

#### CriteriaAuthorizationHandler

Pre-built handler that checks confidence, episode count, modality count, and age criteria.

```python
CriteriaAuthorizationHandler(tier3_config: Tier3Config)
```

---

## 6. Consolidation

### ConsolidationEngine

**Module:** `sfp.memory.consolidation`

Manages knowledge transfer between tiers on configurable intervals.

```python
ConsolidationEngine(
    config: ConsolidationConfig | None = None,
    tier0: StreamingProcessor | None = None,
    tier1: EpisodicMemory | None = None,
    tier2: EssentialMemory | None = None,
    tier3: CoreMemory | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `should_consolidate` | `(step_count) -> ConsolidationMode \| None` | Mode or None | Check if consolidation is due |
| `consolidate` | `(mode, step_count) -> None` | — | Run consolidation (cascades) |
| `mini_consolidate` | `(step_count) -> None` | — | Tier 0 → Tier 1 |
| `standard_consolidate` | `(step_count) -> None` | — | Tier 1 → Tier 2 |
| `deep_consolidate` | `(step_count) -> None` | — | Tier 2 → Tier 3 |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `stats` | `dict` | `total_mini`, `total_standard`, `total_deep` counts |

#### Consolidation Modes

| Mode | Default Interval | Direction | What Happens |
|------|-----------------|-----------|--------------|
| `MINI` | 100 steps | Tier 0 → 1 | Create episode from current working memory state |
| `STANDARD` | 1,000 steps | Tier 1 → 2 | Sample episodes, match to basins or create new ones |
| `DEEP` | 10,000 steps | Tier 2 → 3 | Promote mature basins to core (requires authorization) |

---

## 7. Reasoning

### TransitionStructure

**Module:** `sfp.reasoning.transitions`

Sparse directed graph of typed relations between Tier 2 concept basins. Stored in COO format with capacity-limited eviction.

```python
TransitionStructure(config: TransitionConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `add_edge` | `(src, tgt, relation=ASSOCIATIVE, weight=0.1, confidence=0.1) -> int` | Edge index | Add or update edge |
| `get_outgoing` | `(basin_id) -> (target_ids, weights, edge_indices)` | 3 Tensors | Outgoing edges |
| `get_incoming` | `(basin_id) -> (source_ids, weights, edge_indices)` | 3 Tensors | Incoming edges |
| `compute_transition_scores` | `(basin_id, query_context) -> (scores, target_ids)` | 2 Tensors | Score outgoing transitions |
| `get_edge_info` | `(edge_idx) -> dict` | Dict | Edge metadata |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `n_active_edges` | `int` | Active edge count |
| `relation_prototypes` | `nn.Parameter` | `(n_types, d_relation)` relation embeddings |
| `config` | `TransitionConfig` | Configuration |

#### Relation Types

| Type | Value | Description |
|------|-------|-------------|
| `CAUSAL` | 0 | A causes B |
| `TEMPORAL` | 1 | A precedes B |
| `COMPOSITIONAL` | 2 | A is part of B |
| `ANALOGICAL` | 3 | A is analogous to B |
| `INHIBITORY` | 4 | A inhibits B |
| `ASSOCIATIVE` | 5 | A associates with B |

---

### AssociativeReasoningChain

**Module:** `sfp.reasoning.chain`

Multi-hop traversal through concept graph with cycle detection, convergence checking, and goal/valence biasing.

```python
AssociativeReasoningChain(
    tier2: EssentialMemory,
    transitions: TransitionStructure,
    config: ReasoningChainConfig | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `reason` | `(query, return_trace=False, target_bias=None) -> ReasoningResult` | ReasoningResult | Execute multi-hop chain |

**Parameters:**
- `query` — `(d_model,)` query vector.
- `return_trace` — If `True`, populate `result.trace` with `ChainTrace` events.
- `target_bias` — `dict[int, float]` mapping basin_id → score bonus. Used for goal/valence steering.

**Termination reasons:** `"empty_memory"`, `"max_hops"`, `"convergence"`, `"dead_end"`, `"cycle"`.

---

### ReasoningRouter

**Module:** `sfp.reasoning.router`

Decides single-hop vs multi-hop retrieval based on residual norm, attention entropy, and outgoing edge count.

```python
ReasoningRouter(
    tier2: EssentialMemory,
    transitions: TransitionStructure,
    chain: AssociativeReasoningChain,
    residual_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    min_outgoing_edges: int = 1,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `route` | `(query, return_trace=False, target_bias=None) -> ReasoningResult` | ReasoningResult | Route to single or multi-hop |

---

### TransitionLearner

**Module:** `sfp.reasoning.learning`

Discovers typed relations between basins from episodic patterns.

```python
TransitionLearner(
    tier2: EssentialMemory,
    transitions: TransitionStructure,
    temporal_window: int = 5,
    similarity_threshold: float = 0.3,
    causal_asymmetry_ratio: float = 2.0,
    compositional_distance: float = 0.7,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `learn_from_episodes` | `(episodes: list[Episode]) -> int` | Edge count | Mine temporal/causal relations |
| `learn_compositional_relations` | `() -> int` | Edge count | Discover compositional from geometry |
| `learn_inhibitory_relations` | `() -> int` | Edge count | Discover inhibitory from errors |

---

### ChainShortcutLearner

**Module:** `sfp.reasoning.learning`

Creates shortcut edges from frequently-traversed chains.

```python
ChainShortcutLearner(
    transitions: TransitionStructure,
    config: ReasoningChainConfig | None = None,
)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `observe_chain` | `(basin_sequence: list[int], quality: float) -> None` | — | Record observed chain |
| `create_shortcuts` | `() -> int` | Shortcut count | Create shortcuts from frequent chains |

---

## 8. Cognitive Modules

### PredictiveWorldModel

**Module:** `sfp.prediction.world_model`

RSSM-based predictive world model with 8-subspace directional surprise and pre-activation cache.

```python
PredictiveWorldModel(config: WorldModelConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `step` | `(observation: Tensor) -> WorldModelState` | WorldModelState | Advance RSSM one step |
| `train_step` | `(observation: Tensor, *, spatial_position=None) -> dict[str, float]` | Loss dict | Full training step with internal Adam. If `spatial_position` is a `(3,)` tuple/tensor on consecutive calls, trains spatial prediction head. |
| `predict_spatial_delta` | `() -> Tensor \| None` | `(3,)` or None | Predict next position delta (dx, dy, dz). Returns None if no prior spatial data. |
| `compute_enhanced_surprise` | `(state: WorldModelState) -> float` | Scalar | Enhanced surprise from prediction errors |
| `check_cache` | `(query: Tensor) -> (Tensor \| None, float)` | (output, score) | Pre-activation cache lookup |
| `reset_cache` | `() -> None` | — | Clear cache |
| `imagine` | `(steps: int) -> list[WorldModelState]` | State trajectory | Multi-step imagination |
| `project_multi_step` | `(observation, steps) -> Tensor` | `(steps, d_det)` | Project deterministic states |
| `directional_prediction_error` | `(predicted, actual) -> Tensor` | `(n_subspaces,)` | Per-subspace error |
| `reset_state` | `() -> None` | — | Reset hidden states to zeros |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `_h` | `Tensor` | `(d_deterministic,)` deterministic state |
| `_z` | `Tensor` | `(d_stoch_flat,)` stochastic state |
| `config` | `WorldModelConfig` | Configuration |

---

### SceneGraph

**Module:** `sfp.reasoning.scene_graph`

Generic scene graph for spatial reasoning. Maintains entity nodes with positions and embeddings, classifies spatial relations between pairs, and injects `SPATIAL_*` edges into the `TransitionStructure`.

```python
SceneGraph(d_model: int, max_entities: int = 32)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update` | `(entity_embeddings: Tensor, positions: Tensor) -> list[tuple]` | Spatial relations | Update with `(N, d_model)` embeddings and `(N, 3)` positions. Returns `(i, j, RelationType, confidence)` tuples. |
| `compute_spatial_bias` | `(query_vec: Tensor) -> Tensor` | `(d_model,)` | Compute spatial attention bias for reasoning chain routing. |
| `inject_into_transitions` | `(transitions, tier2) -> None` | — | Add `SPATIAL_*` edges to the transition structure from current relations. |

The scene graph is automatically updated during `process()` when `metadata` contains `entity_positions` and `entity_embeddings`. It is environment-agnostic — any bridge providing entity position data gets spatial reasoning.

---

### GoalRegister

**Module:** `sfp.goals.persistence`

32-slot hierarchical goal register with cosine-similarity progress tracking, deadline management, and reasoning bias.

```python
GoalRegister(config: GoalPersistenceConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `create_goal` | `(instruction_embedding, importance=0.5, urgency=0.5, deadline=None, ttl=None, parent_id=None) -> Goal` | Goal | Create new goal |
| `decompose_goal` | `(goal_id: int) -> list[Goal]` | Subgoals | Decompose into subgoals |
| `update_progress` | `(goal_id, query_embedding) -> float` | Delta | Update goal progress |
| `compute_priorities` | `() -> None` | — | Recompute all priorities |
| `get_goal_context` | `() -> Tensor` | `(d_model,)` | Aggregated goal context |
| `get_reasoning_bias` | `(basin_keys, n_active) -> Tensor` | `(n_active,)` | Bias scores for basins |
| `check_deadlines` | `() -> list[tuple[int, str]]` | Warnings | Check deadline violations |
| `detect_stalled_goals` | `() -> list[int]` | Goal IDs | Detect stalled goals |
| `pause_goal` / `resume_goal` | `(goal_id) -> None` | — | Lifecycle management |
| `remove_goal` | `(goal_id) -> None` | — | Remove goal |
| `save` / `load` | `(path) -> None` | — | Persistence |

---

### MetacognitionEngine

**Module:** `sfp.metacognition.uncertainty`

4-source uncertainty estimation with ECE calibration.

```python
MetacognitionEngine(config: MetacognitionConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `estimate_retrieval_uncertainty` | `(attn_weights, basin_confidence, n_active) -> float` | `[0,1]` | From attention entropy |
| `estimate_chain_uncertainty` | `(chain_trace: list[ChainTrace]) -> float` | `[0,1]` | From reasoning trace |
| `estimate_prediction_uncertainty` | `(wm_state: WorldModelState) -> float` | `[0,1]` | From world model errors |
| `estimate_knowledge_uncertainty` | `(confidence, maturity, modality_coverage) -> float` | `[0,1]` | From basin maturity |
| `compose_uncertainty` | `(r, c, p, k, context) -> UncertaintyEstimate` | UncertaintyEstimate | Compose 4 sources |
| `record_activation` | `(basin_ids: list[int], confidence_values: list[float] \| None) -> None` | — | Record basin activations |
| `get_ece` | `() -> float` | ECE score | Expected Calibration Error |
| `update_calibration` | `(predicted_confidence, actual_correct) -> None` | — | Update calibration bins |
| `get_health_report` | `(tier2) -> dict` | Dict | Memory health metrics |
| `get_info_seeking_suggestions` | `(uncertainty, tier2) -> list[str]` | Suggestions | Info-seeking prompts |

---

### ValenceSystem

**Module:** `sfp.affect.valence`

Affective valence with 3-timescale mood tracking, safety-biased reasoning, and basin/edge hedonic annotation.

```python
ValenceSystem(config: ValenceConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `compute_valence` | `(embedding, reward=0.0, user_feedback=0.0, goal_alignment=0.0, prediction_satisfaction=0.0) -> ValenceSignal` | ValenceSignal | Compute valence for input |
| `annotate_basin` | `(basin_id, scalar_valence, valence_embedding) -> None` | — | Annotate basin with valence |
| `annotate_edge` | `(edge_id, scalar_valence) -> None` | — | Annotate edge with arousal |
| `get_chain_valence` | `(basin_sequence: list[int]) -> float` | Scalar | Compute chain traversal valence |
| `get_reasoning_mode` | `(valence: ValenceSignal) -> str` | "approach"/"avoidance"/"neutral" | Determine reasoning mode |
| `get_reasoning_valence_bias` | `(mode, basin_ids) -> Tensor` | `(N,)` | Basin bias from valence |
| `get_surprise_threshold_modifier` | `(valence) -> float` | `[0.7, 1.0]` | Threshold modifier |
| `get_retention_priority` | `(basin_id) -> float` | Priority | Memory retention scoring |
| `get_consolidation_weights` | `(basin_ids) -> Tensor` | `(N,)` | Consolidation priorities |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `mood` | `tuple[float, float, float]` | (immediate, short_term, baseline) |
| `basin_valence_scalar` | `Tensor` | `(4096,)` per-basin valence |
| `basin_valence_embedding` | `Tensor` | `(4096, d_val)` per-basin vectors |
| `edge_valence` | `Tensor` | `(40960,)` per-edge valence |

---

### SalienceGate

**Module:** `sfp.attention.salience`

Pre-Perceiver salience filter that decides Skip/Skim/Full processing per modality.

```python
SalienceGate(config: SelectiveAttentionConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `evaluate` | `(inputs, goal_context=None, world_model_prediction=None, recent_activation=None, alarm_basins=None) -> SalienceResult` | SalienceResult | Evaluate salience |
| `skim_process` | `(embedding, modality, tier0) -> None` | — | Lightweight skim update |
| `run_hindsight_training` | `() -> None` | — | Train on skim buffer |

**Parameters for `evaluate`:**
- `inputs` — `dict[str, Tensor]` mapping modality name → embedding.
- `goal_context` — Optional `(d_model,)` goal context vector.
- `world_model_prediction` — Optional `(d_model,)` world model prediction.
- `recent_activation` — Optional `(d_model,)` recent basin activation summary.
- `alarm_basins` — Optional `set[int]` of basin IDs that trigger immediate FULL.

---

### GenerativeReplay

**Module:** `sfp.memory.replay`

Off-critical-path synthetic episode generation with three strategies: basin interpolation, chain dreaming, boundary exploration.

```python
GenerativeReplay(config: GenerativeReplayConfig | None = None, d_model: int = 512)
```

#### Methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `should_generate` | `(cycle_count, total_episodes, idle_seconds=0.0) -> (bool, int)` | (should, count) | Check generation schedule |
| `generate_batch` | `(n, tier2, transitions=None, backbone=None, tier3=None, valence_system=None) -> list[SyntheticEpisode]` | Synthetic episodes | Generate validated synthetics |
| `record_inference` | `() -> None` | — | Record real inference event |
| `get_generation_stats` | `() -> dict` | Stats dict | Generation statistics |
| `reset_stats` | `() -> None` | — | Reset statistics |

---

## 9. Defense

### InputSanitizer

**Module:** `sfp.defense.input_validation`

Defense Layer 1: L2 norm clamping, Gaussian smoothing, SHA-256 provenance tracking.

```python
InputSanitizer(
    max_norm: float = 10.0,
    smoothing_sigma: float = 0.01,
    provenance_log_size: int = 10000,
)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `sanitize` | `(input_tensor, modality="tensor") -> Tensor` | Sanitized tensor | Clamp norm + smooth |
| `record_provenance` | `(raw_bytes, modality, source="") -> bytes` | SHA-256 hash | Record provenance |

---

### EmbeddingAnomalyDetector

**Module:** `sfp.defense.input_validation`

Defense Layer 2: Mahalanobis-distance anomaly detection with per-modality statistics.

```python
EmbeddingAnomalyDetector(
    d_model: int = 512,
    threshold: float = 3.0,
    warmup_samples: int = 100,
)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `update_statistics` | `(embedding, modality) -> None` | — | Update running mean/covariance |
| `is_anomalous` | `(embedding, modality) -> bool` | Bool | Check if anomalous |
| `get_statistics` | `(modality) -> dict` | Stats dict | Get mean, n_samples, etc. |

---

### SurpriseHardener

**Module:** `sfp.defense.surprise_hardening`

Defense Layers 2-3: EMA-based surprise clamping, rate limiting, dual-path verification, per-parameter adaptive gradient clipping.

```python
SurpriseHardener(config: Tier0Config)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `compute_hardened_surprise` | `(raw_grad_norm, latent_distance=None) -> float` | Hardened ratio | Apply all defenses |
| `clip_gradients` | `(named_parameters) -> float` | Clip ratio | Adaptive per-param clipping |
| `reset` | `() -> None` | — | Reset state |

| Property | Type | Description |
|----------|------|-------------|
| `is_rate_limited` | `bool` | Currently rate-limited? |
| `surprise_ema` | `float` | Current surprise EMA |

---

### AdaptiveGradientClipper

**Module:** `sfp.defense.gradient_bounds`

Defense Layer 3: Per-parameter ARC-style gradient clipping with EMA tracking.

```python
AdaptiveGradientClipper(model: nn.Module, clip_multiplier: float = 2.0, ema_decay: float = 0.99)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `clip` | `(model: nn.Module) -> float` | Fraction clipped | Apply adaptive clipping |

---

### UpdateBudget

**Module:** `sfp.defense.gradient_bounds`

Defense Layer 3: Per-step weight change L2 budget. Rolls back if budget exceeded.

```python
UpdateBudget(model: nn.Module, budget_fraction: float = 0.005)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `enforce` | `(model: nn.Module) -> bool` | Exceeded? | Enforce budget, rollback if needed |

---

### AnchorVerifier

**Module:** `sfp.defense.anchor_verification`

Defense Layer 5: Verify that known concept anchors still map to expected basins.

```python
AnchorVerifier(
    anchor_exemplars: torch.Tensor,      # (N, d_model)
    expected_basins: torch.Tensor,        # (N,)
    drift_threshold: float = 0.3,
    pairwise_tolerance: float = 0.2,
)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `verify` | `(tier2: EssentialMemory) -> list[str]` | Violations | Empty list = OK |

| Property | Type | Description |
|----------|------|-------------|
| `n_anchors` | `int` | Number of anchor concepts |

---

### ManifoldIntegrityMonitor

**Module:** `sfp.defense.topology_monitor`

Defense Layer 5: Topological manifold monitoring — basin merging, component changes, reasoning loops.

```python
ManifoldIntegrityMonitor(config: DefenseConfig | None = None)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `check_basin_integrity` | `(tier2) -> list[str]` | Alerts | Basin merge/isolation detection |
| `check_transition_integrity` | `(transitions, tier2) -> list[str]` | Alerts | Implausible edge detection |
| `detect_reasoning_loops` | `(transitions) -> list[list[int]]` | Cycle paths | DFS cycle detection |

---

## 10. Architecture

### PerceiverIO

**Module:** `sfp.core.perceiver`

Multi-modal bottleneck that compresses variable-length inputs to fixed-size latents via cross-attention.

```python
PerceiverIO(config: PerceiverConfig)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `forward` | `(x: Tensor) -> Tensor` | `(B, n_latents, d_latent)` | Compress to latents |
| `decode` | `(latents, output_queries) -> Tensor` | `(B, M, d_out)` | Decode via output cross-attention |

---

### BackboneTransformer

**Module:** `sfp.core.backbone`

Shared transformer encoder processing Perceiver output or raw embeddings.

```python
BackboneTransformer(config: BackboneConfig)
```

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `forward` | `(x: Tensor) -> Tensor` | `(B, seq, d_model)` | Process through transformer |

---

## 11. Configuration

All configurations are frozen dataclasses defined in `sfp.config`.

### FieldConfig
```python
@dataclass(frozen=True)
class FieldConfig:
    dim: int = 512              # Hidden dimension
    n_layers: int = 6           # MLP depth
    activation: str = "gelu"    # "gelu" | "relu" | "tanh"
    use_layernorm: bool = True  # Layer normalization
    residual: bool = False      # Residual connections

    @staticmethod
    def from_preset(size: FieldSize) -> FieldConfig
    # TINY: 256d, 4L | SMALL: 512d, 6L | MEDIUM: 1024d, 8L | LARGE: 2048d, 8L
```

### StreamingConfig
```python
@dataclass(frozen=True)
class StreamingConfig:
    lr: float = 1e-4                    # Learning rate
    weight_decay: float = 0.001         # Weight decay
    surprise_threshold: float = 0.1     # Minimum surprise for update
    momentum: float = 0.9              # Surprise EMA momentum
    adaptive_surprise: bool = False     # Use percentile-based threshold
    surprise_percentile: float = 0.9    # Percentile for adaptive threshold
    loss_fn: str = "mse"               # "mse" | "cosine"
```

### Tier0Config
```python
@dataclass(frozen=True)
class Tier0Config:
    surprise_momentum: float = 0.9      # EMA momentum for surprise
    max_surprise_ratio: float = 3.0     # Maximum surprise ratio clamp
    rate_limit_threshold: float = 0.3   # Rate limit activation threshold
    rate_limit_window: int = 100        # Window size for rate limiting
    clip_multiplier: float = 2.0        # Adaptive clip multiplier
    consolidation_threshold: float = 1.5
```

### Tier1Config
```python
@dataclass(frozen=True)
class Tier1Config:
    hot_capacity: int = 2000           # Hot buffer capacity
    cold_capacity: int = 8000          # Cold buffer capacity
    surprise_threshold: float = 0.5    # Minimum surprise for admission
    dedup_threshold: float = 0.95      # Cosine similarity dedup gate
    min_per_basin: int = 5             # Minimum episodes per basin
    eviction_batch_size: int = 100     # Eviction batch size
```

### Tier2Config
```python
@dataclass(frozen=True)
class Tier2Config:
    n_slots: int = 4096               # Maximum concept basins
    d_value: int = 512                # Value dimension
    n_heads: int = 8                  # Attention heads
    temperature: float = 1.0          # Softmax temperature
    consistency_threshold: float = 0.3 # Consistency check threshold
```

### Tier3Config
```python
@dataclass(frozen=True)
class Tier3Config:
    n_slots: int = 512                 # Maximum core axioms
    d_value: int = 512                 # Value dimension
    min_confidence: float = 0.9        # Minimum confidence for promotion
    min_episode_count: int = 1000      # Minimum episode evidence
    min_modalities: int = 2            # Minimum modality diversity
    min_age_days: float = 7.0          # Minimum age before promotion
```

### ConsolidationConfig
```python
@dataclass(frozen=True)
class ConsolidationConfig:
    mini_interval: int = 100           # Steps between mini consolidations
    standard_interval: int = 1000      # Steps between standard
    deep_interval: int = 10000         # Steps between deep
    replay_batch_size: int = 32        # Episodes sampled per replay
    n_replay_steps: int = 100          # Replay steps per consolidation
    representation_threshold: float = 0.1
    new_concept_threshold: int = 5     # Min candidates to create basin
    distillation_threshold: float = 0.5 # Cosine sim for basin matching
```

### TransitionConfig
```python
@dataclass(frozen=True)
class TransitionConfig:
    d_relation: int = 64               # Relation embedding dimension
    max_edges: int = 40960             # Maximum graph edges
    n_relation_types: int = 6          # Number of relation types
```

### ReasoningChainConfig
```python
@dataclass(frozen=True)
class ReasoningChainConfig:
    max_hops: int = 7                  # Maximum chain length
    convergence_threshold: float = 0.01 # Query residual convergence
    context_decay: float = 0.85        # Per-hop context decay
    query_retention: float = 0.3       # Original query retention
    branch_threshold: float = 0.4      # Score threshold for branching
    max_branches: int = 3              # Maximum simultaneous branches
    shortcut_min_traversals: int = 10  # Min traversals for shortcut
```

### DefenseConfig
```python
@dataclass(frozen=True)
class DefenseConfig:
    security_level: str = "normal"          # "high" | "normal" | "accelerated"
    embedding_anomaly_threshold: float = 3.0 # Mahalanobis threshold
    dual_path_verification: bool = True
    topology_check_interval: int = 1000
    anchor_verification: bool = True
    merge_alarm_threshold: int = 5
    topology_change_threshold: int = 3
```

### WorldModelConfig
```python
@dataclass(frozen=True)
class WorldModelConfig:
    d_deterministic: int = 512         # GRU hidden size
    d_stochastic_categories: int = 32  # Categorical distributions
    d_stochastic_classes: int = 32     # Classes per category
    d_observation: int = 512           # Input observation dimension
    d_reward: int = 1                  # Reward dimension
    n_projection_steps: int = 4        # Multi-step projection horizon
    kl_free_nats: float = 1.0         # Free nats for KL
    kl_weight: float = 0.1            # KL loss weight
    reconstruction_weight: float = 1.0
    prediction_error_weight: float = 0.4
    kl_divergence_weight: float = 0.3
    reconstruction_error_weight: float = 0.3
    n_subspace_projections: int = 8   # Directional error subspaces
    cache_size: int = 8               # Pre-activation cache slots
    cache_match_threshold: float = 0.8
    symlog_epsilon: float = 1e-3
```

### GoalPersistenceConfig
```python
@dataclass(frozen=True)
class GoalPersistenceConfig:
    max_goals: int = 32                # Maximum concurrent goals
    d_goal: int = 512                  # Goal embedding dimension
    d_satisfaction: int = 512          # Satisfaction embedding dimension
    max_subgoals: int = 4              # Max subgoals per decomposition
    progress_ema_decay: float = 0.95
    satisfaction_threshold: float = 0.95 # Auto-complete threshold
    stall_threshold: float = 0.01      # Stall detection threshold
    stall_steps: int = 50              # Steps before stall alert
    deadline_warning_ratio: float = 0.8
    reasoning_bias: float = 0.3        # Goal → reasoning bias strength
    salience_boost: float = 0.2        # Goal → salience boost
    ttl_default: float = 3600.0        # Default TTL seconds
    priority_mlp_hidden: int = 64
```

### MetacognitionConfig
```python
@dataclass(frozen=True)
class MetacognitionConfig:
    d_uncertainty_embedding: int = 64
    n_calibration_bins: int = 10
    estimator_hidden: int = 32
    confidence_threshold_high: float = 0.8
    confidence_threshold_low: float = 0.3
    seeking_max_alternatives: int = 3
    health_dormant_threshold: float = 0.01
    health_decline_window: int = 100
```

### ValenceConfig
```python
@dataclass(frozen=True)
class ValenceConfig:
    d_valence_embedding: int = 32      # Valence embedding dimension
    rl_value_weight: float = 0.5       # Reward signal weight
    user_feedback_weight: float = 0.25
    goal_alignment_weight: float = 0.15
    prediction_satisfaction_weight: float = 0.10
    learned_blend: float = 0.4         # Neural vs rule-based blend
    immediate_tau: float = 0.5         # Fast mood EMA
    short_term_tau: float = 0.99       # Medium mood EMA
    baseline_tau: float = 0.9999       # Slow mood EMA
    mood_weights: tuple = (0.3, 0.5, 0.2) # Composite mood weights
    approach_weight: float = 0.2       # Approach bias strength
    avoidance_weight: float = 0.4      # Avoidance bias strength
    basin_valence_ema_decay: float = 0.95
    edge_valence_ema_decay: float = 0.95
```

### SelectiveAttentionConfig
```python
@dataclass(frozen=True)
class SelectiveAttentionConfig:
    n_modalities: int = 5
    modality_names: tuple = ("text", "vision", "audio", "sensor", "structured")
    skip_threshold: float = 0.1        # Below → SKIP
    skim_threshold: float = 0.4        # Below → SKIM, above → FULL
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
```

### GenerativeReplayConfig
```python
@dataclass(frozen=True)
class GenerativeReplayConfig:
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
    warmup_episodes: int = 1000        # Episodes before replay starts
    middle_episodes: int = 10000
    middle_cycle_interval: int = 5
    middle_synthetics: int = 8
    mature_cycle_interval: int = 3
    mature_synthetics: int = 16
    idle_timeout_seconds: float = 30.0
    backbone_coherence_threshold: float = 0.5
```

### Other Configs
```python
@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 1.0
    merge_threshold: float = 0.5
    enabled: bool = True

@dataclass(frozen=True)
class EWCConfig:
    lambda_: float = 1000.0
    decay: float = 0.999
    enabled: bool = True

@dataclass(frozen=True)
class AttractorConfig:
    max_iterations: int = 20
    step_size: float = 0.7
    tolerance: float = 1e-4
    return_trajectory: bool = False

@dataclass(frozen=True)
class PerceiverConfig:
    n_latents: int = 256
    d_latent: int = 512
    d_input: int = 512
    n_cross_attn_layers: int = 2
    n_self_attn_layers: int = 4
    n_heads: int = 8
    dropout: float = 0.0

@dataclass(frozen=True)
class BackboneConfig:
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1

@dataclass(frozen=True)
class QuantizationConfig:
    weight_bits: int = 8
    activation_bits: int = 16
    per_channel: bool = True

@dataclass(frozen=True)
class CommConfig:
    preferred_layer: CommLayer = CommLayer.L2_MANIFOLD_COORD
    surprise_threshold: float = 0.1
    compression_method: str = "topk"
    compression_density: float = 0.01

@dataclass(frozen=True)
class HardwareProfile:
    name: str = "rtx3060"
    vram_budget_mb: int = 12_000
    target_latency_ms: float = 100.0
    fp16_capable: bool = True
```

---

## 12. Types & Enums

All defined in `sfp.types`.

### Enums

```python
class FieldSize(Enum):
    TINY = "tiny"       # 256d, 4L
    SMALL = "small"     # 512d, 6L
    MEDIUM = "medium"   # 1024d, 8L
    LARGE = "large"     # 2048d, 8L

class CommLayer(Enum):
    L0_RAW_TEXT = 0
    L1_EMBEDDING = 1
    L2_MANIFOLD_COORD = 2
    L3_DEFORMATION = 3
    L4_SURPRISE_GATED = 4

class MemoryTier(Enum):
    WORKING = 0         # Tier 0
    EPISODIC = 1        # Tier 1
    ESSENTIAL = 2       # Tier 2
    CORE = 3            # Tier 3

class ConsolidationMode(Enum):
    MINI = "mini"           # Tier 0 → 1
    STANDARD = "standard"   # Tier 1 → 2
    DEEP = "deep"           # Tier 2 → 3

class RelationType(Enum):
    CAUSAL = 0
    TEMPORAL = 1
    COMPOSITIONAL = 2
    ANALOGICAL = 3
    INHIBITORY = 4
    ASSOCIATIVE = 5
    SPATIAL_NEAR = 6
    SPATIAL_APPROACHING = 7
    SPATIAL_FLEEING = 8
    SPATIAL_ABOVE = 9

class SecurityLevel(Enum):
    HIGH = "high"
    NORMAL = "normal"
    ACCELERATED = "accelerated"

class GoalStatus(Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"

class ProcessingLevel(Enum):
    SKIP = "skip"       # Discard input
    SKIM = "skim"       # Lightweight update
    FULL = "full"       # Full pipeline

class ReplayStrategy(Enum):
    BASIN_INTERPOLATION = "interpolation"
    CHAIN_DREAMING = "chain_dreaming"
    BOUNDARY_EXPLORATION = "boundary"
```

### Frozen Dataclasses (Immutable Results)

#### SurpriseMetric
```python
@dataclass(frozen=True)
class SurpriseMetric:
    grad_norm: float        # Gradient L2 norm
    loss: float             # Reconstruction loss
    updated: bool           # Whether weights were updated
    timestamp: float = 0.0
```

#### AttractorResult
```python
@dataclass(frozen=True)
class AttractorResult:
    point: torch.Tensor                          # Converged attractor point
    iterations: int                               # Iterations to converge
    converged: bool                               # Whether convergence was reached
    basin_id: int | None = None                   # Basin assignment
    trajectory: list[torch.Tensor] | None = None  # Optional trajectory
```

#### ReasoningResult
```python
@dataclass
class ReasoningResult:
    knowledge: torch.Tensor         # Accumulated knowledge vector
    n_hops: int                     # Number of hops taken
    visited_basins: list[int]       # Basin IDs visited
    terminated_reason: str          # "convergence" | "dead_end" | "cycle" | "max_hops" | "empty_memory"
    routing: str                    # "single_hop" | "multi_hop"
    chain_weight: float = 1.0
    trace: list[ChainTrace] = field(default_factory=list)
```

#### ChainTrace
```python
@dataclass(frozen=True)
class ChainTrace:
    hop: int                        # Hop number
    basin_id: int | None            # Current basin
    event_type: str                 # "start" | "hop" | "terminate" | "branch"
    confidence: float = 0.0
    score: float = 0.0
    edge_relation: str = ""
    n_branches: int = 0
    knowledge_norm: float = 0.0
    query_residual_norm: float = 0.0
```

#### ValenceSignal
```python
@dataclass(frozen=True)
class ValenceSignal:
    scalar_valence: float           # Combined scalar valence [-1, 1]
    valence_embedding: torch.Tensor # (d_val,) embedding
    projected_context: torch.Tensor # Projected context vector
    mood_immediate: float           # Fast mood EMA
    mood_short_term: float          # Medium mood EMA
    mood_baseline: float            # Slow mood EMA
    composite_mood: float           # Weighted composite
    risk_tolerance: float           # [0, 1]
    vigilance: float                # [0, 1]
```

#### UncertaintyEstimate
```python
@dataclass(frozen=True)
class UncertaintyEstimate:
    retrieval_uncertainty: float     # [0, 1]
    chain_uncertainty: float         # [0, 1]
    prediction_uncertainty: float    # [0, 1]
    knowledge_uncertainty: float     # [0, 1]
    composite_embedding: torch.Tensor
    scalar_confidence: float         # [0, 1] combined confidence
    calibrated: bool = False
```

#### SalienceResult
```python
@dataclass(frozen=True)
class SalienceResult:
    level: ProcessingLevel          # SKIP | SKIM | FULL
    salience_scores: dict[str, float]  # Per-modality scores
    combined_salience: float
    interrupt: bool = False
    interrupt_reason: str = ""
```

#### WorldModelState
```python
@dataclass(frozen=True)
class WorldModelState:
    deterministic: torch.Tensor      # (d_deterministic,)
    stochastic: torch.Tensor         # (d_stoch_flat,)
    prediction_error: float = 0.0
    kl_divergence: float = 0.0
    reconstruction_error: float = 0.0
```

#### SyntheticEpisode
```python
@dataclass(frozen=True)
class SyntheticEpisode:
    embedding: torch.Tensor
    source_basins: list[int]
    strategy: ReplayStrategy
    validation_passed: bool
    weight: float
    is_synthetic: bool = True
```

#### PromotionRequest
```python
@dataclass(frozen=True)
class PromotionRequest:
    basin_id: int
    confidence: float
    episode_count: int
    modality_count: int
    age_days: float
    key_snapshot: torch.Tensor
    value_snapshot: torch.Tensor
```

#### HealthReport
```python
@dataclass(frozen=True)
class HealthReport:
    attractor_count: int
    mean_basin_radius: float
    topological_complexity: dict[str, int]
    information_density: float
    spectral_gap: float
    timestamp: float = 0.0
```

#### TopologySnapshot
```python
@dataclass(frozen=True)
class TopologySnapshot:
    timestamp: float
    betti_numbers: tuple[int, ...]
    persistence_diagram: np.ndarray
    total_persistence: float
```

### Mutable Dataclasses

#### Episode (Tier 1)
```python
@dataclass
class Episode:
    id: int
    timestamp: float
    modality: str
    provenance_hash: bytes
    input_embedding: torch.Tensor
    working_memory_state: torch.Tensor
    logit_snapshot: torch.Tensor
    surprise_at_storage: float
    attractor_basin_id: int
    attractor_distance: float
    preceding_episode_id: int | None
    following_episode_id: int | None
    integrity_hash: bytes
    weight_hash_at_storage: bytes
    consolidation_count: int = 0
    last_consolidated: float = 0.0
    flagged: bool = False
    valence: float = 0.0
    spatial_position: tuple[float, float, float] | None = None
    spatial_orientation: tuple[float, float] | None = None
```

#### Goal
```python
@dataclass
class Goal:
    id: int
    embedding: torch.Tensor
    satisfaction_embedding: torch.Tensor
    description_hash: bytes = b""
    priority: float = 0.5
    urgency: float = 0.5
    importance: float = 0.5
    progress: float = 0.0
    status: GoalStatus = GoalStatus.ACTIVE
    parent_id: int | None = None
    child_ids: list[int] = field(default_factory=list)
    dependency_ids: list[int] = field(default_factory=list)
    created_at: float = 0.0
    deadline: float | None = None
    ttl: float = 3600.0
    last_progress_update: float = 0.0
    progress_history: list[float] = field(default_factory=list)
```

### Protocols

```python
@runtime_checkable
class InputEncoder(Protocol):
    @property
    def output_dim(self) -> int: ...
    def encode(self, inputs: list[Any]) -> torch.Tensor: ...

@runtime_checkable
class ForgetStrategy(Protocol):
    def penalty(self, model: torch.nn.Module) -> torch.Tensor: ...
    def update_importance(self, model: torch.nn.Module) -> None: ...

@runtime_checkable
class ConsistencyChecker(Protocol):
    def check_consistency(self, input_embedding: Tensor, proposed_update: Tensor) -> Tensor: ...

@runtime_checkable
class AuthorizationHandler(Protocol):
    def authorize(self, request: PromotionRequest) -> bool: ...
```

---

## 13. Integrity

**Module:** `sfp.memory.integrity`

Cryptographic hashing utilities for episode and model integrity.

```python
def compute_weight_hash(model: torch.nn.Module) -> bytes
```
Compute SHA-256 hash over all model parameters. Returns 32-byte digest.

```python
def compute_episode_hash(
    input_embedding: torch.Tensor,
    logit_snapshot: torch.Tensor,
    weight_hash: bytes,
) -> bytes
```
Compute integrity hash for an episode. Returns 32-byte digest.

```python
def verify_episode_integrity(
    input_embedding: torch.Tensor,
    logit_snapshot: torch.Tensor,
    weight_hash: bytes,
    stored_hash: bytes,
) -> bool
```
Verify an episode's integrity hash. Returns `True` if valid.

---

## Processing Pipeline Flow

```
Input
  │
  ▼
┌─────────────────┐
│  Salience Gate   │ → SKIP (discard) / SKIM (buffer) / FULL (continue)
└────────┬────────┘
         ▼
┌─────────────────┐
│ Input Sanitizer  │ L2 norm clamping + Gaussian smoothing
│ Anomaly Detector │ Mahalanobis distance check
└────────┬────────┘
         ▼
┌─────────────────┐
│  Perceiver IO    │ Variable-length → fixed latents (optional)
│    Backbone      │ Transformer contextual processing (optional)
└────────┬────────┘
         ▼
┌─────────────────┐
│  World Model     │ RSSM train step → enhanced surprise
│  Goal Register   │ Progress update → reasoning bias
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tier 3 Retrieve  │ Core axiom knowledge
│ Tier 2 Retrieve  │ Hopfield retrieval → reasoning chain
│ Reasoning Router │ Single-hop vs multi-hop decision
│ Reasoning Chain  │ Multi-hop graph traversal
└────────┬────────┘
         ▼
┌─────────────────┐
│  Valence System  │ Hedonic annotation + mood update
│  Metacognition   │ 4-source uncertainty estimation
└────────┬────────┘
         ▼
┌─────────────────┐
│ Tier 0 (Field)   │ Surprise-gated weight update
│ Surprise Hardener│ Clamping + rate limiting + dual-path
│ Gradient Clipper │ Per-parameter adaptive clipping
│ Update Budget    │ L2 budget enforcement
└────────┬────────┘
         ▼
┌─────────────────┐
│  Episode Storage │ Maybe store in Tier 1 (surprise + dedup + integrity gates)
│  Consolidation   │ Auto-check: mini (T0→1), standard (T1→2), deep (T2→3)
│  Shortcut Learn  │ Observe chain → create shortcuts
│  Gen. Replay     │ Off-path synthetic generation
└─────────────────┘
```
