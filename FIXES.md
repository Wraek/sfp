# SFP Pipeline Fixes Log

Fixes applied during the bridge integration debugging sessions. The initial symptom was that all saved sessions had `step_count=0` and zero episodes despite 16+ minutes of connectivity. Each fix revealed the next deeper issue in a cascading chain.

---

## Fix 1: Bridge Logging Visibility

**Status:** Deployed
**File:** `src/sfp/gui/app.py` — `_setup_logging()`
**Root cause:** The GUI log handler was only attached to `logging.getLogger("sfp")`, but the bridge uses `"sfp-bridge.*"` loggers. In Python's logging hierarchy, `"sfp-bridge"` is a **sibling** of `"sfp"`, not a child — so all bridge exceptions were invisible in the GUI.
**Fix:** Added `logging.getLogger("sfp-bridge").addHandler(handler)` alongside the existing `"sfp"` handler.
**Impact:** This single line was the key that unlocked all subsequent discoveries — every `process()` crash was now visible.

---

## Fix 2: step_count Always Increments

**Status:** Deployed
**File:** `src/sfp/memory/processor.py`
**Root cause:** When the salience gate returned `SKIP` or `SKIM`, `process()` returned early **before** `self._step_count += 1`. Result: `step_count` stayed at 0 regardless of how many times `process()` was called.
**Fix:** Moved `self._step_count += 1` to the very top of `process()` (unconditional increment). Added `_salience_stats` dict (`{"skip": 0, "skim": 0, "full": 0}`) to track the gate's decisions.

---

## Fix 3: Queue Exception Logging

**Status:** Deployed
**File:** `sfpmc/bridge/bridge_server.py` — `_put_dropping()`
**Root cause:** `except Exception: pass` silently swallowed all queue insertion failures.
**Fix:** Changed to `logger.warning("Queue put failed", exc_info=True)`.

---

## Fix 4: Salience Stats in Status and Checkpoint

**Status:** Deployed
**Files:** `src/sfp/interface.py`, `src/sfp/storage/serialization.py`
**Root cause:** Salience gate decisions (skip/skim/full counts) were tracked in processor but not exposed through the `status()` API or persisted in session checkpoints.
**Fix:** Added `salience_stats` to `interface.status()` return dict and to `SessionCheckpoint` save/load.

---

## Fix 5: Pipeline Health Counters in Inference Loop

**Status:** Deployed
**File:** `sfpmc/bridge/inference_loop.py`
**Root cause:** No instrumentation around the `process()` call — failures were caught and logged but not counted.
**Fix:** Added `_obs_received`, `_process_ok`, `_process_err` counters. Wrapped `process()` call in try/except with counting. Added periodic pipeline health logging every 200 ticks.

---

## Fix 6: Bridge Status Enrichment

**Status:** Deployed
**Files:** `sfpmc/bridge/bridge_server.py`, `src/sfp/gui/status_panel.py`
**Root cause:** Bridge status dict didn't include pipeline health counters, and the GUI had no display for them.
**Fix:** Added `obs_received`, `process_ok`, `process_err`, `obs_queue_depth` to bridge `status()`. Added `{ok}/{obs}proc` and `{err}err` display to the GUI bridge status line.

---

## Fix 7: None attn_weights in Metacognition

**Status:** Deployed
**File:** `src/sfp/metacognition/uncertainty.py` — `estimate_retrieval_uncertainty()`
**Root cause:** `processor.py` passed `attn_weights=None` to this method, which unconditionally called `attn_weights.float()`. This crashed **every** `process()` call with `AttributeError: 'NoneType' object has no attribute 'float'`.
**Fix:** Added None guard — when `attn_weights is None`, defaults to `norm_entropy = 1.0` (maximum entropy, i.e., maximum retrieval uncertainty). This is semantically correct: no attention data means we should assume high uncertainty.

---

## Fix 8: Salience Gate Context Dimension Mismatch

**Status:** Deployed
**File:** `src/sfp/attention/salience.py` — `evaluate()`
**Root cause:** The `context_aggregator` is `Linear(d_model * 3, d_context)` and expects three `d_model`-sized context vectors. But `goal_context` comes from `GoalsConfig.d_goal=512` and `world_model_prediction` from `WorldModelConfig.d_deterministic=512` — both hardcoded independently of `d_model`. For `d_model=1024`: input was `512 + 1024 + 1024 = 2560` instead of the expected `3072`.
**Fix:** Added `_ensure_d_model()` helper that zero-pads or truncates context vectors to `d_model` before concatenation.

---

## Fix 9: CUDA-Enabled PyTorch

**Status:** Deployed
**Files:** No code changes — pip reinstall only
**Root cause:** Both the dev environment (Python 3.14) and buildenv (Python 3.12) had `torch 2.10.0+cpu` (CPU-only wheel). The system has an RTX 2070 (8GB VRAM) that was completely unused. Tick times were ~1000ms vs the 45ms target.
**Fix:** Reinstalled `torch==2.10.0+cu126` from `https://download.pytorch.org/whl/cu126` in both environments. The existing `resolve_device("auto")` code already checks `torch.cuda.is_available()` — no code changes needed.

---

## Fix 10: Periodic Health Report Polling in GUI

**Status:** Deployed
**File:** `src/sfp/gui/app.py`
**Root cause:** The GUI only requested `health_report()` once at session creation and after batch operations. During bridge processing, the Steps/Tiers/Modules/Memory displays went stale (Steps showed 0 despite hundreds of successful processes).
**Fix:** Added `_poll_health()` method that refreshes `health_report()` and `memory_footprint()` every 2 seconds while a session is active, mirroring the existing `_poll_bridge_status()` pattern. Polling starts on session create/load and stops on shutdown.

---

## Fix 11: LoRA Device Mismatch

**Status:** Deployed
**File:** `src/sfp/core/lora.py` — `LoRALinear.__init__()`
**Root cause:** LoRA A/B parameters were created on CPU (`torch.randn(...)` defaults to CPU), but the base Linear layer they wrap was already on CUDA (the field is moved to CUDA at `processor.py:165` before LoRA wrapping at `streaming.py:60`). This caused `RuntimeError: Expected all tensors to be on the same device` on every forward pass.
**Fix:** Create A/B on the same device as the base layer: `device = base.weight.device`, then `torch.randn(..., device=device)`.

---

## Fix 12: Episode Device Mismatch

**Status:** Deployed
**Files:** `src/sfp/memory/consolidation.py`, `src/sfp/storage/serialization.py`
**Root cause:** Two code paths violated the convention that episode tensors should always be on CPU:
- **consolidation.py:153-155** — Mini-consolidation created episodes from the field's weight summary (on CUDA) without calling `.cpu()`. These CUDA episodes mixed with CPU episodes from `_maybe_store_episode`, causing `torch.stack` to fail in the deduplication gate.
- **serialization.py:430-432** — Session loading moved episode tensors to the processor device (CUDA) with `.to(dev)`, violating the CPU convention.
**Fix:** Added `.cpu()` to consolidation episode tensors. Removed `.to(dev)` from session loading (tensors are already CPU from `torch.load(map_location="cpu")`).

---

## Fix 13: Performance & Resource Optimization

**Status:** Deployed
**Files:** Multiple (see sub-fixes below)
**Root cause:** Pipeline ran in FP32 with vanilla O(n²) attention and non-fused optimizer. Slow-tick threshold was too low for Medium field on RTX 2070. Budget exceeded messages logged at WARNING on every tick.

### Fix 13A: Reduce Log Noise
**File:** `sfpmc/bridge/inference_loop.py`
**Fix:** Raised slow-tick warning threshold from 100ms to 300ms (187ms is expected for Medium+Full on RTX 2070). Rate-limited budget-exceeded messages to every 100th occurrence and lowered to DEBUG level.

### Fix 13B: World Model Device Fix
**File:** `src/sfp/prediction/world_model.py`
**Fix:** Added `device=device` to `_prev_spatial_position` tensor creation (line 284), eliminating a per-tick CPU→GPU transfer when spatial data flows from the bridge.

### Fix 13C: Flash Attention
**Files:** `src/sfp/core/perceiver.py`, `src/sfp/core/backbone.py`
**Fix:** Replaced manual `(Q @ K.T) / scale → softmax → @ V` attention with `F.scaled_dot_product_attention()`. PyTorch auto-dispatches to the most efficient kernel available (flash attention, memory-efficient attention, or vanilla fallback). Expected ~2–4x faster attention in Perceiver + Backbone.

### Fix 13D: Fused AdamW Optimizer
**File:** `src/sfp/core/streaming.py`
**Fix:** Added `fused=True` to AdamW when CUDA parameters are detected. Uses a single CUDA kernel for the optimizer step instead of per-parameter CPU-side iteration. Expected ~20-30% faster weight updates.

### Fix 13E: Mixed Precision (AMP) for Perceiver + Backbone
**File:** `src/sfp/memory/processor.py`
**Fix:** Wrapped Perceiver IO + Backbone forward passes in `torch.amp.autocast("cuda")`. These are the largest compute bottlenecks (~70-90ms combined in FP32). Output is cast back to FP32 via `.float()` before downstream processing. Expected ~2x speedup on attention and matrix multiplies.

---

## Fix 14: Action Head Drift, Sensorimotor Feedback & Demo Training

**Status:** Deployed
**Files:** `sfpmc/bridge/minecraft_action_head.py`, `sfpmc/bridge/inference_loop.py`, `sfpmc/bridge/minecraft_tokenizer.py`, `sfpmc/bridge/demo_pipeline.py`, `sfpmc/bridge/bridge_server.py`
**Root cause:** The AI exhibited the same broken behavior every fresh session — slowly rotating downward until staring at the ground, then randomly pressing buttons. Four compounding issues:

1. **Camera head random init bias** — `Linear(32, 2)` with default Kaiming init produced a consistent directional bias. Even a small raw output of +0.3 → `tanh(0.3)*10` = 2.9°/tick → 58°/s at 20Hz → ground-locked in ~1.5s.
2. **Action head never trained** — `MinecraftActionHead` had no training loop. Demo pipeline stored episodes to SFP memory but never backpropped through the action head.
3. **Movement heads ≈ coin flip** — sigmoid of untrained logits ≈ 0.5 → Bernoulli sampling = random button mashing.
4. **No action feedback** — The 256-dim state vector had no "last_action" field, so the AI couldn't correlate its own motor commands with observed outcomes.

### Fix 14A: Zero-Init Action Head Output Layers
**File:** `sfpmc/bridge/minecraft_action_head.py`
**Fix:** Zero-initialized the camera head's final `Linear(32, 2)` weight and bias (`tanh(0)*10 = 0°/tick`, no drift). Zero-initialized movement and interaction head biases (sigmoid centered at 0.5, no directional bias).

### Fix 14B: Last-Action Sensorimotor Feedback
**Files:** `sfpmc/bridge/inference_loop.py`, `sfpmc/bridge/minecraft_tokenizer.py`
**Fix:** `InferenceLoop` stores the last action dict and injects it into `state_data["last_action"]` before tokenization. `_state_to_vector` encodes 13 action dims (7 movement bools, 2 camera floats normalized by /10, 3 interaction bools, 1 hotbar normalized by /9) into the state vector (~72 → ~85 dims of 256). The AI can now observe its own motor commands and learn action-outcome correlations.

### Fix 14C: Soft-Clamp Camera Pitch
**File:** `sfpmc/bridge/inference_loop.py`
**Fix:** Syncs cumulative pitch from game state each tick (authoritative source). After action sampling, attenuates `delta_pitch` by `max(0, 1 - (|pitch|/85)²)`. At 60° pitch → 50% attenuation; at 80° → 11%. Prevents ground/sky lock while allowing normal camera movement.

### Fix 14D: Supervised Action Head Training from Demonstrations
**Files:** `sfpmc/bridge/minecraft_action_head.py`, `sfpmc/bridge/demo_pipeline.py`, `sfpmc/bridge/bridge_server.py`
**Fix:** Added `MinecraftActionHead.train_on_demos()` — runs 5 epochs of Adam over (embedding, action) pairs with BCE loss for binary heads, MSE for camera, cross-entropy for hotbar. Added `DemoPipeline.extract_training_pairs()` to extract aligned (embedding, action) pairs from the recording buffer. Wired into `BridgeServer._on_mode_change()`: when exiting DEMONSTRATING mode, pairs are extracted before `end_session()` clears buffers, then the action head is trained. Metrics logged on completion.

---

## Fix 15: Text Encoder CUDA Generator Mismatch

**Status:** Deployed
**File:** `sfpmc/bridge/text_encoder.py`
**Root cause:** `encode_text()` created a `torch.Generator(device="cpu")` but passed `device=self._device` (CUDA) to `torch.randn()`. PyTorch requires the generator and output tensor to be on the same device.
**Fix:** Generate the random vector on CPU with the seeded generator, then move to device with `.to(self._device)`. Deterministic seeding is preserved since the generator still runs on CPU.

---

## Fix 16: Expanded Hotbar Inventory Encoding

**Status:** Deployed
**File:** `sfpmc/bridge/minecraft_tokenizer.py`
**Root cause:** The tokenizer encoded hotbar inventory as 9 floats (item count per slot, normalized by /64) but discarded item type, durability, and the `main_hand` field entirely. The AI knew "slot 3 has 32/64 of something" but couldn't distinguish a sword from dirt — making it impossible to learn item-appropriate behaviors like equipping weapons near hostiles or selecting food when hungry. The mod already sends full item data per slot (`item` registry ID, `count`, `durability`, `max_durability`).

### Fix 16A: Rename `_entity_type_hash` → `_registry_id_hash`
**Fix:** Renamed the MD5 hash function to reflect that it works on any `namespace:name` registry ID (entities, items, blocks), not just entity types. Updated docstring and the one call site in `_encode_entities()`.

### Fix 16B: Rich Per-Slot Hotbar Encoding (9 → 54 dims)
**Fix:** Replaced the 9-dim count-only encoding with 6 dims per hotbar slot (× 9 = 54 dims):
- **Item type hash** (4 dims): MD5 → 4 floats in [-1, 1], same pattern as entity type hash. All zeros for empty slots.
- **Count** (1 dim): stack_count / 64.0 (unchanged normalization).
- **Durability ratio** (1 dim): durability / max_durability. 1.0 for undamageable items at full health, 0.0 for empty slots.

### Fix 16C: Main Hand Item Type Hash (4 dims)
**Fix:** Added 4-dim encoding of the `main_hand` field (already sent by the mod but previously unused). Gives the AI a direct signal for the currently held item without needing to cross-reference the selected_slot index with hotbar slot hashes.

**Dim budget:** 9 dims replaced by 54 + 4 = 58 dims (+49 net). Total state vector usage: ~114 / 256 (142 free).

---

## Fix 17: Goal Completion Detection & Advancement

**Status:** Deployed
**Files:** `src/sfp/goals/persistence.py`, `src/sfp/config.py`, `sfpmc/bridge/inference_loop.py`, `sfpmc/bridge/bridge_server.py`
**Root cause:** The goal system never recognized completion for three compounding reasons: (1) the overlay displayed **priority** ("p=0.50"), not **progress**, hiding actual goal state; (2) completion detection required cosine similarity ≥ 0.95 to a **randomly-projected satisfaction embedding** that bore no semantic relationship to actual goal-satisfying states; (3) no code path created a replacement goal when one completed.

### Fix 17A: Display Progress in Overlay
**File:** `sfpmc/bridge/inference_loop.py`
**Fix:** Changed overlay from `Goal #0 (p=0.50)` (priority) to `Goal #0 (12%)` (actual completion progress).

### Fix 17B: Use Goal Embedding as Satisfaction Target
**File:** `src/sfp/goals/persistence.py`
**Fix:** Replaced `satisfaction_emb = self.satisfaction_encoder(instruction_embedding)` with `satisfaction_emb = goal_emb.clone()`. Progress now tracks cosine similarity between the AI's current state and the goal's own embedding — states that "look like" the goal mean progress. The untrained satisfaction encoder produced arbitrary targets that made completion effectively impossible.

### Fix 17C: Auto-Advance Exploration Goal on Completion
**Files:** `sfpmc/bridge/bridge_server.py`, `sfpmc/bridge/inference_loop.py`
**Fix:** Added `_check_goal_advancement()` to bridge_server — when the exploration goal's status is COMPLETED, it removes the old goal and creates a fresh one. Called every tick from inference_loop via a callback, giving the AI a continuous stream of goals.

### Fix 17D: Lower Satisfaction Threshold
**File:** `src/sfp/config.py`
**Fix:** Lowered `satisfaction_threshold` from 0.95 to 0.8. The original 0.95 required near-exact match to the satisfaction embedding, which was unrealistically high for open-ended exploration goals. 0.8 allows completion while still requiring meaningful similarity.

---

## Fix 18: Movement Vector Override — AI Horizontal Movement Broken

**Status:** Deployed
**File:** `sfpmc/mod/src/main/java/com/sfp/minecraft/mixin/KeyboardInputMixin.java`
**Root cause:** The `KeyboardInputMixin` injects at `@At("TAIL")` of `KeyboardInput.tick()`, setting `playerInput` with the AI's movement booleans. However, MC 1.21.4's `tick()` computes `movementForward` and `movementSideways` floats **before** the tail injection point — those floats are derived from keyboard state, not from `playerInput`. The game engine reads the float fields for actual walking, not the `playerInput` booleans. Result: jump/sneak/sprint worked (read from `playerInput` directly or via explicit setters), but forward/back/left/right were completely non-functional.
**Fix:** After setting `playerInput`, also compute and assign `movementForward` and `movementSideways` from the AI's action booleans. Inlines the `getMovementMultiplier` logic (private to `KeyboardInput`): `positive == negative ? 0 : positive ? 1 : -1`.

---

## Fix 19: Camera Locked — Zero Exploration Noise at Startup

**Status:** Deployed
**File:** `sfpmc/bridge/inference_loop.py`
**Root cause:** The camera head was zero-initialized (Fix 14A) to prevent drift: `tanh(0)*10 = 0°/tick`. Exploration noise is computed as `curiosity_weight * exploration_noise_max`, but `curiosity_weight` starts at 0.0 and only rises when world model surprise exceeds 1.0. Until surprise triggers, the camera outputs exactly `0.0, 0.0` every tick. Movement worked despite the same zero noise because Bernoulli sampling from `sigmoid(-2.0) ≈ 12%` provides inherent stochasticity — the camera path is purely deterministic with no sampling step.
**Fix:** Added a minimum exploration noise floor: `max(0.1, curiosity_weight) * exploration_noise_max`. Baseline of 0.05 noise translates to ±0.25°/tick camera jitter (×5.0 scale in action head) — enough for visible looking-around. Negligible effect on other heads which already have stochastic sampling against strong biases.

---

## Fix 20: Local Block Awareness & Crosshair Target Detail

**Status:** Deployed
**Files:** `sfpmc: GameStateExtractor.java`, `sfpmc: minecraft_tokenizer.py`
**Root cause:** The AI had no semantic awareness of surrounding blocks. The state vector contained zero information about block types, walkability, or hazards nearby. The crosshair target already sent block ID and distance from the mod, but the Python side only encoded a 3-dim one-hot. Position coordinates were normalized by /1000, compressing a 200-block journey into 0.2 — too small for meaningful spatial learning.

### Fix 20A: Enhanced Crosshair Target (3 → 8 dims, net +5)
**File:** `sfpmc: minecraft_tokenizer.py`
**Fix:** Extended the crosshair one-hot with a 4-dim block/entity type hash (reusing `_registry_id_hash()`) and 1-dim normalized distance (/5.0). The AI can now distinguish "looking at stone" from "looking at a crafting table" and know how far away the target is.

### Fix 20B: Ground Block (new, +5 dims)
**Files:** `sfpmc: GameStateExtractor.java`, `sfpmc: minecraft_tokenizer.py`
**Fix:** Added `ground_block` field — registry ID and material category of block at player feet-1. Encoded as type hash (4 dims) + category float (1 dim). Immediate ground awareness: stone vs. sand vs. water vs. lava.

### Fix 20C: Nearby Block Grid (new, +75 dims)
**Files:** `sfpmc: GameStateExtractor.java`, `sfpmc: minecraft_tokenizer.py`
**Fix:** Added 5×3×5 axis-aligned block grid centered on player (75 blocks). Each block encoded as material category: 0.0=air, 0.25=passable, 0.5=solid, 0.75=liquid, 1.0=hazard. New `categorizeBlock()` Java helper identifies hazards (lava, fire, magma, cactus, etc.). 75 O(1) chunk lookups ≈ 7.5μs/tick.

### Fix 20D: Directional Distances (new, +6 dims)
**Files:** `sfpmc: GameStateExtractor.java`, `sfpmc: minecraft_tokenizer.py`
**Fix:** Added distance-to-first-solid-block in 6 cardinal directions from player feet, scanning up to 8 blocks, normalized by /8.0. Gives navigational awareness: wall proximity, cliff detection, ceiling height.

### Fix 20E: Better Position Normalization
**File:** `sfpmc: minecraft_tokenizer.py`
**Fix:** Changed coordinate normalization from /1000 (all axes) to /100 (x,z) and /150 (y). At /100, a 200-block journey = 2.0 in the state vector vs. 0.2 before. Height now uses /150 so sea level (y=64) = 0.43, diamond level (y=-59) = -0.39 — much better gradient for spatial learning.

**Dim budget:** +91 dims. State vector: ~204/256 used (~52 remaining).

---

## Fix 21: Goals Panel — Full Goal Display in GUI and Overlay

**Status:** Deployed
**Root cause:** Goals stored as pure embeddings with no human-readable text. GUI showed only "Goals:3" count, overlay showed only the first goal's ID + progress.

### Fix 21A: Goal Description Field
**Files:** `types.py`, `goals/persistence.py`, `interface.py`
**Fix:** Added `description: str = ""` field to the `Goal` dataclass. Threaded through `create_goal()` in GoalRegister, SFPInterface, and `list_goals()` return dicts. Backward-compatible — defaults to empty string for existing goals.

### Fix 21B: Named Goals at Creation
**Files:** `sfpmc: bridge_server.py`, `inference_loop.py`
**Fix:** Exploration goal created with `description="Explore"`. User-set goals (via numpad feedback) pass the goal text as `description=goal_text`.

### Fix 21C: Goals Panel in GUI
**Files:** `gui/goals_panel.py` (NEW), `gui/app.py`, `gui/worker.py`
**Fix:** New `GoalsPanel` widget with a Treeview showing all goals: ID, Name, Status, Progress, Priority, Urgency. Color-coded by status (green=completed, gray=paused, orange=blocked, red=failed/expired). Remove/Pause/Resume buttons. 2-second polling via worker commands (`list_goals`, `remove_goal`, `pause_goal`, `resume_goal`). Added `pause_goal()` and `resume_goal()` to SFPInterface.

### Fix 21D: Enhanced Overlay Goals
**Files:** `sfpmc: inference_loop.py`, `CommOverlayRenderer.java`
**Fix:** Overlay now shows top 3 goals by priority with names (e.g. "Explore (12%) | Find diamonds (0%)") instead of just "Goal #0 (12%)". Goal text truncation widened from 20 to 50 characters.

---

## Summary

| # | Fix | Category | File(s) | Impact |
|---|-----|----------|---------|--------|
| 1 | Bridge logging visibility | Diagnostics | `gui/app.py` | Revealed all subsequent errors |
| 2 | step_count + salience_stats | Data flow | `memory/processor.py` | Steps now tracked correctly |
| 3 | Queue exception logging | Diagnostics | `sfpmc: bridge_server.py` | Queue failures visible |
| 4 | Salience stats in status/checkpoint | Diagnostics | `interface.py`, `serialization.py` | Gate decisions exposed |
| 5 | Pipeline health counters | Diagnostics | `sfpmc: inference_loop.py` | Process success/fail counted |
| 6 | Bridge status enrichment | GUI | `sfpmc: bridge_server.py`, `gui/status_panel.py` | Pipeline health in GUI |
| 7 | None attn_weights crash | Bug fix | `metacognition/uncertainty.py` | 100% process crash fixed |
| 8 | Context dimension mismatch | Bug fix | `attention/salience.py` | Salience gate crash fixed |
| 9 | CUDA PyTorch | Performance | pip install (no code) | ~20x speedup (CPU→GPU) |
| 10 | Health polling | GUI | `gui/app.py` | Live status updates |
| 11 | LoRA device mismatch | Bug fix | `core/lora.py` | CUDA forward pass fixed |
| 12 | Episode device mismatch | Bug fix | `memory/consolidation.py`, `storage/serialization.py` | Episode storage on CUDA fixed |
| 13A | Log noise reduction | Quality of life | `sfpmc: inference_loop.py`, `defense/gradient_bounds.py` | Reduced log spam |
| 13B | World model device fix | Bug fix | `prediction/world_model.py` | Eliminated per-tick CPU→GPU transfer |
| 13C | Flash attention | Performance | `core/perceiver.py`, `core/backbone.py` | ~2–4x faster attention |
| 13D | Fused AdamW | Performance | `core/streaming.py` | ~20-30% faster optimizer step |
| 13E | Mixed precision (AMP) | Performance | `memory/processor.py` | ~2x faster Perceiver+Backbone |
| 14A | Zero-init action head | Bug fix | `sfpmc: minecraft_action_head.py` | Eliminates camera drift on fresh sessions |
| 14B | Last-action feedback | Feature | `sfpmc: inference_loop.py`, `minecraft_tokenizer.py` | AI can observe its own motor commands |
| 14C | Soft-clamp camera pitch | Bug fix | `sfpmc: inference_loop.py` | Prevents ground/sky lock |
| 14D | Demo-based action training | Feature | `sfpmc: minecraft_action_head.py`, `demo_pipeline.py`, `bridge_server.py` | Action head learns from human demos |
| 15 | Text encoder CUDA generator | Bug fix | `sfpmc: text_encoder.py` | Exploration goal creation crash fixed |
| 16A | Rename hash function | Refactor | `sfpmc: minecraft_tokenizer.py` | Generic registry ID hash for entities + items |
| 16B | Rich hotbar encoding | Feature | `sfpmc: minecraft_tokenizer.py` | AI can identify items by type + durability (9→54 dims) |
| 16C | Main hand item hash | Feature | `sfpmc: minecraft_tokenizer.py` | Direct signal for currently held item (+4 dims) |
| 17A | Display progress in overlay | Bug fix | `sfpmc: inference_loop.py` | Shows actual completion % instead of priority |
| 17B | Goal satisfaction target | Bug fix | `goals/persistence.py` | Completion detection now semantically meaningful |
| 17C | Auto-advance exploration goal | Feature | `sfpmc: bridge_server.py`, `inference_loop.py` | Continuous goal cycling on completion |
| 17D | Lower satisfaction threshold | Config | `config.py` | Goal completion achievable (0.95 → 0.8) |
| 18 | Movement vector override | Bug fix | `sfpmc: KeyboardInputMixin.java` | AI horizontal movement restored |
| 19 | Camera exploration noise floor | Bug fix | `sfpmc: inference_loop.py` | Camera moves at startup (baseline exploration) |
| 20A | Enhanced crosshair target | Feature | `sfpmc: minecraft_tokenizer.py` | Block/entity type + distance in crosshair (+5 dims) |
| 20B | Ground block | Feature | `sfpmc: GameStateExtractor.java`, `minecraft_tokenizer.py` | What the player stands on (+5 dims) |
| 20C | Nearby block grid | Feature | `sfpmc: GameStateExtractor.java`, `minecraft_tokenizer.py` | 5×3×5 material category grid (+75 dims) |
| 20D | Directional distances | Feature | `sfpmc: GameStateExtractor.java`, `minecraft_tokenizer.py` | Distance to solid in 6 directions (+6 dims) |
| 20E | Better position normalization | Feature | `sfpmc: minecraft_tokenizer.py` | /1000→/100,/150 for distinguishable coordinates |
| 21A | Goal description field | Feature | `types.py`, `goals/persistence.py`, `interface.py` | Human-readable names on goals |
| 21B | Named goals at creation | Feature | `sfpmc: bridge_server.py`, `inference_loop.py` | "Explore" and user text passed as description |
| 21C | Goals panel in GUI | Feature | `gui/goals_panel.py`, `gui/app.py`, `gui/worker.py` | Treeview with status, progress, priority, controls |
| 21D | Enhanced overlay goals | Feature | `sfpmc: inference_loop.py`, `CommOverlayRenderer.java` | Top 3 goals with names, wider truncation |
