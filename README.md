# SFP — Semantic Field Processor

A cognitive architecture built from scratch in PyTorch. SFP encodes knowledge as the shape of neural network weights — data transforms the manifold and moves on, never stored as raw records.

**18,000+ lines of Python** across 84 modules implementing hierarchical memory, surprise-gated learning, multi-hop reasoning, and six cognitive subsystems that co-adapt during learning.

## Why This Exists

Most AI systems are stateless — they process inputs, produce outputs, and forget. SFP explores a different approach: a persistent neural manifold that *learns continuously* from a stream of inputs, consolidates important patterns into long-term memory, and reasons by traversing associative chains across what it has learned.

This is not a wrapper around an LLM or a fine-tuning script. It is a ground-up architecture for continual learning, built to investigate how memory consolidation, surprise-driven attention, and goal-directed behavior can work together in a single system.

## Architecture

```
Input → Perceiver IO Encoder → MLP Manifold (Tier 0)
                                    ↓
                            Surprise Gate — should we learn from this?
                            (adaptive threshold, soft sigmoid, importance weighting)
                                    ↓
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
              Tier 1 STM      Tier 2 Essential  Tier 3 Core
              (working)       (basin attractors) (long-term)
                    ↓               ↓               ↓
                    └───────── Consolidation ───────┘
                              (scheduled + surprise-triggered)
                                    ↓
                           Reasoning Engine
                    (transition graph, multi-hop chains,
                     scene graph, associative retrieval)
```

### Cognitive Modules

Six optional subsystems co-adapt with the core field through auxiliary loss coupling:

| Module | What It Does | Key Mechanism |
|--------|-------------|---------------|
| **World Model** | Predicts next observation from current state | RSSM (Recurrent State-Space Model) with deterministic + stochastic paths |
| **Goal Persistence** | Maintains active objectives across time | 32-slot register with satisfaction tracking and decay |
| **Metacognition** | Estimates uncertainty of its own outputs | 4-source fusion: ensemble disagreement, gradient magnitude, memory distance, prediction error |
| **Valence** | Tracks emotional/evaluative tone of inputs | Running mood with momentum, surprise-weighted updates |
| **Selective Attention** | Gates what reaches the field | Salience scoring from surprise, goal relevance, novelty |
| **Generative Replay** | Prevents catastrophic forgetting | Generates synthetic rehearsal samples from consolidated memories |

### Memory System

Four-tier hierarchy inspired by complementary learning systems theory:

- **Tier 0 (Working Memory):** The MLP manifold itself — volatile, continuously updated
- **Tier 1 (Short-Term):** Recent patterns with decay, capacity-bounded
- **Tier 2 (Essential):** Basin attractors formed from repeated/important patterns
- **Tier 3 (Core):** Long-term stable knowledge, slow to update, high threshold

Consolidation runs on a configurable schedule and can be triggered by surprise spikes. Patterns promote upward through tiers based on access frequency, surprise at encoding, and goal relevance.

### Continual Learning

Two adapter mechanisms prevent catastrophic forgetting:

- **LoRA (Low-Rank Adaptation):** Adds low-rank update matrices to field layers. Multiple LoRA adapters can be merged based on surprise, goal, and metacognition signals.
- **EWC (Elastic Weight Consolidation):** Penalizes changes to weights that are important for previously learned patterns, estimated via Fisher information.

### Reasoning

- **Transition Graph:** Weighted directed graph of observed state transitions. Edges accumulate evidence from repeated observations.
- **Multi-Hop Chains:** Traverses the transition graph to answer queries that require connecting multiple intermediate states.
- **Scene Graph:** Tracks entity-relation-entity triples for structured scene understanding.

### Defense & Integrity

- Input validation with anomaly scoring
- Gradient magnitude bounds to prevent injection attacks
- Surprise hardening (rejects inputs designed to trigger excessive updates)
- Topology monitoring via persistent homology (requires giotto-tda)
- Anchor verification for detecting manifold drift

## Quick Start

### From Source

```bash
git clone https://github.com/Wraek/sfp.git
cd sfp
pip install -e ".[all]"
python -m sfp
```

### Standalone (Windows)

Download `sfp_gui.zip` from Releases, extract, and run `sfp_gui.exe`. No Python needed.

### Programmatic Use

```python
import sfp

# Streaming processor — learns from each input
processor = sfp.create_field("small", streaming=True)
metric = processor.process(some_tensor)

# Full cognitive system
processor = sfp.create_field("large", hierarchical=True,
    world_model=True, goals=True, metacognition=True)
metric = processor.process(some_tensor)
result = processor.query(some_tensor)
```

## Configuration

Three presets for different use cases:

| Preset | Dimensions | Layers | Cognitive Modules | Use Case |
|--------|-----------|--------|-------------------|----------|
| **Minimal** | 512 | 6 | None | Fast experiments, embedding |
| **Standard** | 1024 | 8 | World Model, Goals | General-purpose learning |
| **Full** | 2048 | 8 | All 6 modules | Research, complex environments |

All 20+ config dataclasses are accessible through the GUI's Advanced Settings dialog, organized by subsystem.

## Project Structure

```
src/sfp/
├── core/           MLP manifold, streaming processor, LoRA, EWC, backbone transformer
├── memory/         4-tier hierarchical memory, consolidation, episodic buffer, replay
├── reasoning/      Transition graph, multi-hop chains, scene graph
├── prediction/     RSSM world model
├── goals/          32-slot goal persistence register
├── metacognition/  4-source uncertainty estimation
├── affect/         Valence and mood tracking
├── attention/      Selective attention / salience gate
├── defense/        Input validation, anomaly detection, gradient bounds, topology monitoring
├── comms/          Multi-agent communication protocol (compression, negotiation, sync)
├── input/          Input encoding adapters (byte-level, embeddings)
├── storage/        Serialization, quantization, mixed precision
├── topology/       Persistent homology analysis and visualization
├── gui/            Desktop GUI (tkinter) with session management
└── utils/          Logging, device resolution, math utilities
```

## Dependencies

| Package | Purpose |
|---------|---------|
| PyTorch >= 2.0 | Neural network framework |
| NumPy >= 1.24 | Numerical computing |
| giotto-tda *(optional)* | Persistent homology for topology analysis |
| sentence-transformers *(optional)* | Text embedding adapters |
| matplotlib *(optional)* | Visualization |

## Related

- **[sfpmc](https://github.com/Wraek/sfpmc)** — Connects SFP to Minecraft 1.21.4 as a reinforcement learning environment. Fabric mod captures frames at 20fps, Python bridge tokenizes visual/state/entity data into SFP's input format. Supports human demonstration → AI play learning loops.

## License

MIT
