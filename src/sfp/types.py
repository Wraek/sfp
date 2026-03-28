"""Shared types, enums, protocols, and dataclasses for the SFP framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


class FieldSize(Enum):
    """Predefined MLP manifold sizes."""

    TINY = auto()  # 4x256,  ~262K params
    SMALL = auto()  # 6x512,  ~1.6M params
    MEDIUM = auto()  # 8x1024, ~8.4M params
    LARGE = auto()  # 8x2048, ~33.6M params


class CommLayer(Enum):
    """Communication protocol layers, ordered by compression aggressiveness."""

    L0_RAW_TEXT = 0
    L1_EMBEDDING = 1
    L2_MANIFOLD_COORD = 2
    L3_DEFORMATION = 3
    L4_SURPRISE_GATED = 4


@dataclass(frozen=True)
class AttractorResult:
    """Result of an attractor query — a converged fixed point."""

    point: torch.Tensor
    iterations: int
    converged: bool
    basin_id: int | None = None
    trajectory: list[torch.Tensor] | None = None


@dataclass(frozen=True)
class SurpriseMetric:
    """Measurement from a single streaming update step."""

    grad_norm: float
    loss: float
    updated: bool
    timestamp: float = 0.0
    auxiliary_loss: float = 0.0
    loss_components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TopologySnapshot:
    """A snapshot of the manifold's topological features at a point in time."""

    timestamp: float
    betti_numbers: tuple[int, ...]
    persistence_diagram: np.ndarray
    total_persistence: float


@dataclass(frozen=True)
class TopologicalEvent:
    """A detected change in manifold topology between snapshots."""

    event_type: str  # "birth" | "death"
    dimension: int
    significance: float


@dataclass(frozen=True)
class HealthReport:
    """Composite health metrics for a concept manifold."""

    attractor_count: int
    mean_basin_radius: float
    topological_complexity: dict[str, int]
    information_density: float
    spectral_gap: float
    timestamp: float = 0.0


@dataclass
class CompressedDeltas:
    """Compressed weight deltas for L3 communication."""

    indices: dict[str, torch.Tensor]
    values: dict[str, torch.Tensor]
    shapes: dict[str, tuple[int, ...]]
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InputEncoder(Protocol):
    """Protocol for input encoders that convert arbitrary inputs to vectors."""

    @property
    def output_dim(self) -> int: ...

    def encode(self, inputs: list[Any]) -> torch.Tensor: ...


@runtime_checkable
class ForgetStrategy(Protocol):
    """Protocol for forgetting mitigation strategies."""

    def penalty(self, model: torch.nn.Module) -> torch.Tensor: ...

    def update_importance(self, model: torch.nn.Module) -> None: ...


# ---------------------------------------------------------------------------
# Hierarchical memory types
# ---------------------------------------------------------------------------


class MemoryTier(Enum):
    """The four tiers of the hierarchical memory system."""

    WORKING = 0  # Tier 0: volatile MLP, updates every step
    EPISODIC = 1  # Tier 1: structured episode buffer
    ESSENTIAL = 2  # Tier 2: key-value with Hopfield retrieval
    CORE = 3  # Tier 3: near-frozen axiomatic memory


class ConsolidationMode(Enum):
    """Which consolidation cycle to run."""

    MINI = auto()  # Tier 0 -> Tier 1 (~100 steps)
    STANDARD = auto()  # Tier 1 -> Tier 2 (~1000 steps)
    DEEP = auto()  # Tier 2 -> Tier 3 (~10000 steps)


class RelationType(Enum):
    """Typed relations in the transition structure between concept basins."""

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
    """Defense framework security posture."""

    HIGH = auto()
    NORMAL = auto()
    ACCELERATED = auto()


@dataclass
class Episode:
    """A single episodic memory entry (Tier 1).

    Mutable because consolidation_count and flagged are updated in-place
    during consolidation cycles.
    """

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


@dataclass(frozen=True)
class ChainTrace:
    """A single hop in a reasoning chain."""

    hop: int
    basin_id: int | None
    event_type: str  # "start", "hop", "branch", "terminate"
    confidence: float = 0.0
    score: float = 0.0
    edge_relation: str = ""
    n_branches: int = 0
    knowledge_norm: float = 0.0
    query_residual_norm: float = 0.0


@dataclass(frozen=True)
class ReasoningResult:
    """Result of a reasoning chain execution."""

    knowledge: torch.Tensor
    n_hops: int
    visited_basins: list[int]
    terminated_reason: str  # "resolved", "dead_end", "cycle", "convergence", "max_hops"
    routing: str  # "single_hop" or "multi_hop"
    chain_weight: float = 1.0
    trace: list[ChainTrace] = field(default_factory=list)


@dataclass(frozen=True)
class PromotionRequest:
    """A request to promote a Tier 2 basin to Tier 3 core memory."""

    basin_id: int
    confidence: float
    episode_count: int
    modality_count: int
    age_days: float
    key_snapshot: torch.Tensor
    value_snapshot: torch.Tensor


@runtime_checkable
class ConsistencyChecker(Protocol):
    """Protocol for Tier 2 consistency checking against Tier 0 updates."""

    def check_consistency(
        self, input_embedding: torch.Tensor, proposed_update: torch.Tensor
    ) -> torch.Tensor: ...


@runtime_checkable
class AuthorizationHandler(Protocol):
    """Protocol for handling Tier 3 promotion authorization events."""

    def authorize(self, request: PromotionRequest) -> bool: ...


# ---------------------------------------------------------------------------
# Cognitive module types (world model, goals, metacognition, valence,
# selective attention, generative replay)
# ---------------------------------------------------------------------------


class GoalStatus(Enum):
    """Status of a goal in the goal register."""

    ACTIVE = auto()
    PAUSED = auto()
    BLOCKED = auto()
    COMPLETED = auto()
    FAILED = auto()
    EXPIRED = auto()


class ProcessingLevel(Enum):
    """Processing level decision from the salience gate."""

    SKIP = auto()  # <0.05ms, discard
    SKIM = auto()  # ~0.3ms, micro-update only
    FULL = auto()  # ~15ms, complete pipeline


class ReplayStrategy(Enum):
    """Generative replay generation strategy."""

    BASIN_INTERPOLATION = auto()
    CHAIN_DREAMING = auto()
    BOUNDARY_EXPLORATION = auto()


@dataclass
class Goal:
    """A single entry in the goal register.

    Mutable because priority, progress, and status are updated in-place
    during processing cycles.
    """

    id: int
    embedding: torch.Tensor  # (d_goal,)
    satisfaction_embedding: torch.Tensor  # (d_goal,)
    description: str = ""
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


@dataclass(frozen=True)
class SalienceResult:
    """Result of the salience gate evaluation."""

    level: ProcessingLevel
    salience_scores: dict[str, float]
    combined_salience: float
    interrupt: bool = False
    interrupt_reason: str = ""


@dataclass(frozen=True)
class UncertaintyEstimate:
    """Structured uncertainty estimate from the metacognition module."""

    retrieval_uncertainty: float
    chain_uncertainty: float
    prediction_uncertainty: float
    knowledge_uncertainty: float
    composite_embedding: torch.Tensor  # (d_uncertainty_embedding,)
    scalar_confidence: float  # [0, 1]
    calibrated: bool = False


@dataclass(frozen=True)
class ValenceSignal:
    """Valence output from the affect system."""

    scalar_valence: float  # [-1, 1]
    valence_embedding: torch.Tensor  # (d_valence_embedding,)
    projected_context: torch.Tensor  # (d_model,)
    mood_immediate: float
    mood_short_term: float
    mood_baseline: float
    composite_mood: float
    risk_tolerance: float
    vigilance: float


@dataclass(frozen=True)
class SyntheticEpisode:
    """A synthetically generated episode from generative replay."""

    embedding: torch.Tensor
    source_basins: list[int]
    strategy: ReplayStrategy
    validation_passed: bool
    weight: float  # 0.2-0.5
    is_synthetic: bool = True


@dataclass(frozen=True)
class WorldModelState:
    """Latent state of the RSSM world model."""

    deterministic: torch.Tensor  # (d_deterministic,)
    stochastic: torch.Tensor  # (d_stochastic_categories * d_stochastic_classes,)
    prediction_error: float = 0.0
    kl_divergence: float = 0.0
    reconstruction_error: float = 0.0
