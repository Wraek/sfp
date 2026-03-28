"""SFP — Semantic Field Processing.

Knowledge encoded as the shape of neural network weights. Data transforms
the manifold and moves on — never retained.

Basic usage::

    import sfp

    # Create a streaming field processor
    processor = sfp.create_field("small")
    metric = processor.process(some_tensor)

    # Query the manifold for nearest attractor
    result = processor.query(some_tensor)
"""

from sfp._version import __version__
from sfp.config import (
    AttractorConfig,
    CommConfig,
    EWCConfig,
    FieldConfig,
    GenerativeReplayConfig,
    GoalPersistenceConfig,
    LoRAConfig,
    MetacognitionConfig,
    QuantizationConfig,
    SelectiveAttentionConfig,
    StreamingConfig,
    ValenceConfig,
    WorldModelConfig,
)
from sfp.core.attractors import AttractorQuery
from sfp.core.field import SemanticFieldProcessor
from sfp.core.forgetting import EWCStrategy, WeightDecayStrategy
from sfp.core.lora import LoRALinear, OnlineLoRAManager
from sfp.core.streaming import StreamingProcessor
from sfp.exceptions import (
    AdapterNotFoundError,
    ConsolidationError,
    ConvergenceError,
    GenerativeReplayError,
    GoalError,
    IntegrityError,
    ManifoldDriftError,
    MemoryTierError,
    MetacognitionError,
    PoisoningDetectedError,
    QuantizationError,
    ReasoningChainError,
    SFPError,
    SalienceError,
    WorldModelError,
)
from sfp.bridge import BridgeLoadError, BridgeLoader, BridgeProtocol
from sfp.interface import SFPInterface
from sfp.memory.processor import HierarchicalMemoryProcessor
from sfp.types import (
    AttractorResult,
    CommLayer,
    CompressedDeltas,
    FieldSize,
    ForgetStrategy,
    GoalStatus,
    HealthReport,
    InputEncoder,
    ProcessingLevel,
    ReplayStrategy,
    SurpriseMetric,
    SyntheticEpisode,
    TopologicalEvent,
    TopologySnapshot,
    UncertaintyEstimate,
    ValenceSignal,
    WorldModelState,
)
from sfp.utils.device import resolve_device


def create_field(
    size: str | FieldSize = FieldSize.SMALL,
    *,
    streaming: bool = True,
    hierarchical: bool = False,
    lora: bool = True,
    ewc: bool = True,
    world_model: bool = False,
    goals: bool = False,
    metacognition: bool = False,
    valence: bool = False,
    selective_attention: bool = False,
    generative_replay: bool = False,
    device: str = "auto",
) -> HierarchicalMemoryProcessor | StreamingProcessor | SemanticFieldProcessor:
    """Create a semantic field processor with sensible defaults.

    Args:
        size: Field size — "tiny", "small", "medium", "large", or FieldSize enum.
        streaming: If True, wraps field in a StreamingProcessor with
            surprise-gated updates.
        hierarchical: If True, creates a full HierarchicalMemoryProcessor with
            Perceiver IO, backbone transformer, four memory tiers, reasoning
            chains, and defense framework.
        lora: Enable Online-LoRA adapters (only when streaming=True).
        ewc: Enable Elastic Weight Consolidation (only when streaming=True).
        world_model: Enable RSSM predictive world model (hierarchical only).
        goals: Enable 32-slot goal persistence register (hierarchical only).
        metacognition: Enable 4-source uncertainty estimation (hierarchical only).
        valence: Enable affective valence and mood tracking (hierarchical only).
        selective_attention: Enable pre-Perceiver salience gate (hierarchical only).
        generative_replay: Enable synthetic episode generation (hierarchical only).
        device: Target device ("auto", "cpu", "cuda", "mps").

    Returns:
        HierarchicalMemoryProcessor if hierarchical=True,
        StreamingProcessor if streaming=True,
        else bare SemanticFieldProcessor.

    Example::

        # Simple streaming processor
        processor = sfp.create_field("small", streaming=True)
        metric = processor.process(torch.randn(512))

        # Full hierarchical memory system
        processor = sfp.create_field("small", hierarchical=True)
        metric = processor.process(torch.randn(512))
        result = processor.query(torch.randn(512))

        # Hierarchical with cognitive modules
        processor = sfp.create_field(
            "small", hierarchical=True,
            world_model=True, goals=True,
        )
    """
    # Resolve size string to enum
    if isinstance(size, str):
        size = FieldSize[size.upper()]

    if hierarchical:
        from sfp.config import (
            BackboneConfig,
            PerceiverConfig,
        )

        field_config = FieldConfig.from_preset(size)
        d = field_config.dim
        return HierarchicalMemoryProcessor(
            field_config=field_config,
            perceiver_config=PerceiverConfig(d_input=d, d_latent=d),
            backbone_config=BackboneConfig(d_model=d),
            lora_config=LoRAConfig(enabled=lora) if lora else None,
            ewc_config=EWCConfig(enabled=ewc) if ewc else None,
            world_model_config=WorldModelConfig(d_observation=d) if world_model else None,
            goal_config=GoalPersistenceConfig(d_goal=d, d_satisfaction=d) if goals else None,
            metacognition_config=MetacognitionConfig() if metacognition else None,
            valence_config=ValenceConfig() if valence else None,
            attention_config=SelectiveAttentionConfig() if selective_attention else None,
            replay_config=GenerativeReplayConfig() if generative_replay else None,
            device=device,
        )

    dev = resolve_device(device)
    config = FieldConfig.from_preset(size)
    field = SemanticFieldProcessor(config).to(dev)

    if not streaming:
        return field

    streaming_config = StreamingConfig()
    lora_config = LoRAConfig(enabled=lora)
    ewc_config = EWCConfig(enabled=ewc)

    return StreamingProcessor(
        field=field,
        streaming_config=streaming_config,
        lora_config=lora_config,
        ewc_config=ewc_config,
    )


__all__ = [
    "__version__",
    # Factory
    "create_field",
    # Interface
    "SFPInterface",
    # Bridge
    "BridgeProtocol",
    "BridgeLoader",
    "BridgeLoadError",
    # Core
    "SemanticFieldProcessor",
    "StreamingProcessor",
    "HierarchicalMemoryProcessor",
    "AttractorQuery",
    "LoRALinear",
    "OnlineLoRAManager",
    "EWCStrategy",
    "WeightDecayStrategy",
    # Config
    "FieldConfig",
    "StreamingConfig",
    "LoRAConfig",
    "EWCConfig",
    "AttractorConfig",
    "QuantizationConfig",
    "CommConfig",
    "WorldModelConfig",
    "GoalPersistenceConfig",
    "MetacognitionConfig",
    "ValenceConfig",
    "SelectiveAttentionConfig",
    "GenerativeReplayConfig",
    # Types
    "FieldSize",
    "CommLayer",
    "AttractorResult",
    "SurpriseMetric",
    "TopologySnapshot",
    "TopologicalEvent",
    "HealthReport",
    "CompressedDeltas",
    "InputEncoder",
    "ForgetStrategy",
    "GoalStatus",
    "ProcessingLevel",
    "ReplayStrategy",
    "UncertaintyEstimate",
    "ValenceSignal",
    "WorldModelState",
    "SyntheticEpisode",
    # Exceptions
    "SFPError",
    "ConvergenceError",
    "ManifoldDriftError",
    "QuantizationError",
    "AdapterNotFoundError",
    "MemoryTierError",
    "ConsolidationError",
    "IntegrityError",
    "PoisoningDetectedError",
    "ReasoningChainError",
    "WorldModelError",
    "GoalError",
    "MetacognitionError",
    "SalienceError",
    "GenerativeReplayError",
    # Utils
    "resolve_device",
]
