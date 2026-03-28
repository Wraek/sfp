"""Manifold checkpoint save/load for persistence and transfer."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from sfp._version import __version__
from sfp.config import EWCConfig, FieldConfig, LoRAConfig, StreamingConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.utils.logging import get_logger

logger = get_logger("storage.serialization")


class ManifoldCheckpoint:
    """Save and load complete manifold state including streaming processor."""

    @staticmethod
    def save(
        path: str | Path,
        field: SemanticFieldProcessor,
        streaming_processor: object | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a manifold checkpoint.

        Args:
            path: File path for the checkpoint.
            field: The SemanticFieldProcessor to save.
            streaming_processor: Optional StreamingProcessor for full state.
            metadata: Optional user metadata dict.
        """
        checkpoint: dict[str, Any] = {
            "version": __version__,
            "field_config": asdict(field.config),
            "field_state_dict": field.state_dict(),
        }

        if streaming_processor is not None:
            sp = streaming_processor
            checkpoint["streaming_config"] = asdict(sp.config)
            checkpoint["optimizer_state"] = sp._optimizer.state_dict()
            checkpoint["surprise_history"] = [
                {
                    "grad_norm": m.grad_norm,
                    "loss": m.loss,
                    "updated": m.updated,
                    "timestamp": m.timestamp,
                }
                for m in sp._history
            ]
            checkpoint["step_count"] = sp._step_count
            checkpoint["surprise_ema"] = sp._surprise_ema

            # EWC state
            if sp._ewc_strategy is not None:
                checkpoint["ewc_config"] = asdict(sp._ewc_strategy.config)
                checkpoint["ewc_fisher"] = {
                    k: v.clone() for k, v in sp._ewc_strategy._fisher.items()
                }
                checkpoint["ewc_anchors"] = {
                    k: v.clone() for k, v in sp._ewc_strategy._anchors.items()
                }

            # LoRA state
            if sp._lora_manager is not None:
                checkpoint["lora_config"] = asdict(sp._lora_manager.config)
                checkpoint["lora_states"] = [
                    {"A": layer.A.data.clone(), "B": layer.B.data.clone()}
                    for layer in sp._lora_manager.lora_layers
                ]

        if metadata is not None:
            checkpoint["metadata"] = metadata

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Saved checkpoint to %s", path)

    @staticmethod
    def load(
        path: str | Path,
        device: str | torch.device = "cpu",
    ) -> tuple[SemanticFieldProcessor, Any, dict[str, Any]]:
        """Load a manifold checkpoint.

        Args:
            path: Path to the checkpoint file.
            device: Target device.

        Returns:
            (field, streaming_processor_or_none, metadata) tuple.
        """
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        logger.info(
            "Loading checkpoint (version=%s) from %s",
            checkpoint.get("version", "unknown"),
            path,
        )

        # Reconstruct field
        field_config = FieldConfig(**checkpoint["field_config"])
        field = SemanticFieldProcessor(field_config)

        streaming_processor = None
        if "streaming_config" in checkpoint:
            from sfp.core.streaming import StreamingProcessor
            from sfp.types import SurpriseMetric

            streaming_config = StreamingConfig(**checkpoint["streaming_config"])

            lora_config = None
            if "lora_config" in checkpoint:
                lora_config = LoRAConfig(**checkpoint["lora_config"])

            ewc_config = None
            if "ewc_config" in checkpoint:
                ewc_config = EWCConfig(**checkpoint["ewc_config"])

            # Build StreamingProcessor before loading state_dict so that
            # LoRA wrapping (if any) modifies the field's module tree to
            # match the saved state_dict keys.
            streaming_processor = StreamingProcessor(
                field=field,
                streaming_config=streaming_config,
                lora_config=lora_config,
                ewc_config=ewc_config,
            )

        # Load state_dict after LoRA wrapping (if any) has been applied
        field.load_state_dict(checkpoint["field_state_dict"])
        field.to(device)

        # Restore streaming processor state
        if streaming_processor is not None:
            # Restore optimizer state
            if "optimizer_state" in checkpoint:
                try:
                    streaming_processor._optimizer.load_state_dict(
                        checkpoint["optimizer_state"]
                    )
                except (ValueError, KeyError):
                    logger.warning(
                        "Could not restore optimizer state; using fresh optimizer"
                    )

            # Restore EWC state
            if (
                streaming_processor._ewc_strategy is not None
                and "ewc_fisher" in checkpoint
            ):
                for k, v in checkpoint["ewc_fisher"].items():
                    if k in streaming_processor._ewc_strategy._fisher:
                        streaming_processor._ewc_strategy._fisher[k] = v.to(
                            device
                        )
                for k, v in checkpoint["ewc_anchors"].items():
                    if k in streaming_processor._ewc_strategy._anchors:
                        streaming_processor._ewc_strategy._anchors[k] = v.to(
                            device
                        )

            # Restore LoRA state
            if (
                streaming_processor._lora_manager is not None
                and "lora_states" in checkpoint
            ):
                for layer, state in zip(
                    streaming_processor._lora_manager.lora_layers,
                    checkpoint["lora_states"],
                ):
                    layer.A.data = state["A"].to(device)
                    layer.B.data = state["B"].to(device)

            # Restore surprise history
            from sfp.types import SurpriseMetric

            if "surprise_history" in checkpoint:
                streaming_processor._history = [
                    SurpriseMetric(**m) for m in checkpoint["surprise_history"]
                ]

            streaming_processor._step_count = checkpoint.get("step_count", 0)
            streaming_processor._surprise_ema = checkpoint.get(
                "surprise_ema", 0.0
            )

        metadata = checkpoint.get("metadata", {})
        return field, streaming_processor, metadata


class SessionCheckpoint:
    """Save and load complete HierarchicalMemoryProcessor session state.

    Stores the create_field() kwargs so the system can be reconstructed
    identically, then overlays saved state_dicts and non-Module data.
    """

    @staticmethod
    def save(
        path: str | Path,
        processor: object,
        create_kwargs: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save a full session checkpoint.

        Args:
            path: File path for the checkpoint.
            processor: A HierarchicalMemoryProcessor or StreamingProcessor.
            create_kwargs: The kwargs dict passed to sfp.create_field().
            metadata: Optional session metadata (name, preset, etc.).
        """
        from sfp.core.streaming import StreamingProcessor
        from sfp.memory.processor import HierarchicalMemoryProcessor

        checkpoint: dict[str, Any] = {
            "version": __version__,
            "checkpoint_type": "session",
            "create_kwargs": create_kwargs,
            "timestamp": time.time(),
        }

        if isinstance(processor, HierarchicalMemoryProcessor):
            checkpoint["processor_type"] = "hierarchical"
            checkpoint["step_count"] = processor._step_count
            checkpoint["salience_stats"] = processor._salience_stats

            # Neural module state dicts
            checkpoint["perceiver_state"] = processor._perceiver.state_dict()
            checkpoint["backbone_state"] = processor._backbone.state_dict()
            checkpoint["tier2_state"] = processor._tier2.state_dict()
            checkpoint["tier3_state"] = processor._tier3.state_dict()
            checkpoint["transitions_state"] = processor._transitions.state_dict()

            # Tier 0 (StreamingProcessor) state via ManifoldCheckpoint logic
            t0 = processor._tier0
            checkpoint["field_state_dict"] = t0.field.state_dict()
            checkpoint["optimizer_state"] = t0._optimizer.state_dict()
            checkpoint["surprise_history"] = [
                {
                    "grad_norm": m.grad_norm,
                    "loss": m.loss,
                    "updated": m.updated,
                    "timestamp": m.timestamp,
                }
                for m in t0._history
            ]
            checkpoint["t0_step_count"] = t0._step_count
            checkpoint["t0_surprise_ema"] = t0._surprise_ema

            if t0._ewc_strategy is not None:
                checkpoint["ewc_fisher"] = {
                    k: v.clone() for k, v in t0._ewc_strategy._fisher.items()
                }
                checkpoint["ewc_anchors"] = {
                    k: v.clone() for k, v in t0._ewc_strategy._anchors.items()
                }

            if t0._lora_manager is not None:
                checkpoint["lora_states"] = [
                    {"A": layer.A.data.clone(), "B": layer.B.data.clone()}
                    for layer in t0._lora_manager.lora_layers
                ]

            # Tier 1 episodic memory (list of Episode dataclasses)
            episodes = []
            for ep in processor._tier1._hot + processor._tier1._cold:
                episodes.append({
                    "id": ep.id,
                    "timestamp": ep.timestamp,
                    "modality": ep.modality,
                    "provenance_hash": ep.provenance_hash,
                    "input_embedding": ep.input_embedding.detach().cpu(),
                    "working_memory_state": ep.working_memory_state.detach().cpu(),
                    "logit_snapshot": ep.logit_snapshot.detach().cpu(),
                    "surprise_at_storage": ep.surprise_at_storage,
                    "attractor_basin_id": ep.attractor_basin_id,
                    "attractor_distance": ep.attractor_distance,
                    "preceding_episode_id": ep.preceding_episode_id,
                    "following_episode_id": ep.following_episode_id,
                    "integrity_hash": ep.integrity_hash,
                    "weight_hash_at_storage": ep.weight_hash_at_storage,
                    "consolidation_count": ep.consolidation_count,
                    "last_consolidated": ep.last_consolidated,
                    "flagged": ep.flagged,
                    "valence": ep.valence,
                })
            checkpoint["tier1_episodes"] = episodes
            checkpoint["tier1_next_id"] = processor._tier1._next_id

            # Cognitive module state dicts
            if processor._world_model is not None:
                checkpoint["world_model_state"] = processor._world_model.state_dict()
            if processor._goals is not None:
                checkpoint["goals_state"] = processor._goals.state_dict()
            if processor._metacognition is not None:
                checkpoint["metacognition_state"] = processor._metacognition.state_dict()
            if processor._valence is not None:
                checkpoint["valence_state"] = processor._valence.state_dict()
            if processor._salience_gate is not None:
                checkpoint["salience_state"] = processor._salience_gate.state_dict()

        elif isinstance(processor, StreamingProcessor):
            checkpoint["processor_type"] = "streaming"
            # Delegate to existing ManifoldCheckpoint logic
            checkpoint["field_state_dict"] = processor.field.state_dict()
            checkpoint["field_config"] = asdict(processor.field.config)
            checkpoint["streaming_config"] = asdict(processor.config)
            checkpoint["optimizer_state"] = processor._optimizer.state_dict()
            checkpoint["surprise_history"] = [
                {
                    "grad_norm": m.grad_norm,
                    "loss": m.loss,
                    "updated": m.updated,
                    "timestamp": m.timestamp,
                }
                for m in processor._history
            ]
            checkpoint["step_count"] = processor._step_count
            checkpoint["surprise_ema"] = processor._surprise_ema
        else:
            msg = f"Unsupported processor type: {type(processor)}"
            raise TypeError(msg)

        if metadata is not None:
            checkpoint["metadata"] = metadata

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, path)
        logger.info("Saved session checkpoint to %s", path)

    @staticmethod
    def load(
        path: str | Path,
        device: str = "auto",
    ) -> tuple[object, dict[str, Any], dict[str, Any]]:
        """Load a session checkpoint.

        Args:
            path: Path to the checkpoint file.
            device: Target device.

        Returns:
            (processor, create_kwargs, metadata) tuple.
        """
        import sfp

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        logger.info(
            "Loading session checkpoint (version=%s) from %s",
            checkpoint.get("version", "unknown"),
            path,
        )

        create_kwargs = checkpoint["create_kwargs"]
        metadata = checkpoint.get("metadata", {})

        if device != "auto":
            create_kwargs["device"] = device

        processor = sfp.create_field(**create_kwargs)

        if checkpoint.get("processor_type") == "hierarchical":
            from sfp.types import SurpriseMetric

            processor._step_count = checkpoint.get("step_count", 0)
            processor._salience_stats = checkpoint.get(
                "salience_stats", {"skip": 0, "skim": 0, "full": 0},
            )

            # Neural module state dicts
            processor._perceiver.load_state_dict(checkpoint["perceiver_state"])
            processor._backbone.load_state_dict(checkpoint["backbone_state"])
            processor._tier2.load_state_dict(checkpoint["tier2_state"])
            processor._tier3.load_state_dict(checkpoint["tier3_state"])
            processor._transitions.load_state_dict(checkpoint["transitions_state"])

            # Tier 0 state
            processor._tier0.field.load_state_dict(checkpoint["field_state_dict"])

            if "optimizer_state" in checkpoint:
                try:
                    processor._tier0._optimizer.load_state_dict(
                        checkpoint["optimizer_state"]
                    )
                except (ValueError, KeyError):
                    logger.warning("Could not restore optimizer state")

            if processor._tier0._ewc_strategy is not None and "ewc_fisher" in checkpoint:
                dev = processor._device
                for k, v in checkpoint["ewc_fisher"].items():
                    if k in processor._tier0._ewc_strategy._fisher:
                        processor._tier0._ewc_strategy._fisher[k] = v.to(dev)
                for k, v in checkpoint["ewc_anchors"].items():
                    if k in processor._tier0._ewc_strategy._anchors:
                        processor._tier0._ewc_strategy._anchors[k] = v.to(dev)

            if processor._tier0._lora_manager is not None and "lora_states" in checkpoint:
                dev = processor._device
                for layer, state in zip(
                    processor._tier0._lora_manager.lora_layers,
                    checkpoint["lora_states"],
                ):
                    layer.A.data = state["A"].to(dev)
                    layer.B.data = state["B"].to(dev)

            if "surprise_history" in checkpoint:
                processor._tier0._history = [
                    SurpriseMetric(**m) for m in checkpoint["surprise_history"]
                ]
            processor._tier0._step_count = checkpoint.get("t0_step_count", 0)
            processor._tier0._surprise_ema = checkpoint.get("t0_surprise_ema", 0.0)

            # Tier 1 episodes
            if "tier1_episodes" in checkpoint:
                from sfp.types import Episode

                hot_cap = processor._tier1._config.hot_capacity
                episodes = []
                for ep_dict in checkpoint["tier1_episodes"]:
                    # Episode tensors stay on CPU by convention
                    # (processor.py and interface.py both call .cpu() on creation)
                    episodes.append(Episode(**ep_dict))

                processor._tier1._hot = episodes[:hot_cap]
                processor._tier1._cold = episodes[hot_cap:]
                processor._tier1._next_id = checkpoint.get("tier1_next_id", len(episodes))

            # Cognitive module state dicts
            if processor._world_model is not None and "world_model_state" in checkpoint:
                processor._world_model.load_state_dict(checkpoint["world_model_state"])
            if processor._goals is not None and "goals_state" in checkpoint:
                processor._goals.load_state_dict(checkpoint["goals_state"])
            if processor._metacognition is not None and "metacognition_state" in checkpoint:
                processor._metacognition.load_state_dict(checkpoint["metacognition_state"])
            if processor._valence is not None and "valence_state" in checkpoint:
                processor._valence.load_state_dict(checkpoint["valence_state"])
            if processor._salience_gate is not None and "salience_state" in checkpoint:
                processor._salience_gate.load_state_dict(checkpoint["salience_state"])

            # Move all modules to device
            dev = processor._device
            processor._perceiver.to(dev)
            processor._backbone.to(dev)
            processor._tier0.field.to(dev)
            processor._tier2.to(dev)
            processor._tier3.to(dev)
            processor._transitions.to(dev)

        elif checkpoint.get("processor_type") == "streaming":
            from sfp.types import SurpriseMetric

            processor.field.load_state_dict(checkpoint["field_state_dict"])
            if "optimizer_state" in checkpoint:
                try:
                    processor._optimizer.load_state_dict(checkpoint["optimizer_state"])
                except (ValueError, KeyError):
                    logger.warning("Could not restore optimizer state")

            if "surprise_history" in checkpoint:
                processor._history = [
                    SurpriseMetric(**m) for m in checkpoint["surprise_history"]
                ]
            processor._step_count = checkpoint.get("step_count", 0)
            processor._surprise_ema = checkpoint.get("surprise_ema", 0.0)

        return processor, create_kwargs, metadata
