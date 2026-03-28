"""Preset configurations for the SFP GUI.

Each preset defines the kwargs for sfp.create_field() and groups
advanced config overrides by category for the config panel.
"""

from __future__ import annotations

from dataclasses import fields
from typing import Any

from sfp import config as cfg

# ── create_field() kwargs per preset ────────────────────────────────

PRESETS: dict[str, dict[str, Any]] = {
    "minimal": {
        "size": "small",
        "streaming": True,
        "hierarchical": False,
        "lora": True,
        "ewc": True,
        "world_model": False,
        "goals": False,
        "metacognition": False,
        "valence": False,
        "selective_attention": False,
        "generative_replay": False,
        "device": "auto",
    },
    "standard": {
        "size": "medium",
        "streaming": True,
        "hierarchical": True,
        "lora": True,
        "ewc": True,
        "world_model": True,
        "goals": True,
        "metacognition": False,
        "valence": False,
        "selective_attention": False,
        "generative_replay": False,
        "device": "auto",
    },
    "full": {
        "size": "large",
        "streaming": True,
        "hierarchical": True,
        "lora": True,
        "ewc": True,
        "world_model": True,
        "goals": True,
        "metacognition": True,
        "valence": True,
        "selective_attention": True,
        "generative_replay": True,
        "device": "auto",
    },
}

# ── Config dataclass categories for the advanced panel ──────────────

CONFIG_CATEGORIES: dict[str, list[type]] = {
    "Core": [cfg.FieldConfig, cfg.StreamingConfig],
    "Adapters": [cfg.LoRAConfig, cfg.EWCConfig],
    "Memory": [
        cfg.Tier0Config,
        cfg.Tier1Config,
        cfg.Tier2Config,
        cfg.Tier3Config,
        cfg.ConsolidationConfig,
    ],
    "Reasoning": [cfg.TransitionConfig, cfg.ReasoningChainConfig],
    "Cognitive": [
        cfg.WorldModelConfig,
        cfg.GoalPersistenceConfig,
        cfg.MetacognitionConfig,
        cfg.ValenceConfig,
        cfg.SelectiveAttentionConfig,
        cfg.GenerativeReplayConfig,
    ],
    "Defense": [cfg.DefenseConfig],
    "Architecture": [cfg.PerceiverConfig, cfg.BackboneConfig],
}


def get_default_overrides() -> dict[str, dict[str, Any]]:
    """Return default values for every config dataclass, keyed by class name."""
    result: dict[str, dict[str, Any]] = {}
    for classes in CONFIG_CATEGORIES.values():
        for cls in classes:
            defaults = {}
            for f in fields(cls):
                defaults[f.name] = f.default if f.default is not f.default_factory else f.default_factory()  # type: ignore[arg-type]
            result[cls.__name__] = defaults
    return result
