"""Generic observation token type specifications.

Defines the standard token contracts that any bridge can produce.
Bridges are responsible for mapping environment-specific data (game frames,
entity lists, sensor readings, etc.) into these generic token formats before
sending them to the SFP.

The SFP itself is environment-agnostic — it processes (1, N, d_model) token
tensors without knowing what game or simulation produced them.  These specs
document the expected semantics so bridge authors know what to produce.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenTypeSpec:
    """Specification for a type of token in a multi-token observation.

    Attributes:
        name: Identifier for this token type (e.g., ``"visual"``, ``"entity"``).
        min_features: Minimum raw feature dimensions before projection to d_model.
        max_count: Suggested maximum tokens of this type per observation.
            ``None`` means no inherent limit.
        description: Human-readable description of what this token type carries.
    """

    name: str
    min_features: int
    max_count: int | None = None
    description: str = ""


# ---------------------------------------------------------------------------
# Standard token types — any bridge can produce these
# ---------------------------------------------------------------------------

VISUAL_PATCH = TokenTypeSpec(
    name="visual",
    min_features=3,
    max_count=None,
    description=(
        "Visual patch token.  Each token represents a spatial region of "
        "a visual frame (e.g., a 16×15 pixel patch).  May include RGB, "
        "depth, or fused visual features.  Bridges should project raw "
        "patch features to d_model before sending."
    ),
)

ENTITY = TokenTypeSpec(
    name="entity",
    min_features=10,
    max_count=64,
    description=(
        "Per-entity token.  Each token represents one discrete object or "
        "agent in the environment.  Minimum features: type identifier, "
        "relative position (3D), distance, health/state.  Bridges should "
        "include threat/category indicators and screen position if available."
    ),
)

STATE = TokenTypeSpec(
    name="state",
    min_features=1,
    max_count=None,
    description=(
        "Structured state token.  Encodes non-visual environment state "
        "(agent vitals, inventory, time-of-day, etc.) as one or more "
        "d_model-dimensional tokens."
    ),
)

DEPTH_PATCH = TokenTypeSpec(
    name="depth",
    min_features=1,
    max_count=None,
    description=(
        "Depth patch token.  Each token carries per-pixel distance "
        "information for a spatial region.  Can be fused with VISUAL_PATCH "
        "tokens by the bridge (recommended) or sent as separate tokens."
    ),
)

TEMPORAL_DIFF = TokenTypeSpec(
    name="temporal",
    min_features=1,
    max_count=None,
    description=(
        "Temporal difference token.  Encodes what changed between "
        "consecutive observations.  Bridges should only emit tokens for "
        "regions with significant change (sparse representation)."
    ),
)

# Registry of all standard token types for programmatic access
STANDARD_TOKEN_TYPES: dict[str, TokenTypeSpec] = {
    spec.name: spec
    for spec in [VISUAL_PATCH, ENTITY, STATE, DEPTH_PATCH, TEMPORAL_DIFF]
}
