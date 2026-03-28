"""Tests for input.token_types — generic observation token specs."""

from sfp.input.token_types import (
    DEPTH_PATCH,
    ENTITY,
    STANDARD_TOKEN_TYPES,
    STATE,
    TEMPORAL_DIFF,
    VISUAL_PATCH,
    TokenTypeSpec,
)


class TestTokenTypeSpec:
    def test_frozen(self):
        spec = TokenTypeSpec(name="test", min_features=8)
        assert spec.name == "test"
        assert spec.min_features == 8
        assert spec.max_count is None
        assert spec.description == ""

    def test_with_max_count(self):
        spec = TokenTypeSpec(name="entity", min_features=10, max_count=64)
        assert spec.max_count == 64


class TestStandardTokenTypes:
    def test_visual_patch(self):
        assert VISUAL_PATCH.name == "visual"
        assert VISUAL_PATCH.min_features == 3

    def test_entity(self):
        assert ENTITY.name == "entity"
        assert ENTITY.min_features == 10
        assert ENTITY.max_count == 64

    def test_state(self):
        assert STATE.name == "state"
        assert STATE.min_features == 1

    def test_depth_patch(self):
        assert DEPTH_PATCH.name == "depth"

    def test_temporal_diff(self):
        assert TEMPORAL_DIFF.name == "temporal"

    def test_registry_keys(self):
        assert set(STANDARD_TOKEN_TYPES.keys()) == {
            "visual", "entity", "state", "depth", "temporal",
        }

    def test_registry_values_are_specs(self):
        for spec in STANDARD_TOKEN_TYPES.values():
            assert isinstance(spec, TokenTypeSpec)
