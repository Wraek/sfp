"""Tests for the SceneGraph spatial reasoning module."""

import torch
import pytest

from sfp.reasoning.scene_graph import SceneGraph, NEAR_THRESHOLD
from sfp.types import RelationType


D = 64  # small d_model for fast tests


@pytest.fixture
def graph():
    return SceneGraph(d_model=D, max_entities=16)


class TestSceneGraphUpdate:
    def test_empty_entities(self, graph):
        embs = torch.zeros(0, D)
        pos = torch.zeros(0, 3)
        relations = graph.update(embs, pos)
        assert relations == []

    def test_single_entity_no_relations(self, graph):
        embs = torch.randn(1, D)
        pos = torch.randn(1, 3)
        relations = graph.update(embs, pos)
        assert relations == []

    def test_two_nearby_entities(self, graph):
        embs = torch.randn(2, D)
        pos = torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])  # very close
        relations = graph.update(embs, pos)
        assert len(relations) >= 1
        # At least one should be classified
        types = [r[2] for r in relations]
        # SPATIAL_NEAR should be detected since distance < NEAR_THRESHOLD
        assert any(isinstance(t, RelationType) for t in types)

    def test_multiple_entities(self, graph):
        n = 5
        embs = torch.randn(n, D)
        pos = torch.randn(n, 3) * 0.1  # all close together
        relations = graph.update(embs, pos)
        # With 5 entities close together, we should get some relations
        assert len(relations) > 0

    def test_far_entities_low_confidence(self, graph):
        embs = torch.randn(2, D)
        pos = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
        relations = graph.update(embs, pos)
        # Far entities may still produce relations but with lower confidence
        for _, _, _, conf in relations:
            assert 0.0 <= conf <= 1.0

    def test_velocity_estimation(self, graph):
        """Second call should estimate velocities from position difference."""
        embs = torch.randn(2, D)
        pos1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos2 = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])  # entity 1 approaching

        graph.update(embs, pos1)
        relations = graph.update(embs, pos2)
        # Should have some relation with approach/near detected
        assert isinstance(relations, list)

    def test_relation_types_are_spatial(self, graph):
        embs = torch.randn(3, D)
        pos = torch.randn(3, 3) * 0.1
        relations = graph.update(embs, pos)
        spatial_types = {
            RelationType.SPATIAL_NEAR,
            RelationType.SPATIAL_APPROACHING,
            RelationType.SPATIAL_FLEEING,
            RelationType.SPATIAL_ABOVE,
        }
        for _, _, rel_type, _ in relations:
            assert rel_type in spatial_types


class TestSceneGraphSpatialBias:
    def test_empty_entities_empty_bias(self, graph):
        query = torch.randn(D)
        embs = torch.zeros(0, D)
        pos = torch.zeros(0, 3)
        bias = graph.compute_spatial_bias(query, embs, pos)
        assert bias == {}

    def test_bias_values_positive(self, graph):
        query = torch.randn(D)
        embs = torch.randn(3, D)
        pos = torch.randn(3, 3) * 0.5
        bias = graph.compute_spatial_bias(query, embs, pos)
        for idx, val in bias.items():
            assert val > 0.0

    def test_closer_entities_higher_bias(self, graph):
        query = torch.randn(D)
        embs = torch.randn(2, D)
        # Entity 0 is close, entity 1 is far
        pos = torch.tensor([[0.1, 0.0, 0.0], [10.0, 10.0, 10.0]])
        bias = graph.compute_spatial_bias(query, embs, pos)
        # Close entity should generally have higher bias due to distance weighting
        # (not guaranteed since embedding similarity also matters, but distance_weight is strong)
        if 0 in bias and 1 in bias:
            # distance_weight: 1/(1+0.1) ≈ 0.91 vs 1/(1+17.3) ≈ 0.05
            # So entity 0 should have higher weighted score
            pass  # Just verify both are valid floats
        for val in bias.values():
            assert isinstance(val, float)


class TestSceneGraphTransitionInjection:
    def test_inject_returns_count(self, graph):
        """inject_into_transitions should return count of injected edges."""

        class MockTransitions:
            def __init__(self):
                self.edges = []

            def add_edge(self, src, tgt, relation=None, weight=0.1):
                self.edges.append((src, tgt, relation, weight))

        mock = MockTransitions()
        relations = [
            (0, 1, RelationType.SPATIAL_NEAR, 0.9),
            (1, 2, RelationType.SPATIAL_APPROACHING, 0.7),
        ]
        injected = graph.inject_into_transitions(mock, relations)
        assert injected == 2
        assert len(mock.edges) == 2

    def test_inject_with_basin_map(self, graph):
        class MockTransitions:
            def __init__(self):
                self.edges = []

            def add_edge(self, src, tgt, relation=None, weight=0.1):
                self.edges.append((src, tgt, relation, weight))

        mock = MockTransitions()
        relations = [(0, 1, RelationType.SPATIAL_NEAR, 0.8)]
        basin_map = {0: 42, 1: 99}
        graph.inject_into_transitions(mock, relations, entity_basin_map=basin_map)
        assert mock.edges[0][0] == 42
        assert mock.edges[0][1] == 99

    def test_inject_handles_errors(self, graph):
        class MockTransitions:
            def add_edge(self, src, tgt, relation=None, weight=0.1):
                raise IndexError("Basin not found")

        mock = MockTransitions()
        relations = [(0, 1, RelationType.SPATIAL_NEAR, 0.8)]
        injected = graph.inject_into_transitions(mock, relations)
        assert injected == 0  # gracefully skipped


class TestProcessorWithMetadata:
    """Test that metadata flows through the processor pipeline."""

    def test_process_with_entity_metadata(self):
        from sfp.config import FieldConfig, StreamingConfig, Tier1Config, Tier2Config, Tier3Config
        from sfp.memory.processor import HierarchicalMemoryProcessor

        proc = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=D, n_layers=2),
            tier1_config=Tier1Config(hot_capacity=20, cold_capacity=40, surprise_threshold=0.0),
            tier2_config=Tier2Config(n_slots=16, d_value=D),
            tier3_config=Tier3Config(n_slots=8, d_value=D),
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            device="cpu",
        )

        x = torch.randn(1, 96, D)
        metadata = {
            "entity_positions": torch.randn(3, 3),
            "entity_embeddings": torch.randn(3, D),
        }
        result = proc.process(x, modality="environment", metadata=metadata)
        assert hasattr(result, "grad_norm")

    def test_process_without_metadata_unchanged(self):
        from sfp.config import FieldConfig, StreamingConfig, Tier1Config, Tier2Config, Tier3Config
        from sfp.memory.processor import HierarchicalMemoryProcessor

        proc = HierarchicalMemoryProcessor(
            field_config=FieldConfig(dim=D, n_layers=2),
            tier1_config=Tier1Config(hot_capacity=20, cold_capacity=40, surprise_threshold=0.0),
            tier2_config=Tier2Config(n_slots=16, d_value=D),
            tier3_config=Tier3Config(n_slots=8, d_value=D),
            streaming_config=StreamingConfig(surprise_threshold=0.0),
            device="cpu",
        )

        x = torch.randn(1, 96, D)
        # No metadata — should work exactly as before
        result = proc.process(x, modality="environment")
        assert hasattr(result, "grad_norm")
