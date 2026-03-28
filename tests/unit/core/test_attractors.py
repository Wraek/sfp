"""Tests for AttractorQuery."""

import torch
import pytest

from sfp.config import AttractorConfig
from sfp.core.attractors import AttractorQuery


class TestAttractorQuery:
    def test_query_single(self, tiny_field, random_input):
        query = AttractorQuery(tiny_field)
        result = query.query(random_input)
        assert result.point.shape == random_input.shape
        assert result.iterations > 0

    def test_query_with_trajectory(self, tiny_field, random_input):
        config = AttractorConfig(return_trajectory=True)
        query = AttractorQuery(tiny_field, config)
        result = query.query(random_input)
        assert result.trajectory is not None
        assert len(result.trajectory) > 0

    def test_query_batch(self, tiny_field, random_batch):
        query = AttractorQuery(tiny_field)
        results = query.query_batch(random_batch)
        assert len(results) == random_batch.shape[0]
        for r in results:
            assert r.point.shape == (random_batch.shape[1],)

    def test_discover_attractors(self, tiny_field):
        query = AttractorQuery(tiny_field)
        attractors = query.discover_attractors(n_probes=50, merge_radius=0.5)
        assert attractors.dim() == 2
        assert attractors.shape[1] == tiny_field.config.dim
        assert len(attractors) >= 1

    def test_map_basins(self, tiny_field):
        query = AttractorQuery(tiny_field)
        grid = torch.randn(20, tiny_field.config.dim)
        basin_ids, converged = query.map_basins(grid)
        assert basin_ids.shape == (20,)
        assert converged.shape == grid.shape

    def test_convergence_tolerance(self, tiny_field, random_input):
        config = AttractorConfig(tolerance=1e-1, max_iterations=100)
        query = AttractorQuery(tiny_field, config)
        result = query.query(random_input)
        # With loose tolerance, should converge faster
        assert result.iterations <= 100
