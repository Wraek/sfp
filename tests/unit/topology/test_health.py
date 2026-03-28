"""Tests for topology.health — ManifoldHealthMetrics."""

import torch
import pytest

from sfp.config import FieldConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.topology.health import ManifoldHealthMetrics
from sfp.types import HealthReport


@pytest.fixture
def metrics():
    field = SemanticFieldProcessor(FieldConfig(dim=32, n_layers=2))
    return ManifoldHealthMetrics(field)


class TestManifoldHealthMetrics:
    def test_compute_returns_health_report(self, metrics):
        samples = torch.randn(50, 32)
        report = metrics.compute(samples)
        assert isinstance(report, HealthReport)
        assert report.attractor_count >= 0
        assert report.mean_basin_radius >= 0
        assert report.spectral_gap > 0
        assert report.timestamp > 0

    def test_compute_with_few_samples(self, metrics):
        samples = torch.randn(5, 32)
        report = metrics.compute(samples)
        assert isinstance(report, HealthReport)

    def test_spectral_gap_reasonable(self, metrics):
        gap = metrics._compute_spectral_gap(n_samples=5)
        assert gap >= 0.0  # ratio of singular values, always positive

    def test_info_density_positive(self, metrics):
        density = metrics._compute_info_density()
        assert density >= 0.0
