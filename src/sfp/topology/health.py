"""Manifold health metrics: composite health report for a concept manifold."""

from __future__ import annotations

import torch

from sfp.config import AttractorConfig
from sfp.core.attractors import AttractorQuery
from sfp.core.field import SemanticFieldProcessor
from sfp.types import HealthReport
from sfp.utils.logging import get_logger

logger = get_logger("topology.health")


class ManifoldHealthMetrics:
    """Computes composite health metrics for a semantic field manifold.

    Measures attractor count, basin radius, topological complexity,
    information density, and spectral gap.
    """

    def __init__(self, field: SemanticFieldProcessor) -> None:
        self.field = field

    def compute(
        self,
        sample_points: torch.Tensor,
        attractor_config: AttractorConfig | None = None,
    ) -> HealthReport:
        """Compute a full health report.

        Args:
            sample_points: Points to probe the manifold, shape (N, dim).
            attractor_config: Optional config for attractor queries.

        Returns:
            HealthReport with all metrics.
        """
        import time

        config = attractor_config or AttractorConfig()
        query = AttractorQuery(self.field, config)

        # 1. Discover attractors
        attractors = query.discover_attractors(
            n_probes=min(len(sample_points), 500), merge_radius=0.1
        )
        attractor_count = len(attractors)

        # 2. Mean basin radius
        mean_basin_radius = self._compute_basin_radius(
            query, sample_points, attractors
        )

        # 3. Topological complexity (optional)
        topo_complexity = self._compute_topo_complexity(sample_points)

        # 4. Information density
        info_density = self._compute_info_density()

        # 5. Spectral gap
        spectral_gap = self._compute_spectral_gap()

        report = HealthReport(
            attractor_count=attractor_count,
            mean_basin_radius=mean_basin_radius,
            topological_complexity=topo_complexity,
            information_density=info_density,
            spectral_gap=spectral_gap,
            timestamp=time.monotonic(),
        )

        logger.info(
            "Health: %d attractors, basin_r=%.4f, info=%.2f bits, spectral_gap=%.4f",
            attractor_count,
            mean_basin_radius,
            info_density,
            spectral_gap,
        )
        return report

    def _compute_basin_radius(
        self,
        query: AttractorQuery,
        sample_points: torch.Tensor,
        attractors: torch.Tensor,
    ) -> float:
        """Compute mean distance from sample points to their attractor."""
        if len(attractors) == 0:
            return 0.0

        results = query.query_batch(sample_points)
        converged = torch.stack([r.point for r in results])

        # Assign each point to nearest attractor
        from sfp.utils.math import pairwise_l2

        dists_to_attractors = pairwise_l2(converged, attractors)
        min_dists = dists_to_attractors.min(dim=-1).values

        return min_dists.mean().item()

    def _compute_topo_complexity(
        self, sample_points: torch.Tensor
    ) -> dict[str, int]:
        """Compute topological complexity via Betti numbers (if giotto-tda available)."""
        try:
            from sfp.topology.homology import PersistentHomologyTracker

            tracker = PersistentHomologyTracker(max_dimension=2, n_sample_points=200)
            snap = tracker.snapshot(self.field, sample_points[:200])
            return {str(i): b for i, b in enumerate(snap.betti_numbers)}
        except ImportError:
            return {}

    def _compute_info_density(self) -> float:
        """Estimate information density via weight entropy."""
        from sfp.storage.quantization import ManifoldQuantizer

        return ManifoldQuantizer.estimate_information_content(self.field)

    def _compute_spectral_gap(self, n_samples: int = 10) -> float:
        """Compute spectral gap from Jacobian singular values.

        Samples random points, computes the Jacobian at each, extracts
        singular values, and reports the median ratio sigma_1/sigma_2.
        Large spectral gap indicates well-separated attractor dynamics.
        """
        device = next(self.field.parameters()).device
        dim = self.field.config.dim

        ratios: list[float] = []
        for _ in range(n_samples):
            x = torch.randn(dim, device=device)
            try:
                jac = self.field.jacobian(x)
                svs = torch.linalg.svdvals(jac)
                if len(svs) >= 2 and svs[1].item() > 1e-8:
                    ratios.append(svs[0].item() / svs[1].item())
            except RuntimeError:
                continue

        if not ratios:
            return 1.0

        ratios.sort()
        return ratios[len(ratios) // 2]  # Median
