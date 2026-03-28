"""Attractor-based querying — content-addressable memory via iterative fixed-point convergence."""

from __future__ import annotations

import torch

from sfp.config import AttractorConfig
from sfp.core.field import SemanticFieldProcessor
from sfp.types import AttractorResult
from sfp.utils.logging import get_logger
from sfp.utils.math import pairwise_l2

logger = get_logger("core.attractors")


class AttractorQuery:
    """Content-addressable memory via iterative convergence to fixed points.

    Implements: x_{t+1} = (1 - step) * x_t + step * field(x_t)
    Convergence to a fixed point means the input has been mapped to its
    nearest stored concept in the manifold.
    """

    def __init__(self, field: SemanticFieldProcessor, config: AttractorConfig | None = None) -> None:
        self.field = field
        self.config = config or AttractorConfig()

    @torch.no_grad()
    def query(self, x: torch.Tensor) -> AttractorResult:
        """Converge a single input to its nearest attractor.

        Args:
            x: Input tensor of shape (dim,) or (1, dim).

        Returns:
            AttractorResult with the converged point, iteration count, etc.
        """
        self.field.eval()
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        step = self.config.step_size
        trajectory = [x.clone()] if self.config.return_trajectory else None

        for i in range(self.config.max_iterations):
            x_new = (1 - step) * x + step * self.field(x)
            delta = (x_new - x).norm(dim=-1).mean().item()

            if trajectory is not None:
                trajectory.append(x_new.clone())

            if delta < self.config.tolerance:
                point = x_new.squeeze(0) if single else x_new
                traj = [t.squeeze(0) for t in trajectory] if trajectory and single else trajectory
                return AttractorResult(
                    point=point, iterations=i + 1, converged=True, trajectory=traj
                )
            x = x_new

        point = x.squeeze(0) if single else x
        traj = [t.squeeze(0) for t in trajectory] if trajectory and single else trajectory
        return AttractorResult(
            point=point, iterations=self.config.max_iterations, converged=False, trajectory=traj
        )

    @torch.no_grad()
    def query_batch(self, xs: torch.Tensor) -> list[AttractorResult]:
        """Converge a batch of inputs to their nearest attractors.

        Args:
            xs: Batch tensor of shape (batch, dim).

        Returns:
            List of AttractorResult, one per input.
        """
        self.field.eval()
        batch_size = xs.shape[0]
        step = self.config.step_size

        # Track per-sample convergence
        active = torch.ones(batch_size, dtype=torch.bool, device=xs.device)
        converged_at = torch.full((batch_size,), self.config.max_iterations, dtype=torch.long)
        trajectories: list[list[torch.Tensor]] | None = (
            [[xs[i].clone()] for i in range(batch_size)]
            if self.config.return_trajectory
            else None
        )

        current = xs.clone()

        for iteration in range(self.config.max_iterations):
            # Only compute for active (not yet converged) samples
            if not active.any():
                break

            active_idx = active.nonzero(as_tuple=True)[0]
            active_input = current[active_idx]

            x_new_active = (1 - step) * active_input + step * self.field(active_input)
            deltas = (x_new_active - active_input).norm(dim=-1)

            # Check convergence per sample
            newly_converged = deltas < self.config.tolerance

            # Update current positions for active samples
            current[active_idx] = x_new_active

            # Record trajectories
            if trajectories is not None:
                for j, idx in enumerate(active_idx.tolist()):
                    trajectories[idx].append(current[idx].clone())

            # Mark newly converged samples
            for j, idx in enumerate(active_idx.tolist()):
                if newly_converged[j]:
                    active[idx] = False
                    converged_at[idx] = iteration + 1

        # Build results
        results: list[AttractorResult] = []
        for i in range(batch_size):
            results.append(
                AttractorResult(
                    point=current[i],
                    iterations=converged_at[i].item(),
                    converged=converged_at[i].item() < self.config.max_iterations,
                    trajectory=trajectories[i] if trajectories else None,
                )
            )
        return results

    @torch.no_grad()
    def discover_attractors(
        self, n_probes: int = 1000, merge_radius: float = 0.1
    ) -> torch.Tensor:
        """Discover unique attractors by probing from random starting points.

        Samples random points, converges each to its attractor, then clusters
        the converged points to find unique attractors.

        Args:
            n_probes: Number of random starting points.
            merge_radius: Distance threshold for merging nearby attractors.

        Returns:
            Tensor of shape (n_attractors, dim) with unique attractor centers.
        """
        device = next(self.field.parameters()).device
        dim = self.field.config.dim

        probes = torch.randn(n_probes, dim, device=device)
        results = self.query_batch(probes)
        converged_points = torch.stack([r.point for r in results])

        # Cluster via greedy merge: assign each point to its nearest existing
        # cluster center, or start a new cluster if none within merge_radius.
        centers: list[torch.Tensor] = []
        center_counts: list[int] = []

        for point in converged_points:
            if len(centers) == 0:
                centers.append(point.clone())
                center_counts.append(1)
                continue

            center_stack = torch.stack(centers)
            dists = (center_stack - point.unsqueeze(0)).norm(dim=-1)
            min_dist, min_idx = dists.min(dim=0)

            if min_dist.item() < merge_radius:
                # Merge: running average of the cluster center
                idx = min_idx.item()
                n = center_counts[idx]
                centers[idx] = (centers[idx] * n + point) / (n + 1)
                center_counts[idx] = n + 1
            else:
                centers.append(point.clone())
                center_counts.append(1)

        attractor_tensor = torch.stack(centers)
        logger.info(
            "Discovered %d attractors from %d probes (merge_radius=%.3f)",
            len(centers),
            n_probes,
            merge_radius,
        )
        return attractor_tensor

    @torch.no_grad()
    def map_basins(self, grid_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map each grid point to its attractor basin.

        Args:
            grid_points: Tensor of shape (N, dim).

        Returns:
            (basin_ids, converged_points) where basin_ids is a LongTensor
            of shape (N,) indexing into the attractor codebook, and
            converged_points is (N, dim).
        """
        results = self.query_batch(grid_points)
        converged_points = torch.stack([r.point for r in results])

        # Discover attractors first to get codebook
        codebook = self.discover_attractors(
            n_probes=min(500, grid_points.shape[0]), merge_radius=0.1
        )

        # Assign each converged point to nearest codebook entry
        dists = pairwise_l2(converged_points, codebook)
        basin_ids = dists.argmin(dim=-1)

        return basin_ids, converged_points
