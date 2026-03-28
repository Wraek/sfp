"""Visualization utilities for persistence diagrams, Betti evolution, and basins."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from sfp.utils.logging import get_logger

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.pyplot as plt

    from sfp.core.field import SemanticFieldProcessor
    from sfp.topology.betti import BettiNumberMonitor
    from sfp.types import TopologySnapshot

logger = get_logger("topology.visualization")

# Dimension color map
_DIM_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]


def _get_ax(ax: object | None = None) -> tuple:
    """Get or create a matplotlib axes object."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install sfp[viz]"
        ) from e

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    return ax


def plot_persistence_diagram(
    snapshot: TopologySnapshot, ax: object | None = None
) -> object:
    """Plot a persistence diagram (birth vs death) for each homology dimension.

    Args:
        snapshot: TopologySnapshot with persistence_diagram.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib axes object.
    """
    ax = _get_ax(ax)
    diagram = snapshot.persistence_diagram

    # Get unique dimensions
    dims = sorted(set(int(d) for d in diagram[:, 2]))

    for dim in dims:
        mask = diagram[:, 2] == dim
        births = diagram[mask, 0]
        deaths = diagram[mask, 1]

        # Handle infinite deaths
        finite = np.isfinite(deaths)
        color = _DIM_COLORS[dim % len(_DIM_COLORS)]

        ax.scatter(
            births[finite],
            deaths[finite],
            c=color,
            label=f"H{dim}",
            alpha=0.6,
            s=20,
        )

        # Plot infinite features as triangles at top
        if not np.all(finite):
            max_death = np.max(deaths[finite]) if np.any(finite) else 1.0
            ax.scatter(
                births[~finite],
                np.full(np.sum(~finite), max_death * 1.1),
                c=color,
                marker="^",
                alpha=0.6,
                s=40,
            )

    # Diagonal line
    all_finite = diagram[np.isfinite(diagram[:, 1])]
    if len(all_finite) > 0:
        max_val = max(all_finite[:, 0].max(), all_finite[:, 1].max())
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1)

    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title("Persistence Diagram")
    ax.legend()
    return ax


def plot_betti_evolution(
    monitor: BettiNumberMonitor, ax: object | None = None
) -> object:
    """Plot Betti number evolution over snapshots.

    Args:
        monitor: BettiNumberMonitor with recorded history.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib axes object.
    """
    ax = _get_ax(ax)
    series = monitor.betti_series()

    for dim, values in series.items():
        if not values:
            continue
        color = _DIM_COLORS[dim % len(_DIM_COLORS)]
        ax.plot(values, label=f"beta_{dim}", color=color, marker="o", markersize=3)

    ax.set_xlabel("Snapshot")
    ax.set_ylabel("Betti Number")
    ax.set_title("Betti Number Evolution")
    ax.legend()
    ax.set_ylim(bottom=0)
    return ax


def plot_attractor_basins_2d(
    field: SemanticFieldProcessor,
    grid_range: tuple[float, float] = (-3.0, 3.0),
    resolution: int = 100,
    ax: object | None = None,
) -> object:
    """Plot 2D attractor basin map (first two dimensions, others=0).

    Args:
        field: The SemanticFieldProcessor to visualize.
        grid_range: (min, max) range for the grid.
        resolution: Number of points per axis.
        ax: Optional matplotlib axes.

    Returns:
        matplotlib axes object.
    """
    ax = _get_ax(ax)

    dim = field.config.dim
    device = next(field.parameters()).device

    # Create 2D grid
    xs = torch.linspace(grid_range[0], grid_range[1], resolution)
    ys = torch.linspace(grid_range[0], grid_range[1], resolution)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")

    # Build grid points: set first 2 dims to grid, rest to 0
    grid_flat = torch.zeros(resolution * resolution, dim, device=device)
    grid_flat[:, 0] = xx.flatten()
    if dim > 1:
        grid_flat[:, 1] = yy.flatten()

    # Map basins
    from sfp.core.attractors import AttractorQuery

    query = AttractorQuery(field)
    results = query.query_batch(grid_flat)
    converged = torch.stack([r.point for r in results])

    # Discover attractors for coloring
    attractors = query.discover_attractors(n_probes=200, merge_radius=0.1)

    # Assign basin IDs
    from sfp.utils.math import pairwise_l2

    if len(attractors) > 0:
        dists = pairwise_l2(converged, attractors)
        basin_ids = dists.argmin(dim=-1).cpu().numpy()
    else:
        basin_ids = np.zeros(resolution * resolution)

    # Plot
    basin_map = basin_ids.reshape(resolution, resolution)
    ax.imshow(
        basin_map,
        extent=[grid_range[0], grid_range[1], grid_range[0], grid_range[1]],
        origin="lower",
        cmap="tab20",
        aspect="auto",
    )

    # Mark attractor centers
    if len(attractors) > 0:
        centers = attractors.cpu().numpy()
        ax.scatter(
            centers[:, 0],
            centers[:, 1] if dim > 1 else np.zeros(len(centers)),
            c="red",
            marker="x",
            s=100,
            linewidths=2,
            zorder=5,
        )

    ax.set_xlabel("Dim 0")
    ax.set_ylabel("Dim 1")
    ax.set_title(f"Attractor Basins (2D slice, {len(attractors)} attractors)")
    return ax
