"""Example 03: Attractor Queries — content-addressable memory.

Demonstrates:
- Training a field on clustered data
- Querying for nearest attractors (fixed points)
- Discovering all attractors in the manifold
- Basin mapping
"""

import torch

import sfp
from sfp.core.attractors import AttractorQuery

# Train a field on clustered data
processor = sfp.create_field("tiny", lora=False, ewc=False, device="cpu")

# Create 3 well-separated clusters
centers = [
    torch.randn(256) * 0.1 + torch.tensor([3.0] + [0.0] * 255),
    torch.randn(256) * 0.1 + torch.tensor([-3.0] + [0.0] * 255),
    torch.randn(256) * 0.1 + torch.tensor([0.0, 3.0] + [0.0] * 254),
]

print("=== Training on 3 clusters ===")
for epoch in range(3):
    for center in centers:
        cluster = center.unsqueeze(0) + torch.randn(20, 256) * 0.3
        for point in cluster:
            processor.process(point)
    print(f"Epoch {epoch+1}: loss={processor.surprise_history[-1].loss:.4f}")

# Query: find the nearest attractor to a point near cluster 0
query = AttractorQuery(processor.field)

probe = centers[0] + torch.randn(256) * 0.1
result = query.query(probe)
print(f"\n=== Attractor Query ===")
print(f"Probe near cluster 0")
print(f"Converged: {result.converged} in {result.iterations} iterations")
print(f"Attractor point norm: {result.point.norm():.4f}")

# Discover all attractors
print(f"\n=== Attractor Discovery ===")
attractors = query.discover_attractors(n_probes=200, merge_radius=0.5)
print(f"Found {len(attractors)} attractors")
for i, att in enumerate(attractors):
    print(f"  Attractor {i}: norm={att.norm():.4f}, dim0={att[0]:.2f}, dim1={att[1]:.2f}")

# Basin mapping
print(f"\n=== Basin Mapping ===")
grid = torch.randn(50, 256)
basin_ids, converged = query.map_basins(grid)
unique_basins = basin_ids.unique()
print(f"Mapped 50 points to {len(unique_basins)} basins")
for bid in unique_basins:
    count = (basin_ids == bid).sum().item()
    print(f"  Basin {bid.item()}: {count} points")
