"""Example 01: Basic Semantic Field — create a manifold and explore it.

Demonstrates:
- Creating a SemanticFieldProcessor
- Forward pass (the manifold mapping)
- Jacobian computation for spectral analysis
- Parameter counting and memory estimation
"""

import torch

import sfp

# Create a TINY field (4x256 MLP, ~262K params)
config = sfp.FieldConfig.from_preset(sfp.FieldSize.TINY)
field = sfp.SemanticFieldProcessor(config)

print(f"Field: {config.n_layers} layers x {config.dim} dim")
print(f"Parameters: {field.param_count:,}")
print(f"Memory (FP32): {field.memory_bytes() / 1024:.1f} KB")
print(f"Memory (FP16): {field.memory_bytes(torch.float16) / 1024:.1f} KB")

# Forward pass: the manifold maps R^d -> R^d
x = torch.randn(config.dim)
y = field(x)
print(f"\nInput:  norm={x.norm():.4f}")
print(f"Output: norm={y.norm():.4f}")

# Batch forward pass
batch = torch.randn(8, config.dim)
batch_out = field(batch)
print(f"\nBatch: {batch.shape} -> {batch_out.shape}")

# Jacobian: how the field deforms space at a point
jac = field.jacobian(x)
svs = torch.linalg.svdvals(jac)
print(f"\nJacobian at x: shape={jac.shape}")
print(f"Singular values (top 5): {svs[:5].tolist()}")
print(f"Spectral gap (s1/s2): {svs[0]/svs[1]:.4f}")

# The field defines a manifold — identical inputs always produce identical outputs
y1 = field(x)
y2 = field(x)
print(f"\nDeterminism check: outputs identical = {torch.allclose(y1, y2)}")
