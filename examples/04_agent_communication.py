"""Example 04: Agent Communication — L0-L4 protocol demo.

Demonstrates:
- Setting up two CommEndpoints with shared manifolds
- L0 (raw text), L1 (embedding), L2 (manifold coordinate) communication
- Compression ratios at each layer
- Manifold synchronization and drift detection
"""

import torch

import sfp
from sfp.comms.protocol import CommEndpoint
from sfp.comms.sync import ManifoldSynchronizer

# Create two agents with identical initial manifolds
config = sfp.FieldConfig(dim=64, n_layers=3)
field_a = sfp.SemanticFieldProcessor(config)
field_b = sfp.SemanticFieldProcessor(config)
field_b.load_state_dict(field_a.state_dict())  # Same initial weights

ep_a = CommEndpoint(field_a, "Agent-A")
ep_b = CommEndpoint(field_b, "Agent-B")

raw_size = config.dim * 4  # FP32 bytes for one concept
concept = torch.randn(config.dim)

# --- L0: Raw Text (baseline) ---
print("=== L0: Raw Text ===")
msg_l0 = ep_a.encode_text("The concept of gravity causes objects to fall")
print(f"Size: {msg_l0.size_bytes()} bytes")

decoded_l0 = ep_b.decode(msg_l0)
print(f"Decoded: {decoded_l0[:50]}...")

# --- L1: Embedding ---
print(f"\n=== L1: Embedding (INT8 quantized) ===")
msg_l1 = ep_a.encode(concept, layer=sfp.CommLayer.L1_EMBEDDING)
decoded_l1 = ep_b.decode(msg_l1)
mse_l1 = (concept - decoded_l1).pow(2).mean().item()
print(f"Size: {msg_l1.size_bytes()} bytes (vs {raw_size} raw)")
print(f"Compression: {msg_l1.compression_ratio(raw_size):.1f}x")
print(f"Round-trip MSE: {mse_l1:.6f}")

# --- L2: Manifold Coordinate ---
print(f"\n=== L2: Manifold Coordinate ===")
n_attractors = ep_a.build_codebook(n_probes=200, merge_radius=0.3)
ep_b._l2.set_codebook(ep_a.codebook)  # Share codebook

msg_l2 = ep_a.encode(concept, layer=sfp.CommLayer.L2_MANIFOLD_COORD)
decoded_l2 = ep_b.decode(msg_l2)
mse_l2 = (concept - decoded_l2).pow(2).mean().item()
print(f"Codebook: {n_attractors} attractors")
print(f"Size: {msg_l2.size_bytes()} bytes (vs {raw_size} raw)")
print(f"Compression: {msg_l2.compression_ratio(raw_size):.1f}x")
print(f"Round-trip MSE: {mse_l2:.6f}")

# --- L3: Weight Deformation ---
print(f"\n=== L3: Weight Deformation ===")
# Simulate learning: agent A processes some data
sp_a = sfp.StreamingProcessor(field_a)
for _ in range(10):
    sp_a.process(torch.randn(config.dim))

# Compute weight delta
delta = {}
for (name_a, p_a), (name_b, p_b) in zip(
    field_a.named_parameters(), field_b.named_parameters()
):
    delta[name_a] = p_a.data - p_b.data

msg_l3 = ep_a.encode_deformation(delta)
print(f"Size: {msg_l3.size_bytes()} bytes")
print(f"Compression vs full sync: {(raw_size * len(delta)) / max(1, msg_l3.size_bytes()):.1f}x")

# Agent B applies the deformation
ep_b.apply_deformation(msg_l3)

# --- Drift Detection ---
print(f"\n=== Manifold Synchronization ===")
sync_a = ManifoldSynchronizer(field_a)
sync_b = ManifoldSynchronizer(field_b, anchor_points=sync_a._anchors)

fp_a = sync_a.compute_fingerprint()
fp_b = sync_b.compute_fingerprint()
drift = sync_a.detect_drift(fp_a, fp_b)
print(f"Post-sync drift: {drift:.6f}")
print(f"Needs sync: {sync_a.needs_sync(fp_b)}")
