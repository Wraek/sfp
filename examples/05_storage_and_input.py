"""Example 05: Storage and Input Layer — quantization, checkpointing, byte-level encoding.

Demonstrates:
- INT8 quantization with fidelity check
- Checkpoint save/load round-trip
- Byte-level encoding with entropy-based patching
- Dimensionality projection for input alignment
"""

import tempfile
import os

import torch

import sfp
from sfp.storage.quantization import ManifoldQuantizer
from sfp.storage.serialization import ManifoldCheckpoint
from sfp.input.bytelevel import ByteLevelEncoder, EntropyEstimator, BytePatcher, ByteLevelConfig
from sfp.input.projection import DimensionalityProjection
from sfp.input.adapters import PrecomputedAdapter
from sfp.input.registry import EncoderRegistry

# === Quantization ===
print("=== INT8 Quantization ===")
field = sfp.SemanticFieldProcessor(sfp.FieldConfig.from_preset(sfp.FieldSize.TINY))
x = torch.randn(10, 256)
orig_out = field(x)

qstate = ManifoldQuantizer.quantize(field)
info_bits = ManifoldQuantizer.estimate_information_content(field)
print(f"Information content: {info_bits:.2f} bits/param")

restored = ManifoldQuantizer.dequantize(qstate)
rest_out = restored(x)
mse = (orig_out - rest_out).pow(2).mean().item()
print(f"Quantization MSE: {mse:.6f}")
print(f"Memory: {field.memory_bytes() // 1024}KB (FP32) -> ~{field.param_count // 1024}KB (INT8)")

# === Checkpointing ===
print(f"\n=== Checkpoint Save/Load ===")
processor = sfp.create_field("tiny", device="cpu")
for _ in range(5):
    processor.process(torch.randn(256))

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "manifold.pt")
    ManifoldCheckpoint.save(path, processor.field, processor, metadata={"example": True})
    size_kb = os.path.getsize(path) / 1024
    print(f"Saved: {size_kb:.1f} KB")

    loaded_field, loaded_sp, meta = ManifoldCheckpoint.load(path)
    print(f"Loaded: field_dim={loaded_field.config.dim}, history={len(loaded_sp.surprise_history)}")
    print(f"Metadata: {meta}")

# === Byte-Level Encoding ===
print(f"\n=== Byte-Level Encoder ===")
encoder = ByteLevelEncoder()

texts = [
    "Knowledge is encoded as manifold shape",
    "def process(x): return field(x)",
    '{"config": {"dim": 512}}',
]
embeddings = encoder.encode(texts)
print(f"Encoded {len(texts)} inputs -> {embeddings.shape}")

# Show entropy-based patching
est = EntropyEstimator.fit_default()
config = ByteLevelConfig(min_patch_size=2, max_patch_size=16)
patcher = BytePatcher(est, config)

for text in texts:
    patches = patcher.patch(text.encode("utf-8"))
    patch_lens = [len(p) for p in patches]
    print(f"  '{text[:30]}...' -> {len(patches)} patches, sizes={patch_lens}")

# === Projection ===
print(f"\n=== Dimensionality Projection ===")
proj = DimensionalityProjection(256, 64)
projected = proj(embeddings)
print(f"Projected: {embeddings.shape} -> {projected.shape}")

# Train alignment
src = torch.randn(100, 256)
tgt = torch.randn(100, 64)
loss = proj.fit(src, tgt, epochs=50)
print(f"Alignment training: final_loss={loss:.6f}")

# === Encoder Registry ===
print(f"\n=== Encoder Registry ===")
available = EncoderRegistry.list_available()
print(f"Available: {available}")

# Get a precomputed adapter
adapter = EncoderRegistry.get("precomputed", dim=256)
print(f"Instantiated: {adapter}")
