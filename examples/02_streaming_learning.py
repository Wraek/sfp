"""Example 02: Streaming Learning — surprise-gated weight updates.

Demonstrates:
- Creating a StreamingProcessor with LoRA and EWC
- Processing a stream of data with surprise gating
- Adaptive surprise thresholds
- The manifold changing shape as it learns
"""

import torch

import sfp

# Create a streaming processor (with LoRA + EWC forgetting protection)
processor = sfp.create_field("tiny", lora=True, ewc=True, device="cpu")

print("=== Streaming Learning ===")
print(f"Field: {processor.field.config.n_layers}x{processor.field.config.dim}")
print(f"LoRA: rank={processor.lora_manager.config.rank}, "
      f"params={processor.lora_manager.total_lora_params:,}")

# Generate two clusters of data
cluster_a = torch.randn(50, 256) + torch.tensor([2.0] + [0.0] * 255)
cluster_b = torch.randn(50, 256) + torch.tensor([-2.0] + [0.0] * 255)

# Stream cluster A
print("\n--- Streaming Cluster A (50 points) ---")
results_a = processor.process_stream(cluster_a)
updates_a = sum(1 for r in results_a if r.updated)
avg_surprise_a = sum(r.grad_norm for r in results_a) / len(results_a)
print(f"Updates: {updates_a}/{len(results_a)}")
print(f"Avg surprise: {avg_surprise_a:.4f}")
print(f"Final loss: {results_a[-1].loss:.4f}")

# Stream cluster B (distribution shift!)
print("\n--- Streaming Cluster B (50 points) ---")
results_b = processor.process_stream(cluster_b)
updates_b = sum(1 for r in results_b if r.updated)
avg_surprise_b = sum(r.grad_norm for r in results_b) / len(results_b)
print(f"Updates: {updates_b}/{len(results_b)}")
print(f"Avg surprise: {avg_surprise_b:.4f}")
print(f"Final loss: {results_b[-1].loss:.4f}")

# Check total history
print(f"\n--- Summary ---")
print(f"Total steps: {len(processor.surprise_history)}")
print(f"Total updates: {updates_a + updates_b}")
