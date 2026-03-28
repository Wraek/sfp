"""Cryptographic integrity utilities for the memory hierarchy.

Implements SHA-256 hashing over episode contents and model weights
as described in Defense Layer 4 (Replay Buffer Integrity).
"""

from __future__ import annotations

import hashlib

import torch


def compute_weight_hash(model: torch.nn.Module) -> bytes:
    """Compute a SHA-256 hash over all model parameters.

    This captures the exact state of the model's weights at a given point,
    used to fingerprint the model state when episodes are created.
    """
    hasher = hashlib.sha256()
    for param in model.parameters():
        hasher.update(param.data.cpu().numpy().tobytes())
    return hasher.digest()


def compute_episode_hash(
    input_embedding: torch.Tensor,
    logit_snapshot: torch.Tensor,
    weight_hash: bytes,
) -> bytes:
    """Compute the integrity hash for an episode.

    The hash covers the triplet of (input embedding, model logits at storage time,
    model weight hash). Modifying any component invalidates the hash, preventing
    undetected tampering of stored episodes.

    Args:
        input_embedding: The episode's input embedding vector.
        logit_snapshot: The model's output logits when the episode was stored.
        weight_hash: SHA-256 hash of model weights at storage time.

    Returns:
        SHA-256 digest bytes.
    """
    hasher = hashlib.sha256()
    hasher.update(input_embedding.detach().cpu().numpy().tobytes())
    hasher.update(logit_snapshot.detach().cpu().numpy().tobytes())
    hasher.update(weight_hash)
    return hasher.digest()


def verify_episode_integrity(
    input_embedding: torch.Tensor,
    logit_snapshot: torch.Tensor,
    weight_hash: bytes,
    stored_hash: bytes,
) -> bool:
    """Verify an episode's integrity hash matches its contents.

    Returns True if the hash is valid, False if corruption or tampering is detected.
    """
    expected = compute_episode_hash(input_embedding, logit_snapshot, weight_hash)
    return expected == stored_hash
