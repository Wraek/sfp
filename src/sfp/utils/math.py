"""Shared math operations used across SFP modules."""

from __future__ import annotations

import torch
import torch.nn as nn


def grad_norm(model: nn.Module) -> float:
    """Compute the L2 norm of all gradients in a model.

    Only considers parameters that have gradients attached.
    Returns 0.0 if no parameters have gradients.
    """
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += p.grad.data.norm(2).item() ** 2
    return total_sq**0.5


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between two sets of vectors.

    Args:
        a: (N, D) tensor
        b: (M, D) tensor

    Returns:
        (N, M) tensor of cosine similarities.
    """
    a_norm = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b_norm = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return a_norm @ b_norm.T


def pairwise_l2(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 (Euclidean) distances between two sets of vectors.

    Args:
        a: (N, D) tensor
        b: (M, D) tensor

    Returns:
        (N, M) tensor of distances.
    """
    return torch.cdist(a, b, p=2)
