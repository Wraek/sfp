"""Device management utilities."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a device string to a torch.device.

    Args:
        device: "auto" (CUDA if available, else CPU), "cpu", "cuda", "cuda:0", "mps".
    """
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)
