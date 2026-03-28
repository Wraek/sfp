"""Abstract InputEncoder base class for the pluggable input layer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseInputEncoder(ABC):
    """Abstract base class for all input encoders.

    Provides the common interface that all encoder backends must implement.
    Subclasses handle specific input types: text embeddings, byte-level
    encoding, precomputed vectors, etc.
    """

    def __init__(self, output_dim: int) -> None:
        self._output_dim = output_dim

    @property
    def output_dim(self) -> int:
        """Dimension of encoded output vectors."""
        return self._output_dim

    @abstractmethod
    def encode(self, inputs: list[Any]) -> torch.Tensor:
        """Encode a list of inputs into a (N, output_dim) tensor.

        Args:
            inputs: List of inputs (type depends on encoder backend).

        Returns:
            Tensor of shape (N, output_dim).
        """
        ...

    def encode_single(self, inp: Any) -> torch.Tensor:
        """Convenience: encode a single input and squeeze the batch dim.

        Args:
            inp: Single input.

        Returns:
            Tensor of shape (output_dim,).
        """
        return self.encode([inp]).squeeze(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output_dim={self._output_dim})"
