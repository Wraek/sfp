"""Dimensionality projection for aligning heterogeneous embedding spaces."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.utils.logging import get_logger

logger = get_logger("input.projection")


class DimensionalityProjection(nn.Module):
    """Learned linear projection between embedding spaces.

    Projects vectors from input_dim to output_dim with optional LayerNorm.
    Can be trained via .fit() to align two embedding spaces (e.g., for
    L1 communication between agents with different embedding models).
    """

    def __init__(
        self, input_dim: int, output_dim: int, normalize: bool = True
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim) if normalize else nn.Identity()

        # Xavier uniform for better gradient flow
        nn.init.xavier_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project and normalize.

        Args:
            x: Tensor of shape (*, input_dim).

        Returns:
            Tensor of shape (*, output_dim).
        """
        return self.norm(self.proj(x))

    def fit(
        self,
        source_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        epochs: int = 100,
        lr: float = 1e-3,
    ) -> float:
        """Train the projection to align source to target embedding space.

        Minimizes MSE between proj(source) and target using Adam optimizer.

        Args:
            source_embeddings: Tensor of shape (N, input_dim).
            target_embeddings: Tensor of shape (N, output_dim).
            epochs: Training epochs.
            lr: Learning rate.

        Returns:
            Final MSE loss value.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.train()

        final_loss = 0.0
        for epoch in range(epochs):
            projected = self(source_embeddings)
            loss = F.mse_loss(projected, target_embeddings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

        self.eval()
        logger.info(
            "Projection fit complete: %d epochs, final_loss=%.6f", epochs, final_loss
        )
        return final_loss
