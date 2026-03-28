"""SemanticFieldProcessor — the MLP whose weights define a concept manifold."""

from __future__ import annotations

import torch
import torch.nn as nn

from sfp.config import FieldConfig


class ResidualBlock(nn.Module):
    """Wraps a sub-module with a residual (skip) connection: output = x + block(x)."""

    def __init__(self, block: nn.Module) -> None:
        super().__init__()
        self.block = block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SemanticFieldProcessor(nn.Module):
    """An MLP whose weights define a concept manifold.

    Knowledge is encoded in the weight geometry — curvature, attractor basins,
    and topological features. Maps R^d -> R^d.

    Args:
        config: FieldConfig specifying dim, n_layers, activation, etc.
    """

    def __init__(self, config: FieldConfig) -> None:
        super().__init__()
        self.config = config
        self._build_network()

    def _build_network(self) -> None:
        activation_fn: type[nn.Module]
        if self.config.activation == "gelu":
            activation_fn = nn.GELU
        elif self.config.activation == "relu":
            activation_fn = nn.ReLU
        else:
            raise ValueError(f"Unknown activation: {self.config.activation!r}")

        dim = self.config.dim

        if self.config.residual:
            blocks: list[nn.Module] = []
            for i in range(self.config.n_layers):
                linear = nn.Linear(dim, dim)
                if i < self.config.n_layers - 1:
                    block = nn.Sequential(linear, activation_fn())
                    blocks.append(ResidualBlock(block))
                else:
                    # Last layer: no activation, no residual
                    blocks.append(linear)
            self.net = nn.Sequential(*blocks)
        else:
            layers: list[nn.Module] = []
            for i in range(self.config.n_layers):
                layers.append(nn.Linear(dim, dim))
                if i < self.config.n_layers - 1:
                    layers.append(activation_fn())
            self.net = nn.Sequential(*layers)

        if self.config.use_layernorm:
            self.norm = nn.LayerNorm(dim)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the manifold. Shape: (*, dim) -> (*, dim)."""
        return self.norm(self.net(x))

    def linear_layers(self) -> list[nn.Linear]:
        """Return all Linear layers in the network (needed by LoRA, EWC)."""
        return [m for m in self.net.modules() if isinstance(m, nn.Linear)]

    @property
    def param_count(self) -> int:
        """Total number of parameters in the manifold."""
        return sum(p.numel() for p in self.parameters())

    def memory_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        """Estimated memory for weights in the given dtype."""
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.param_count * element_size

    def get_weight_summary(self) -> torch.Tensor:
        """Return a single vector summarizing the current weight state.

        Used by episodic memory (Tier 1) to snapshot the working memory state
        when creating episodes. Returns the mean of all parameter tensors,
        flattened and concatenated.
        """
        with torch.no_grad():
            summaries = []
            for p in self.parameters():
                # Per-parameter: mean and std as a 2-element summary
                summaries.append(torch.stack([p.data.mean(), p.data.std()]))
            return torch.cat(summaries)

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the Jacobian matrix dF/dx at point(s) x.

        Args:
            x: (dim,) for single point or (batch, dim) for batched.

        Returns:
            (dim, dim) for single point or (batch, dim, dim) for batched.
        """
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        batch_size = x.shape[0]
        dim = self.config.dim
        x_rep = x.unsqueeze(1).expand(batch_size, dim, dim).reshape(batch_size * dim, dim)
        x_rep = x_rep.detach().requires_grad_(True)

        y = self.forward(x_rep)
        # y is (batch*dim, dim). We want dy_j/dx_i for each batch element.
        # Create selector: for the k-th copy of batch element b, select output dimension k.
        selector = torch.eye(dim, device=x.device, dtype=x.dtype).unsqueeze(0)
        selector = selector.expand(batch_size, dim, dim).reshape(batch_size * dim, dim)

        # Sum over output dims weighted by selector so each copy focuses on one output
        (y * selector).sum().backward()

        # x_rep.grad is (batch*dim, dim) — reshape to (batch, dim, dim)
        jac = x_rep.grad.reshape(batch_size, dim, dim)

        if single:
            jac = jac.squeeze(0)
        return jac
