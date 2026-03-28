"""BackboneTransformer — shared transformer encoder processing Perceiver output."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import BackboneConfig


class _RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def _rotary_embedding(seq_len: int, dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute rotary positional embedding sin/cos tables."""
    pos = torch.arange(seq_len, device=device, dtype=torch.float32).unsqueeze(1)
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    angles = pos * freqs
    return angles.sin(), angles.cos()


def _apply_rotary(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """Apply rotary embedding to input tensor. x: (..., seq, dim)."""
    d_half = x.shape[-1] // 2
    x1, x2 = x[..., :d_half], x[..., d_half:]
    sin = sin[:x.shape[-2], :]
    cos = cos[:x.shape[-2], :]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class _SelfAttention(nn.Module):
    """Multi-head self-attention with rotary positional embeddings."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        Q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply rotary embeddings to Q and K
        sin, cos = _rotary_embedding(N, self.head_dim, x.device)
        Q = _apply_rotary(Q, sin, cos)
        K = _apply_rotary(K, sin, cos)

        dp = self.dropout.p if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, dropout_p=dp)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out_proj(out)


class _FeedForward(nn.Module):
    """SwiGLU-style feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    _silu = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self._silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm, rotary attention, SwiGLU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = _RMSNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads, dropout)
        self.norm2 = _RMSNorm(d_model)
        self.ff = _FeedForward(d_model, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class BackboneTransformer(nn.Module):
    """Shared backbone transformer encoder.

    Processes the fixed-size latent output from Perceiver IO before the
    hierarchical memory system. Uses pre-norm with RMSNorm, rotary positional
    embeddings, and SwiGLU feed-forward networks.

    Args:
        config: BackboneConfig specifying layers, dimensions, heads, etc.
    """

    def __init__(self, config: BackboneConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ]
        )
        self.norm = _RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process latent sequence through transformer layers.

        Args:
            x: (B, N, d_model) — latent sequence from Perceiver IO.

        Returns:
            (B, N, d_model) — contextualized representations.
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.param_count * element_size
