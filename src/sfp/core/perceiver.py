"""Perceiver IO — multi-modal bottleneck that compresses arbitrary inputs to fixed-size latents."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import PerceiverConfig


class _MultiHeadCrossAttention(nn.Module):
    """Cross-attention: queries attend to key-value pairs from a different sequence."""

    def __init__(self, d_model: int, n_heads: int, d_kv: int | None = None) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        d_kv = d_kv or d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_kv, d_model, bias=False)
        self.v_proj = nn.Linear(d_kv, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: (B, Nq, D), kv: (B, Nkv, D_kv) -> (B, Nq, D)."""
        B, Nq, _ = q.shape
        Nkv = kv.shape[1]

        Q = self.q_proj(q).view(B, Nq, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(kv).view(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(kv).view(B, Nkv, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(Q, K, V)
        out = out.transpose(1, 2).reshape(B, Nq, -1)
        return self.out_proj(out)


class _MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.cross_attn = _MultiHeadCrossAttention(d_model, n_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cross_attn(x, x)


class _FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + FFN block."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _MultiHeadSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = _FeedForward(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class PerceiverIO(nn.Module):
    """Perceiver IO multi-modal bottleneck.

    Compresses arbitrary-length input sequences to a fixed set of learned latent
    vectors via cross-attention, processes them with self-attention, and optionally
    decodes via output cross-attention.

    Args:
        config: PerceiverConfig specifying dimensions, layer counts, heads.
    """

    def __init__(self, config: PerceiverConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_latent

        # Learned latent array
        self.latents = nn.Parameter(torch.randn(config.n_latents, d) * 0.02)

        # Input projection (maps arbitrary d_input to d_latent for cross-attn KV)
        self.input_proj = nn.Linear(config.d_input, d, bias=False)

        # Cross-attention layers (input -> latents)
        self.cross_attn_layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_latent": nn.LayerNorm(d),
                        "norm_input": nn.LayerNorm(d),
                        "cross_attn": _MultiHeadCrossAttention(d, config.n_heads),
                    }
                )
                for _ in range(config.n_cross_attn_layers)
            ]
        )

        # Self-attention layers over latents
        self.self_attn_layers = nn.ModuleList(
            [_SelfAttentionBlock(d, config.n_heads) for _ in range(config.n_self_attn_layers)]
        )

        self.output_norm = nn.LayerNorm(d)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode arbitrary-length input to fixed-size latent representation.

        Args:
            inputs: (B, N_tokens, d_input) — variable-length input sequence.

        Returns:
            (B, n_latents, d_latent) — fixed-size latent representation.
        """
        B = inputs.shape[0]

        # Project inputs to latent dimension
        kv = self.input_proj(inputs)

        # Expand learned latents for the batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Cross-attention: latents attend to inputs
        for layer in self.cross_attn_layers:
            latents = latents + layer["cross_attn"](
                layer["norm_latent"](latents),
                layer["norm_input"](kv),
            )

        # Self-attention over latents
        for layer in self.self_attn_layers:
            latents = layer(latents)

        return self.output_norm(latents)

    def decode(self, latents: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        """Decode from latent space using output cross-attention.

        Args:
            latents: (B, n_latents, d_latent) — encoded latent representation.
            query: (B, N_out, d_latent) — output query vectors.

        Returns:
            (B, N_out, d_latent) — decoded output.
        """
        # Simple output cross-attention: queries attend to latent KV
        norm_q = self.output_norm(query)
        norm_kv = self.output_norm(latents)
        # Reuse last cross-attn layer's attention mechanism
        return query + self.cross_attn_layers[-1]["cross_attn"](norm_q, norm_kv)

    @property
    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def memory_bytes(self, dtype: torch.dtype = torch.float32) -> int:
        element_size = torch.tensor([], dtype=dtype).element_size()
        return self.param_count * element_size
