"""Byte-level encoder with entropy-based dynamic patching (BLT-inspired)."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from sfp.input.encoder import BaseInputEncoder
from sfp.utils.logging import get_logger

logger = get_logger("input.bytelevel")


@dataclass
class ByteLevelConfig:
    """Configuration for byte-level encoding."""

    patch_dim: int = 256
    min_patch_size: int = 2
    max_patch_size: int = 32
    entropy_threshold: float = 2.0
    ngram_order: int = 3


class EntropyEstimator:
    """Character n-gram language model for entropy estimation.

    Builds an empirical n-gram model over raw bytes and estimates the
    conditional entropy at each position. Used to place patch boundaries
    at high-entropy (surprising) byte positions.
    """

    def __init__(self, ngram_order: int = 3) -> None:
        self.ngram_order = ngram_order
        self._counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self._total: dict[tuple[int, ...], int] = defaultdict(int)

    def fit(self, data: bytes) -> None:
        """Fit the n-gram model on raw byte data.

        Args:
            data: Byte string to learn from.
        """
        for i in range(self.ngram_order, len(data)):
            context = tuple(data[i - self.ngram_order : i])
            current = data[i]
            self._counts[context][current] += 1
            self._total[context] += 1

    def estimate_entropy(self, context: bytes) -> float:
        """Estimate conditional entropy given a byte context.

        Args:
            context: The preceding bytes (length should be ngram_order).

        Returns:
            Entropy in bits. Returns 8.0 (max for a byte) if context unseen.
        """
        ctx = tuple(context[-self.ngram_order :])
        if ctx not in self._counts or self._total[ctx] == 0:
            return 8.0  # Maximum entropy for a byte

        total = self._total[ctx]
        entropy = 0.0
        for count in self._counts[ctx].values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    @classmethod
    def fit_default(cls) -> EntropyEstimator:
        """Create an estimator pre-fit on representative mixed data.

        Trains on a mix of English text, Python code, and JSON to provide
        reasonable defaults without requiring user-supplied training data.

        Returns:
            A pre-fit EntropyEstimator.
        """
        estimator = cls(ngram_order=3)

        # Representative English text
        english = (
            "The quick brown fox jumps over the lazy dog. "
            "Knowledge is encoded as the shape of neural network weights, "
            "not as stored data. The manifold's curvature, attractor basins, "
            "and topological features constitute understanding. "
            "External data enters, deforms the manifold via surprise-gated "
            "updates, and is never retained. Each concept occupies a basin "
            "of attraction in the weight-defined landscape. "
            "Communication between agents occurs through shared geometric "
            "coordinates rather than raw text, enabling extreme compression. "
            "The system learns continuously without catastrophic forgetting "
            "through elastic weight consolidation and low-rank adaptation. "
        )

        # Representative Python code
        python_code = (
            "def process(self, x: torch.Tensor) -> SurpriseMetric:\n"
            "    y = self.field(x)\n"
            "    loss = F.mse_loss(y, x)\n"
            "    for strategy in self._forget_strategies:\n"
            "        loss = loss + strategy.penalty(self.field)\n"
            "    self._optimizer.zero_grad()\n"
            "    loss.backward()\n"
            "    grad = utils.grad_norm(self.field)\n"
            "    if grad > self.threshold:\n"
            "        self._optimizer.step()\n"
            "    return SurpriseMetric(grad, loss.item(), True)\n"
        )

        # Representative JSON
        json_data = (
            '{"config": {"dim": 512, "n_layers": 6, "activation": "gelu"}, '
            '"metrics": {"loss": 0.0142, "grad_norm": 0.523, "updated": true}, '
            '"attractor": {"point": [0.1, -0.3, 0.7], "converged": true}}'
        )

        combined = english + python_code + json_data
        estimator.fit(combined.encode("utf-8"))
        return estimator


class BytePatcher:
    """Segments raw bytes into variable-length patches based on entropy.

    Places patch boundaries at high-entropy positions (where the next byte
    is hard to predict), creating shorter patches for complex/surprising
    content and longer patches for predictable content.
    """

    def __init__(
        self, entropy_estimator: EntropyEstimator, config: ByteLevelConfig
    ) -> None:
        self.estimator = entropy_estimator
        self.config = config

    def patch(self, data: bytes) -> list[bytes]:
        """Segment byte data into entropy-based patches.

        Args:
            data: Raw byte data to segment.

        Returns:
            List of byte-string patches.
        """
        if len(data) == 0:
            return []

        patches: list[bytes] = []
        patch_start = 0

        for i in range(len(data)):
            current_len = i - patch_start + 1

            # Force boundary at max patch size
            if current_len >= self.config.max_patch_size:
                patches.append(data[patch_start : i + 1])
                patch_start = i + 1
                continue

            # Check entropy for boundary decision
            if current_len >= self.config.min_patch_size:
                context_start = max(0, i - self.estimator.ngram_order)
                context = data[context_start : i]
                entropy = self.estimator.estimate_entropy(context)

                if entropy > self.config.entropy_threshold:
                    patches.append(data[patch_start : i + 1])
                    patch_start = i + 1

        # Final patch
        if patch_start < len(data):
            patches.append(data[patch_start:])

        return patches


class ByteLevelEncoder(BaseInputEncoder, nn.Module):
    """Encodes raw bytes/text via entropy-based patching and learned embeddings.

    Each byte is embedded, patches are formed via entropy boundaries, then
    a small MLP projects each patch to a fixed-dim vector. The per-input
    output is the mean of all patch vectors.
    """

    def __init__(self, config: ByteLevelConfig | None = None) -> None:
        self._config = config or ByteLevelConfig()
        BaseInputEncoder.__init__(self, self._config.patch_dim)
        nn.Module.__init__(self)

        # Learnable byte embeddings (256 possible byte values)
        self._byte_embedding = nn.Embedding(256, 64)

        # Patch encoder: flattened byte embeddings -> patch_dim
        flat_dim = 64 * self._config.max_patch_size
        self._patch_encoder = nn.Sequential(
            nn.Linear(flat_dim, self._config.patch_dim),
            nn.GELU(),
            nn.Linear(self._config.patch_dim, self._config.patch_dim),
            nn.LayerNorm(self._config.patch_dim),
        )

        # Entropy model and patcher
        self._entropy_estimator = EntropyEstimator.fit_default()
        self._patcher = BytePatcher(self._entropy_estimator, self._config)

    def _encode_patch(self, patch_bytes: bytes) -> torch.Tensor:
        """Encode a single byte patch to a vector.

        Args:
            patch_bytes: Raw bytes for this patch.

        Returns:
            Tensor of shape (patch_dim,).
        """
        # Convert bytes to int tensor
        byte_ids = torch.tensor(
            list(patch_bytes), dtype=torch.long, device=self._byte_embedding.weight.device
        )

        # Look up byte embeddings: (patch_len, 64)
        embedded = self._byte_embedding(byte_ids)

        # Pad to max_patch_size
        pad_len = self._config.max_patch_size - len(patch_bytes)
        if pad_len > 0:
            padding = torch.zeros(
                pad_len, 64, device=embedded.device, dtype=embedded.dtype
            )
            embedded = torch.cat([embedded, padding], dim=0)

        # Flatten and project
        flat = embedded.flatten()
        return self._patch_encoder(flat)

    def encode(self, inputs: list[str | bytes]) -> torch.Tensor:
        """Encode a list of text/bytes inputs to vectors.

        Each input is patched by entropy, each patch is encoded, and
        the per-input result is the mean of patch vectors.

        Args:
            inputs: List of strings or bytes objects.

        Returns:
            Tensor of shape (N, patch_dim).
        """
        results: list[torch.Tensor] = []

        for inp in inputs:
            data = inp.encode("utf-8") if isinstance(inp, str) else inp
            patches = self._patcher.patch(data)

            if not patches:
                # Empty input -> zero vector
                results.append(
                    torch.zeros(
                        self._config.patch_dim,
                        device=self._byte_embedding.weight.device,
                    )
                )
                continue

            patch_vecs = [self._encode_patch(p) for p in patches]
            stacked = torch.stack(patch_vecs)
            results.append(stacked.mean(dim=0))

        return torch.stack(results)

    def encode_patches(self, inputs: list[str | bytes]) -> list[torch.Tensor]:
        """Encode inputs returning per-patch embeddings (not averaged).

        Useful when the downstream consumer wants to process patches
        individually (e.g., streaming processor).

        Args:
            inputs: List of strings or bytes objects.

        Returns:
            List of tensors, each of shape (n_patches_i, patch_dim).
        """
        results: list[torch.Tensor] = []

        for inp in inputs:
            data = inp.encode("utf-8") if isinstance(inp, str) else inp
            patches = self._patcher.patch(data)

            if not patches:
                results.append(
                    torch.zeros(
                        1,
                        self._config.patch_dim,
                        device=self._byte_embedding.weight.device,
                    )
                )
                continue

            patch_vecs = [self._encode_patch(p) for p in patches]
            results.append(torch.stack(patch_vecs))

        return results

    def fit_entropy_model(self, data: bytes) -> None:
        """Re-fit the entropy estimator on domain-specific data.

        Args:
            data: Raw bytes of domain-representative content.
        """
        self._entropy_estimator = EntropyEstimator(self._config.ngram_order)
        self._entropy_estimator.fit(data)
        self._patcher = BytePatcher(self._entropy_estimator, self._config)
        logger.info(
            "Re-fit entropy model on %d bytes of domain data", len(data)
        )
