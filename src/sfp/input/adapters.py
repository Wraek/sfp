"""Embedding backend adapters: sentence-transformers, CLIP, precomputed."""

from __future__ import annotations

from typing import Any

import torch

from sfp.input.encoder import BaseInputEncoder
from sfp.utils.logging import get_logger

logger = get_logger("input.adapters")


class SentenceTransformerAdapter(BaseInputEncoder):
    """Wraps sentence-transformers models as an InputEncoder.

    Requires the `sentence-transformers` package (install with `pip install sfp[embeddings]`).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for SentenceTransformerAdapter. "
                "Install with: pip install sfp[embeddings]"
            ) from e

        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        super().__init__(dim)
        logger.info("Loaded SentenceTransformer '%s' (dim=%d)", model_name, dim)

    def encode(self, inputs: list[str]) -> torch.Tensor:
        """Encode text strings to embeddings.

        Args:
            inputs: List of text strings.

        Returns:
            Tensor of shape (N, output_dim).
        """
        embeddings = self._model.encode(inputs, convert_to_numpy=True)
        return torch.tensor(embeddings, dtype=torch.float32)


class CLIPAdapter(BaseInputEncoder):
    """Wraps OpenAI CLIP models as an InputEncoder for text and images.

    Requires the `transformers` package (install with `pip install sfp[embeddings]`).
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32") -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as e:
            raise ImportError(
                "transformers is required for CLIPAdapter. "
                "Install with: pip install sfp[embeddings]"
            ) from e

        self._model = CLIPModel.from_pretrained(model_name)
        self._processor = CLIPProcessor.from_pretrained(model_name)
        dim = self._model.config.projection_dim
        super().__init__(dim)
        logger.info("Loaded CLIP '%s' (dim=%d)", model_name, dim)

    def encode(self, inputs: list[str]) -> torch.Tensor:
        """Encode text strings to CLIP embeddings.

        Args:
            inputs: List of text strings.

        Returns:
            Normalized tensor of shape (N, output_dim).
        """
        processed = self._processor(text=inputs, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self._model.get_text_features(**processed)
        # L2 normalize
        return outputs / outputs.norm(dim=-1, keepdim=True)

    def encode_images(self, images: list[Any]) -> torch.Tensor:
        """Encode images to CLIP embeddings.

        Args:
            images: List of PIL Images.

        Returns:
            Normalized tensor of shape (N, output_dim).
        """
        processed = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            outputs = self._model.get_image_features(**processed)
        return outputs / outputs.norm(dim=-1, keepdim=True)


class PrecomputedAdapter(BaseInputEncoder):
    """Pass-through adapter for pre-computed embedding tensors.

    Use when embeddings are already computed externally.
    """

    def __init__(self, dim: int) -> None:
        super().__init__(dim)

    def encode(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        """Stack pre-computed tensors.

        Args:
            inputs: List of tensors, each of shape (output_dim,).

        Returns:
            Tensor of shape (N, output_dim).
        """
        stacked = torch.stack(inputs)
        if stacked.shape[-1] != self.output_dim:
            raise ValueError(
                f"Expected dim {self.output_dim}, got {stacked.shape[-1]}"
            )
        return stacked
