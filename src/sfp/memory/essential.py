"""Tier 2: Essential Memory — key-value associative memory with Hopfield retrieval.

Implements the essential memory tier from sfp-hierarchical-memory-implementation.md.
Uses modern continuous Hopfield networks for energy-based retrieval with exponential
storage capacity. Each slot is a concept basin with learned keys, values, confidence
tracking, and modality masks. Also implements the ConsistencyChecker protocol for
Tier 0 defense integration.
"""

from __future__ import annotations

import math
import time

import torch
import torch.nn as nn

from sfp.config import Tier2Config
from sfp.types import ConsistencyChecker
from sfp.utils.logging import get_logger

logger = get_logger("memory.essential")


class EssentialMemory(nn.Module):
    """Tier 2: Key-value associative memory with Hopfield-style retrieval.

    Stores up to n_slots concept basins, each consisting of a learned key vector,
    a value vector, and metadata (confidence, episode count, modality mask, etc.).
    Retrieval uses softmax-attention over keys with confidence weighting.

    Implements the ConsistencyChecker protocol: check_consistency() compares a
    proposed Tier 0 update direction against existing knowledge to detect
    potentially adversarial inputs.

    Args:
        config: Tier2Config specifying slot count, dimensions, etc.
        d_model: Input/key dimensionality.
    """

    def __init__(self, config: Tier2Config | None = None, d_model: int = 512) -> None:
        super().__init__()
        cfg = config or Tier2Config()
        self._config = cfg
        self._d_model = d_model

        # Learnable key-value memory
        self.keys = nn.Parameter(torch.zeros(cfg.n_slots, d_model))
        self.values = nn.Parameter(torch.zeros(cfg.n_slots, cfg.d_value))

        # Projection layers for multi-head attention retrieval
        self.query_proj = nn.Linear(d_model, d_model, bias=False)
        self.key_proj = nn.Linear(d_model, d_model, bias=False)
        self.value_proj = nn.Linear(cfg.d_value, cfg.d_value, bias=False)
        self.output_proj = nn.Linear(cfg.d_value, d_model, bias=False)

        # Metadata buffers (not learnable, but persistent across save/load)
        self.register_buffer("confidence", torch.zeros(cfg.n_slots))
        self.register_buffer("episode_count", torch.zeros(cfg.n_slots, dtype=torch.long))
        self.register_buffer("modality_mask", torch.zeros(cfg.n_slots, dtype=torch.long))
        self.register_buffer("created_at", torch.zeros(cfg.n_slots, dtype=torch.float64))
        self.register_buffer("last_activated", torch.zeros(cfg.n_slots, dtype=torch.float64))
        self.register_buffer("importance", torch.zeros(cfg.n_slots))
        self.register_buffer("active_mask", torch.zeros(cfg.n_slots, dtype=torch.bool))

        # Track number of active slots
        self._n_active: int = 0

        # Initialize projections
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.key_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.xavier_uniform_(self.output_proj.weight)

    def retrieve(
        self, query: torch.Tensor, top_k: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve from essential memory using Hopfield-style energy-based lookup.

        Args:
            query: (B, d_model) or (d_model,) query vector.
            top_k: If > 0, only attend to the top-k most relevant basins.

        Returns:
            Tuple of:
              - output: (B, d_model) retrieved knowledge vector.
              - basin_ids: (B,) index of the nearest (highest-attention) basin per query.
              - attention_weights: (B, n_active) attention distribution over active basins.
        """
        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)
        B = query.shape[0]

        if self._n_active == 0:
            zeros = torch.zeros(B, self._d_model, device=query.device)
            no_basin = torch.full((B,), -1, dtype=torch.long, device=query.device)
            no_attn = torch.zeros(B, 0, device=query.device)
            if single:
                return zeros.squeeze(0), no_basin.squeeze(0), no_attn.squeeze(0)
            return zeros, no_basin, no_attn

        # Get active keys and values
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        active_keys = self.keys[active_idx]  # (A, d_model)
        active_values = self.values[active_idx]  # (A, d_value)
        active_confidence = self.confidence[active_idx]  # (A,)

        # Project query and keys
        Q = self.query_proj(query)  # (B, d_model)
        K = self.key_proj(active_keys)  # (A, d_model)
        V = self.value_proj(active_values)  # (A, d_value)

        # Compute attention scores with temperature scaling
        scale = math.sqrt(self._d_model) * self._config.temperature
        scores = (Q @ K.T) / scale  # (B, A)

        # Confidence weighting: multiply scores by confidence
        scores = scores * active_confidence.unsqueeze(0)

        # Optional top-k masking
        if top_k > 0 and top_k < scores.shape[1]:
            topk_vals, topk_idx = scores.topk(top_k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(1, topk_idx, topk_vals)
            scores = mask

        # Softmax attention
        attn = scores.softmax(dim=-1)  # (B, A)

        # Weighted sum of values
        output = attn @ V  # (B, d_value)
        output = self.output_proj(output)  # (B, d_model)

        # Identify nearest basin per query (highest attention)
        best_local = attn.argmax(dim=-1)  # (B,) indices into active set
        basin_ids = active_idx[best_local]  # (B,) global slot indices

        # Update last_activated timestamps for retrieved basins
        now = time.time()
        unique_basins = basin_ids.unique()
        self.last_activated[unique_basins] = now

        if single:
            return output.squeeze(0), basin_ids.squeeze(0), attn.squeeze(0)
        return output, basin_ids, attn

    def check_consistency(
        self, input_embedding: torch.Tensor, proposed_update: torch.Tensor
    ) -> torch.Tensor:
        """Check whether a proposed Tier 0 update is consistent with essential memory.

        Retrieves the nearest concept basin and computes a consistency score based on
        how much the proposed update direction agrees with the stored knowledge.

        Args:
            input_embedding: The input that triggered the update, shape (B, d) or (d,).
            proposed_update: The output of the Tier 0 forward pass, shape same as input.

        Returns:
            Consistency score in [0, 1]. High scores indicate the update is consistent
            with existing knowledge; low scores suggest potential adversarial manipulation.
        """
        if self._n_active == 0:
            # No basins yet — permit all updates
            if input_embedding.dim() == 1:
                return torch.tensor(1.0, device=input_embedding.device)
            return torch.ones(input_embedding.shape[0], device=input_embedding.device)

        with torch.no_grad():
            output, basin_ids, attn = self.retrieve(input_embedding)

            # Compute cosine similarity between proposed update direction and retrieved knowledge
            single = proposed_update.dim() == 1
            if single:
                proposed_update = proposed_update.unsqueeze(0)
                output = output.unsqueeze(0) if output.dim() == 1 else output

            # Normalize
            update_norm = proposed_update / (proposed_update.norm(dim=-1, keepdim=True) + 1e-8)
            output_norm = output / (output.norm(dim=-1, keepdim=True) + 1e-8)

            # Cosine similarity: 1.0 = fully consistent, -1.0 = contradictory
            cos_sim = (update_norm * output_norm).sum(dim=-1)

            # Map from [-1, 1] to [0, 1] consistency score
            consistency = (cos_sim + 1.0) / 2.0

            # Weight by the confidence of the retrieved basin
            if single:
                bid = basin_ids.item() if basin_ids.dim() == 0 else basin_ids[0].item()
                basin_conf = self.confidence[bid].item()
            else:
                basin_conf_tensor = self.confidence[basin_ids]
                # Low-confidence basins should not block updates
                consistency = consistency * basin_conf_tensor + (1.0 - basin_conf_tensor)
                return consistency

            # For low-confidence basins, don't restrict updates
            consistency_val = consistency.item() * basin_conf + (1.0 - basin_conf)
            return torch.tensor(consistency_val, device=input_embedding.device)

    def allocate_slot(self, key: torch.Tensor, value: torch.Tensor | None = None) -> int:
        """Allocate a new concept basin slot.

        If all slots are full, evicts the slot with the lowest importance score.

        Args:
            key: Initial key vector for the basin, shape (d_model,).
            value: Initial value vector, shape (d_value,). Defaults to zeros.

        Returns:
            The global slot index that was allocated.
        """
        cfg = self._config

        if self._n_active < cfg.n_slots:
            # Find first inactive slot
            inactive = (~self.active_mask).nonzero(as_tuple=True)[0]
            slot = inactive[0].item()
        else:
            # Evict lowest-importance active slot
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            importance_scores = self.importance[active_idx]
            worst = importance_scores.argmin()
            slot = active_idx[worst].item()
            logger.debug("Evicting slot %d (importance=%.4f)", slot, importance_scores[worst].item())
            self._n_active -= 1

        # Initialize the slot
        with torch.no_grad():
            self.keys.data[slot] = key
            if value is not None:
                self.values.data[slot] = value
            else:
                self.values.data[slot].zero_()
            self.confidence[slot] = 0.0
            self.episode_count[slot] = 0
            self.modality_mask[slot] = 0
            self.created_at[slot] = time.time()
            self.last_activated[slot] = time.time()
            self.importance[slot] = 0.0
            self.active_mask[slot] = True

        self._n_active += 1
        return slot

    def update_slot(
        self,
        slot: int,
        key_delta: torch.Tensor | None = None,
        value_delta: torch.Tensor | None = None,
        confidence_update: float = 0.0,
        episode_increment: int = 0,
        modality_bit: int = 0,
        importance_update: float = 0.0,
        ema_decay: float = 0.9,
    ) -> None:
        """Update an existing slot with new information from consolidation.

        Uses exponential moving average for key/value updates.

        Args:
            slot: The global slot index.
            key_delta: Additive update to key, blended via EMA.
            value_delta: Additive update to value, blended via EMA.
            confidence_update: Value to add to confidence (clamped to [0, 1]).
            episode_increment: Number of new episodes to add to count.
            modality_bit: Bitmask OR'd into the modality mask.
            importance_update: Value to add to importance.
            ema_decay: Decay factor for EMA key/value updates.
        """
        with torch.no_grad():
            if key_delta is not None:
                self.keys.data[slot] = ema_decay * self.keys.data[slot] + (1 - ema_decay) * key_delta
            if value_delta is not None:
                self.values.data[slot] = ema_decay * self.values.data[slot] + (1 - ema_decay) * value_delta
            self.confidence[slot] = min(1.0, max(0.0, self.confidence[slot].item() + confidence_update))
            self.episode_count[slot] += episode_increment
            self.modality_mask[slot] |= modality_bit
            self.importance[slot] += importance_update
            self.last_activated[slot] = time.time()

    def get_slot_info(self, slot: int) -> dict:
        """Get metadata for a specific slot."""
        return {
            "slot": slot,
            "active": self.active_mask[slot].item(),
            "confidence": self.confidence[slot].item(),
            "episode_count": self.episode_count[slot].item(),
            "modality_mask": self.modality_mask[slot].item(),
            "created_at": self.created_at[slot].item(),
            "last_activated": self.last_activated[slot].item(),
            "importance": self.importance[slot].item(),
        }

    @property
    def n_active(self) -> int:
        """Number of active concept basins."""
        return self._n_active

    @property
    def active_keys_tensor(self) -> torch.Tensor:
        """Return keys for all active basins, shape (n_active, d_model)."""
        if self._n_active == 0:
            return torch.zeros(0, self._d_model, device=self.keys.device)
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        return self.keys[active_idx]

    @property
    def active_values_tensor(self) -> torch.Tensor:
        """Return values for all active basins, shape (n_active, d_value)."""
        if self._n_active == 0:
            return torch.zeros(0, self._config.d_value, device=self.values.device)
        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        return self.values[active_idx]

    @property
    def active_indices(self) -> torch.Tensor:
        """Return global indices of all active basins."""
        return self.active_mask.nonzero(as_tuple=True)[0]
