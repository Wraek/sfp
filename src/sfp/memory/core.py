"""Tier 3: Core Memory — near-frozen axiomatic memory with cryptographic integrity.

Implements the core memory tier from sfp-hierarchical-memory-implementation.md.
Stores the most stable, high-confidence knowledge as near-frozen axioms with
cryptographic integrity hashing. Promotion from Tier 2 requires meeting strict
criteria and authorization via the event system.
"""

from __future__ import annotations

import hashlib
import time

import torch
import torch.nn as nn

from sfp.config import Tier3Config
from sfp.exceptions import IntegrityError
from sfp.memory.essential import EssentialMemory
from sfp.memory.events import PromotionEventEmitter
from sfp.types import PromotionRequest
from sfp.utils.logging import get_logger

logger = get_logger("memory.core")


class CoreMemory(nn.Module):
    """Tier 3: Near-frozen axiomatic memory with cryptographic integrity.

    Stores the most stable, thoroughly validated knowledge as axioms that rarely
    change. Each slot carries a SHA-256 integrity hash that is verified at boot
    and periodically during operation.

    Promotion from Tier 2 requires:
      - Confidence > min_confidence (default 0.9)
      - Episode count > min_episode_count (default 1000)
      - Multi-modal evidence (>= min_modalities modalities observed)
      - Age > min_age_days (default 7 days)
      - Authorization from the event system

    Args:
        config: Tier3Config specifying slot count and promotion criteria.
        d_model: Key dimensionality.
        event_emitter: PromotionEventEmitter for authorization.
    """

    def __init__(
        self,
        config: Tier3Config | None = None,
        d_model: int = 512,
        event_emitter: PromotionEventEmitter | None = None,
    ) -> None:
        super().__init__()
        cfg = config or Tier3Config()
        self._config = cfg
        self._d_model = d_model
        self._event_emitter = event_emitter or PromotionEventEmitter(default_approve=False)

        # Key-value storage
        self.keys = nn.Parameter(torch.zeros(cfg.n_slots, d_model))
        self.values = nn.Parameter(torch.zeros(cfg.n_slots, cfg.d_value))

        # Metadata
        self.register_buffer("confidence", torch.zeros(cfg.n_slots))
        self.register_buffer("episode_count", torch.zeros(cfg.n_slots, dtype=torch.long))
        self.register_buffer("modality_mask", torch.zeros(cfg.n_slots, dtype=torch.long))
        self.register_buffer("active_mask", torch.zeros(cfg.n_slots, dtype=torch.bool))
        self.register_buffer("promoted_at", torch.zeros(cfg.n_slots, dtype=torch.float64))

        # Integrity hashes (stored as Python list since they're bytes, not tensors)
        self._slot_hashes: list[bytes | None] = [None] * cfg.n_slots
        self._n_active: int = 0

    def retrieve(self, query: torch.Tensor) -> torch.Tensor:
        """Retrieve from core memory using softmax attention over active axiom keys.

        Args:
            query: (B, d_model) or (d_model,) query vector.

        Returns:
            (B, d_model) or (d_model,) retrieved axiom knowledge.
        """
        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)

        if self._n_active == 0:
            zeros = torch.zeros(query.shape[0], self._d_model, device=query.device)
            return zeros.squeeze(0) if single else zeros

        active_idx = self.active_mask.nonzero(as_tuple=True)[0]
        active_keys = self.keys[active_idx]  # (A, d_model)
        active_values = self.values[active_idx]  # (A, d_value)
        active_conf = self.confidence[active_idx]  # (A,)

        # Attention scores with confidence weighting
        import math

        scale = math.sqrt(self._d_model)
        scores = (query @ active_keys.T) / scale  # (B, A)
        scores = scores * active_conf.unsqueeze(0)
        attn = scores.softmax(dim=-1)  # (B, A)
        output = attn @ active_values  # (B, d_value)

        # If d_value != d_model, we need to project back (for simplicity, pad/truncate)
        if output.shape[-1] != self._d_model:
            if output.shape[-1] < self._d_model:
                pad = torch.zeros(
                    output.shape[0],
                    self._d_model - output.shape[-1],
                    device=output.device,
                )
                output = torch.cat([output, pad], dim=-1)
            else:
                output = output[..., : self._d_model]

        return output.squeeze(0) if single else output

    def promote_from_tier2(
        self, tier2: EssentialMemory, slot_id: int
    ) -> bool:
        """Attempt to promote a Tier 2 basin to core memory.

        Checks all promotion criteria and requests authorization from the event system.

        Args:
            tier2: The Tier 2 essential memory instance.
            slot_id: The Tier 2 slot index to promote.

        Returns:
            True if promotion succeeded, False if criteria not met or denied.
        """
        cfg = self._config
        info = tier2.get_slot_info(slot_id)

        # Check criteria
        confidence = info["confidence"]
        episode_count = info["episode_count"]
        modality_mask = info["modality_mask"]
        created_at = info["created_at"]

        # Count modalities from bitmask
        modality_count = bin(modality_mask).count("1")

        # Age in days
        age_days = (time.time() - created_at) / 86400.0

        if confidence < cfg.min_confidence:
            logger.debug("Promotion rejected: confidence %.3f < %.3f", confidence, cfg.min_confidence)
            return False
        if episode_count < cfg.min_episode_count:
            logger.debug("Promotion rejected: episodes %d < %d", episode_count, cfg.min_episode_count)
            return False
        if modality_count < cfg.min_modalities:
            logger.debug("Promotion rejected: modalities %d < %d", modality_count, cfg.min_modalities)
            return False
        if age_days < cfg.min_age_days:
            logger.debug("Promotion rejected: age %.1f days < %.1f", age_days, cfg.min_age_days)
            return False

        # Build promotion request and emit to event system
        key_snapshot = tier2.keys[slot_id].detach().clone()
        value_snapshot = tier2.values[slot_id].detach().clone()

        request = PromotionRequest(
            basin_id=slot_id,
            confidence=confidence,
            episode_count=episode_count,
            modality_count=modality_count,
            age_days=age_days,
            key_snapshot=key_snapshot,
            value_snapshot=value_snapshot,
        )

        authorized = self._event_emitter.emit(request)
        if not authorized:
            return False

        # Perform promotion
        return self._write_slot(key_snapshot, value_snapshot, confidence, episode_count, modality_mask)

    def _write_slot(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        confidence: float,
        episode_count: int,
        modality_mask: int,
    ) -> bool:
        """Write a new axiom to core memory.

        Allocates a slot (or evicts the lowest-confidence active slot if full),
        writes the key-value pair, and computes the integrity hash.
        """
        if self._n_active < self._config.n_slots:
            inactive = (~self.active_mask).nonzero(as_tuple=True)[0]
            slot = inactive[0].item()
        else:
            # Evict lowest-confidence axiom
            active_idx = self.active_mask.nonzero(as_tuple=True)[0]
            conf_scores = self.confidence[active_idx]
            worst = conf_scores.argmin()
            slot = active_idx[worst].item()
            logger.info(
                "Evicting core slot %d (confidence=%.3f) to make room",
                slot,
                conf_scores[worst].item(),
            )
            self._n_active -= 1

        with torch.no_grad():
            self.keys.data[slot] = key
            self.values.data[slot] = value
            self.confidence[slot] = confidence
            self.episode_count[slot] = episode_count
            self.modality_mask[slot] = modality_mask
            self.active_mask[slot] = True
            self.promoted_at[slot] = time.time()

        # Compute and store integrity hash
        self._slot_hashes[slot] = self._compute_slot_hash(slot)
        self._n_active += 1

        logger.info("Promoted to core slot %d (confidence=%.3f)", slot, confidence)
        return True

    def verify_integrity(self) -> list[int]:
        """Verify integrity hashes of all active core memory slots.

        Returns:
            List of slot indices that failed integrity verification.
        """
        failed: list[int] = []
        for slot in range(self._config.n_slots):
            if not self.active_mask[slot]:
                continue
            if self._slot_hashes[slot] is None:
                failed.append(slot)
                continue
            current_hash = self._compute_slot_hash(slot)
            if current_hash != self._slot_hashes[slot]:
                failed.append(slot)
                logger.error("Core memory slot %d failed integrity check!", slot)

        if failed:
            logger.error("%d core memory slots failed integrity verification", len(failed))

        return failed

    def _compute_slot_hash(self, slot: int) -> bytes:
        """Compute SHA-256 hash for a single slot's key-value pair."""
        hasher = hashlib.sha256()
        hasher.update(self.keys.data[slot].detach().cpu().numpy().tobytes())
        hasher.update(self.values.data[slot].detach().cpu().numpy().tobytes())
        hasher.update(self.confidence[slot].cpu().numpy().tobytes())
        return hasher.digest()

    @property
    def n_active(self) -> int:
        """Number of active axiom slots."""
        return self._n_active

    @property
    def event_emitter(self) -> PromotionEventEmitter:
        """Access the promotion event emitter for handler registration."""
        return self._event_emitter
