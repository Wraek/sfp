"""Defense Layer 2-3: Surprise-gate hardening with clamping, rate limiting, and dual-path verification."""

from __future__ import annotations

from collections import deque

import torch

from sfp.config import Tier0Config
from sfp.utils.logging import get_logger

logger = get_logger("defense.surprise")


class SurpriseHardener:
    """Hardens the surprise gate against inflation, suppression, and targeted manipulation.

    Implements three defense mechanisms:
      1. **Surprise ratio clamping** — bounds the max update magnitude per step.
      2. **Sliding-window rate limiting** — detects and throttles bursts of high-surprise events.
      3. **Dual-path surprise verification** — requires both gradient-magnitude and latent-space
         distance to agree before applying full-strength updates.

    Composed into StreamingProcessor to keep concerns separated.
    """

    def __init__(self, config: Tier0Config) -> None:
        self._config = config

        # Surprise EMA for ratio computation
        self._surprise_ema: float = 0.0
        self._step_count: int = 0

        # Per-parameter gradient EMA for adaptive clipping
        self._grad_ema: dict[str, torch.Tensor] = {}

        # Rate-limiting sliding window: stores 1 if high-surprise, 0 otherwise
        self._rate_window: deque[int] = deque(maxlen=config.rate_limit_window)
        self._rate_limited: bool = False
        self._cooldown_remaining: int = 0

    def compute_hardened_surprise(
        self,
        raw_grad_norm: float,
        latent_distance: float | None = None,
    ) -> float:
        """Compute a hardened surprise score from the raw gradient norm.

        Args:
            raw_grad_norm: Raw L2 gradient norm from the backward pass.
            latent_distance: Optional distance in Perceiver latent space to nearest attractor.
                If provided, dual-path verification is applied.

        Returns:
            Hardened surprise ratio in [0, max_surprise_ratio], possibly scaled down
            by rate limiting or dual-path disagreement.
        """
        cfg = self._config

        # Update surprise EMA
        self._surprise_ema = (
            cfg.surprise_momentum * self._surprise_ema
            + (1.0 - cfg.surprise_momentum) * raw_grad_norm
        )
        self._step_count += 1

        # Compute surprise ratio relative to EMA
        if self._surprise_ema > 1e-12:
            surprise_ratio = raw_grad_norm / self._surprise_ema
        else:
            surprise_ratio = 1.0

        # --- Defense 1: Clamp surprise ratio ---
        surprise_ratio = min(surprise_ratio, cfg.max_surprise_ratio)

        # --- Defense 2: Rate limiting ---
        # Decrement cooldown counter (post-consolidation suppression)
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        is_high = surprise_ratio > 1.5  # above 1.5x EMA counts as "high"
        self._rate_window.append(1 if is_high else 0)

        if len(self._rate_window) >= 10 and self._cooldown_remaining <= 0:
            high_rate = sum(self._rate_window) / len(self._rate_window)
            if high_rate > cfg.rate_limit_threshold:
                if not self._rate_limited:
                    logger.warning(
                        "Rate limiting activated: %.1f%% high-surprise in window",
                        high_rate * 100,
                    )
                self._rate_limited = True
                surprise_ratio *= 0.1  # Drastically reduce learning during burst
            else:
                if self._rate_limited:
                    logger.info("Rate limiting deactivated")
                self._rate_limited = False
        elif self._cooldown_remaining > 0 and self._rate_limited:
            self._rate_limited = False

        # --- Defense 3: Dual-path verification ---
        if latent_distance is not None and self._step_count > 10:
            # Both gradient surprise and latent distance must agree
            # Normalize latent_distance relative to a running baseline
            # If latent distance is low (input is near existing attractor) but gradient
            # surprise is high, scale down — likely adversarial
            latent_surprise = min(latent_distance, cfg.max_surprise_ratio)
            agreement = min(surprise_ratio, latent_surprise) / max(
                max(surprise_ratio, latent_surprise), 1e-8
            )
            surprise_ratio *= agreement

        return max(surprise_ratio, 0.0)

    def clip_gradients(self, named_parameters: list[tuple[str, torch.nn.Parameter]]) -> float:
        """Apply per-parameter adaptive gradient clipping (ARC-style).

        Maintains an EMA of gradient magnitudes per parameter and clips any gradient
        exceeding clip_multiplier * EMA.

        Args:
            named_parameters: List of (name, parameter) tuples from the model.

        Returns:
            Total clipping ratio (clipped_norm / original_norm). Values < 1.0 indicate clipping occurred.
        """
        cfg = self._config
        total_original_sq = 0.0
        total_clipped_sq = 0.0

        for name, param in named_parameters:
            if param.grad is None:
                continue

            grad = param.grad.data
            grad_mag = grad.abs()
            total_original_sq += grad.pow(2).sum().item()

            # Initialize or update per-parameter EMA
            if name not in self._grad_ema:
                self._grad_ema[name] = grad_mag.clone()
            else:
                ema = self._grad_ema[name]
                ema.mul_(cfg.surprise_momentum).add_(grad_mag, alpha=1.0 - cfg.surprise_momentum)

            # Clip: any gradient element exceeding clip_multiplier * EMA is clamped
            clip_bound = self._grad_ema[name] * cfg.clip_multiplier
            # Ensure minimum clip bound to avoid zero-clamping early on
            clip_bound = torch.clamp(clip_bound, min=1e-6)
            param.grad.data = torch.clamp(grad, min=-clip_bound, max=clip_bound)

            total_clipped_sq += param.grad.data.pow(2).sum().item()

        if total_original_sq < 1e-12:
            return 1.0

        return (total_clipped_sq / total_original_sq) ** 0.5

    def suppress_rate_limiting(self, n_steps: int) -> None:
        """Temporarily suppress rate limiting for n_steps.

        Call after consolidation to prevent replay bursts from triggering
        defensive throttling of subsequent normal learning.
        """
        self._cooldown_remaining = max(n_steps, 0)
        if self._rate_limited:
            self._rate_limited = False
            logger.info("Rate limiting suppressed for %d steps (post-consolidation)", n_steps)

    @property
    def is_rate_limited(self) -> bool:
        """Whether the system is currently in rate-limited defensive mode."""
        return self._rate_limited

    @property
    def surprise_ema(self) -> float:
        """Current surprise exponential moving average."""
        return self._surprise_ema

    def reset(self) -> None:
        """Reset all state (e.g. on session reset)."""
        self._surprise_ema = 0.0
        self._step_count = 0
        self._grad_ema.clear()
        self._rate_window.clear()
        self._rate_limited = False
        self._cooldown_remaining = 0
