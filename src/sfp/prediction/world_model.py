"""Predictive World Model — RSSM-based next-state prediction and imagination.

Implements a Recurrent State-Space Model (DreamerV3-inspired) that maintains
evolving latent state and predicts future inputs.  Enables anticipatory
retrieval (pre-activation), planning via imagination, and richer surprise
signals composed of prediction error, KL divergence, and reconstruction error.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sfp.config import WorldModelConfig
from sfp.types import WorldModelState
from sfp.utils.logging import get_logger

logger = get_logger("prediction.world_model")


class PredictiveWorldModel(nn.Module):
    """RSSM world model with pre-activation cache and imagination.

    Architecture:
      - GRU cell for deterministic path
      - Categorical stochastic path (32 categories × 32 classes)
      - Prior network (predicts next stochastic from deterministic)
      - Posterior network (corrects prior with observation)
      - Decoder (reconstructs observation from latent state)
      - Reward head + continue head
      - 8 subspace projections for directional prediction error
      - Pre-activation cache (8 entries, cosine > 0.8 match)

    Args:
        config: WorldModelConfig with RSSM dimensions and loss weights.
        d_model: Observation dimensionality (should match backbone output).
    """

    def __init__(
        self,
        config: WorldModelConfig | None = None,
        d_model: int = 512,
    ) -> None:
        super().__init__()
        cfg = config or WorldModelConfig()
        self._config = cfg

        d_det = cfg.d_deterministic
        d_obs = cfg.d_observation if cfg.d_observation != 512 else d_model
        n_cat = cfg.d_stochastic_categories
        n_cls = cfg.d_stochastic_classes
        d_stoch_flat = n_cat * n_cls  # 1024 by default
        d_latent = d_det + d_stoch_flat

        self._d_det = d_det
        self._d_obs = d_obs
        self._n_categories = n_cat
        self._n_classes = n_cls
        self._d_stoch_flat = d_stoch_flat

        # --- Deterministic path: GRU cell ---
        self.gru = nn.GRUCell(d_stoch_flat + d_obs, d_det)

        # --- Stochastic path ---
        # Prior: predicts next stochastic from deterministic alone
        self.prior_net = nn.Sequential(
            nn.Linear(d_det, d_det),
            nn.ELU(),
            nn.Linear(d_det, n_cat * n_cls),
        )

        # Posterior: corrects prior using actual observation
        self.posterior_net = nn.Sequential(
            nn.Linear(d_det + d_obs, d_det),
            nn.ELU(),
            nn.Linear(d_det, n_cat * n_cls),
        )

        # --- Decoder: reconstructs observation from full latent ---
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_det),
            nn.ELU(),
            nn.Linear(d_det, d_obs),
        )

        # --- Auxiliary heads ---
        self.reward_head = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.ELU(),
            nn.Linear(256, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(d_latent, 256),
            nn.ELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

        # --- Directional prediction error projections ---
        # 8 subspaces (spatial, temporal, semantic, relational, magnitude,
        # identity, modality, context)
        n_sub = cfg.n_subspace_projections
        d_sub = d_obs // n_sub
        self._d_sub = d_sub
        self.subspace_projections = nn.Parameter(
            torch.randn(n_sub, d_obs, d_sub) * 0.02
        )

        # --- Spatial prediction head ---
        # Predicts 3D position delta from latent state (generic, any bridge)
        self.spatial_predictor = nn.Sequential(
            nn.Linear(d_latent, 128),
            nn.ELU(),
            nn.Linear(128, 3),  # (dx, dy, dz)
        )
        self._prev_spatial_position: torch.Tensor | None = None
        self._spatial_loss_ema: float = 1.0

        # --- Pre-activation cache ---
        cache_size = cfg.cache_size
        self.register_buffer("cache_keys", torch.zeros(cache_size, d_obs))
        self.register_buffer("cache_values", torch.zeros(cache_size, d_obs))
        self.register_buffer(
            "cache_valid", torch.zeros(cache_size, dtype=torch.bool)
        )
        self._cache_idx: int = 0

        # --- RSSM state ---
        self.register_buffer("_h", torch.zeros(d_det))
        self.register_buffer("_z", torch.zeros(d_stoch_flat))

        # --- Running normalizers for enhanced surprise ---
        self._pred_error_ema: float = 1.0
        self._kl_ema: float = 1.0
        self._recon_error_ema: float = 1.0
        self._ema_momentum: float = 0.99

        # --- Internal optimizer (trains every inference step) ---
        self._optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)

        logger.info(
            "PredictiveWorldModel initialized: d_det=%d, d_obs=%d, "
            "stoch=%dx%d=%d, cache=%d",
            d_det, d_obs, n_cat, n_cls, d_stoch_flat, cache_size,
        )

    # ------------------------------------------------------------------
    # Core RSSM step
    # ------------------------------------------------------------------

    def step(self, observation: torch.Tensor) -> WorldModelState:
        """Advance the RSSM by one step given a new observation.

        Args:
            observation: (d_obs,) input embedding from backbone.

        Returns:
            WorldModelState with deterministic/stochastic state and losses.
        """
        obs = observation.detach()
        device = obs.device

        # Ensure state is on the right device
        if self._h.device != device:
            self._h = self._h.to(device)
            self._z = self._z.to(device)

        # 1. GRU deterministic update
        gru_input = torch.cat([self._z, obs])  # (d_stoch + d_obs,)
        h_new = self.gru(gru_input.unsqueeze(0), self._h.unsqueeze(0))
        h_new = h_new.squeeze(0)  # (d_det,)

        # 2. Prior: predict stochastic from deterministic alone
        prior_logits = self.prior_net(h_new)  # (n_cat * n_cls,)
        prior_logits = prior_logits.view(self._n_categories, self._n_classes)

        # 3. Posterior: correct using observation
        posterior_input = torch.cat([h_new, obs])
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(
            self._n_categories, self._n_classes
        )

        # 4. Sample from posterior (straight-through Gumbel-Softmax)
        z_new = self._sample_stochastic(posterior_logits)  # (n_cat, n_cls)
        z_flat = z_new.reshape(-1)  # (d_stoch_flat,)

        # 5. Decode: reconstruct expected observation
        latent = torch.cat([h_new, z_flat])  # (d_det + d_stoch,)
        predicted_obs = self.decoder(latent)  # (d_obs,)

        # 6. Compute losses
        pred_error = (predicted_obs - obs).norm().item()
        kl = self._kl_divergence(posterior_logits, prior_logits).item()
        recon_error = self._symlog_mse(predicted_obs, obs).item()

        # 7. Update state
        self._h = h_new.detach()
        self._z = z_flat.detach()

        # 8. Cache prediction for pre-activation
        self._write_cache(predicted_obs.detach())

        return WorldModelState(
            deterministic=h_new.detach(),
            stochastic=z_flat.detach(),
            prediction_error=pred_error,
            kl_divergence=kl,
            reconstruction_error=recon_error,
        )

    def train_step(
        self,
        observation: torch.Tensor,
        *,
        spatial_position: tuple[float, float, float] | None = None,
    ) -> dict[str, float]:
        """Advance one step and backprop the world model loss.

        Designed to run every inference step (~0.5 ms).

        Args:
            observation: (d_obs,) input embedding from backbone.
            spatial_position: Optional (x, y, z) world position from bridge.
                When provided, trains the spatial prediction head on
                position deltas between consecutive steps.

        Returns:
            Dict of loss component values.
        """
        obs = observation.detach().requires_grad_(False)
        device = obs.device

        if self._h.device != device:
            self._h = self._h.to(device)
            self._z = self._z.to(device)

        # --- Forward (with gradients for training) ---
        gru_input = torch.cat([self._z, obs])
        h_new = self.gru(gru_input.unsqueeze(0), self._h.unsqueeze(0)).squeeze(0)

        prior_logits = self.prior_net(h_new).view(
            self._n_categories, self._n_classes
        )
        posterior_input = torch.cat([h_new, obs])
        posterior_logits = self.posterior_net(posterior_input).view(
            self._n_categories, self._n_classes
        )

        z_new = self._sample_stochastic(posterior_logits)
        z_flat = z_new.reshape(-1)

        latent = torch.cat([h_new, z_flat])
        predicted_obs = self.decoder(latent)

        # --- Losses ---
        kl_loss = self._kl_divergence(posterior_logits, prior_logits)
        recon_loss = self._symlog_mse(predicted_obs, obs)
        pred_error = (predicted_obs - obs).norm()

        cfg = self._config
        total_loss = cfg.kl_weight * kl_loss + cfg.reconstruction_weight * recon_loss

        # --- Spatial prediction loss (when position data is available) ---
        spatial_loss_val = 0.0
        if spatial_position is not None and self._prev_spatial_position is not None:
            curr_pos = torch.tensor(
                spatial_position, dtype=torch.float32, device=device,
            )
            prev_pos = self._prev_spatial_position.to(device)
            actual_delta = (curr_pos - prev_pos) / 100.0  # normalize
            predicted_delta = self.spatial_predictor(latent)
            spatial_loss = F.mse_loss(predicted_delta, actual_delta)
            total_loss = total_loss + 0.1 * spatial_loss
            spatial_loss_val = spatial_loss.item()
            self._spatial_loss_ema = (
                0.99 * self._spatial_loss_ema + 0.01 * spatial_loss_val
            )

        if spatial_position is not None:
            self._prev_spatial_position = torch.tensor(
                spatial_position, dtype=torch.float32, device=device,
            )

        # --- Backprop ---
        self._optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self._config.grad_clip_norm)
        self._optimizer.step()

        # --- Update state (detached) ---
        self._h = h_new.detach()
        self._z = z_flat.detach()

        # --- Cache prediction ---
        self._write_cache(predicted_obs.detach())

        losses = {
            "total_loss": total_loss.item(),
            "kl_divergence": kl_loss.item(),
            "reconstruction_error": recon_loss.item(),
            "prediction_error": pred_error.item(),
        }
        if spatial_loss_val > 0.0:
            losses["spatial_prediction_loss"] = spatial_loss_val

        # Add continue probability for episode boundary detection
        with torch.no_grad():
            detached_latent = torch.cat([self._h, self._z])
            losses["continue_probability"] = self.continue_head(
                detached_latent,
            ).item()

        return losses

    # ------------------------------------------------------------------
    # Spatial prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_spatial_delta(self) -> torch.Tensor | None:
        """Predict the next position delta from the current latent state.

        Returns:
            (3,) tensor of predicted (dx, dy, dz) in normalized units,
            or None if the model has no state yet.
        """
        if self._h.norm().item() < 1e-8:
            return None
        latent = torch.cat([self._h, self._z])
        return self.spatial_predictor(latent) * 100.0  # denormalize

    @property
    def spatial_loss(self) -> float:
        """EMA of spatial prediction loss (for monitoring)."""
        return self._spatial_loss_ema

    @torch.no_grad()
    def predict_continue_probability(self) -> float:
        """Predict the probability that the current episode continues.

        Returns:
            Float in [0, 1]. Low values indicate an episode boundary.
            Returns 1.0 if the model has no state yet.
        """
        if self._h.norm().item() < 1e-8:
            return 1.0
        latent = torch.cat([self._h, self._z])
        return self.continue_head(latent).item()

    @torch.no_grad()
    def compute_spatial_prediction_error(
        self,
        actual_position: tuple[float, float, float],
    ) -> float | None:
        """Compute error between predicted and actual spatial position delta.

        Args:
            actual_position: Current (x, y, z) world position.

        Returns:
            L2 norm of the prediction error, or None if no previous position
            or model has no state.
        """
        if self._prev_spatial_position is None:
            return None
        if self._h.norm().item() < 1e-8:
            return None

        device = self._h.device
        curr_pos = torch.tensor(
            actual_position, dtype=torch.float32, device=device,
        )
        prev_pos = self._prev_spatial_position.to(device)
        actual_delta = (curr_pos - prev_pos) / 100.0  # same normalization as training

        latent = torch.cat([self._h, self._z])
        predicted_delta = self.spatial_predictor(latent)

        return (predicted_delta - actual_delta).norm().item()

    # ------------------------------------------------------------------
    # Enhanced surprise
    # ------------------------------------------------------------------

    def compute_enhanced_surprise(self, state: WorldModelState) -> float:
        """Compute enhanced surprise as a weighted, normalized combination.

        Each component is normalized by its running EMA to produce a
        dimensionless surprise signal.

        Args:
            state: WorldModelState from the most recent step.

        Returns:
            Float surprise value (higher = more unexpected).
        """
        cfg = self._config
        mom = self._ema_momentum

        # Update running EMAs
        self._pred_error_ema = mom * self._pred_error_ema + (1 - mom) * max(state.prediction_error, 1e-8)
        self._kl_ema = mom * self._kl_ema + (1 - mom) * max(state.kl_divergence, 1e-8)
        self._recon_error_ema = mom * self._recon_error_ema + (1 - mom) * max(state.reconstruction_error, 1e-8)

        # Normalize each by its EMA
        norm_pred = state.prediction_error / max(self._pred_error_ema, 1e-8)
        norm_kl = state.kl_divergence / max(self._kl_ema, 1e-8)
        norm_recon = state.reconstruction_error / max(self._recon_error_ema, 1e-8)

        surprise = (
            cfg.prediction_error_weight * norm_pred
            + cfg.kl_divergence_weight * norm_kl
            + cfg.reconstruction_error_weight * norm_recon
        )
        return surprise

    # ------------------------------------------------------------------
    # Directional prediction error
    # ------------------------------------------------------------------

    def compute_directional_prediction_error(
        self, prediction: torch.Tensor, observation: torch.Tensor
    ) -> torch.Tensor:
        """Decompose prediction error into 8 learned subspace components.

        Args:
            prediction: (d_obs,) predicted observation.
            observation: (d_obs,) actual observation.

        Returns:
            (n_subspace_projections,) tensor of per-subspace error norms.
        """
        error = observation - prediction  # (d_obs,)
        # Project into each subspace: error @ projection[i] -> (d_sub,)
        # subspace_projections: (n_sub, d_obs, d_sub)
        projected = torch.einsum("d,nds->ns", error, self.subspace_projections)
        # projected: (n_sub, d_sub) — compute norm per subspace
        return projected.norm(dim=-1)  # (n_sub,)

    # ------------------------------------------------------------------
    # Pre-activation cache
    # ------------------------------------------------------------------

    def check_cache(
        self, query: torch.Tensor
    ) -> tuple[torch.Tensor | None, float]:
        """Check if a query matches a cached prediction.

        Args:
            query: (d_obs,) embedding to check against cache.

        Returns:
            (cached_value, similarity) if match found, else (None, 0.0).
        """
        if not self.cache_valid.any():
            return None, 0.0

        valid_mask = self.cache_valid
        valid_keys = self.cache_keys[valid_mask]  # (V, d_obs)
        valid_values = self.cache_values[valid_mask]  # (V, d_obs)

        # Cosine similarity
        query_norm = query / (query.norm() + 1e-8)
        key_norms = valid_keys / (valid_keys.norm(dim=-1, keepdim=True) + 1e-8)
        similarities = (query_norm.unsqueeze(0) @ key_norms.T).squeeze(0)  # (V,)

        best_sim, best_idx = similarities.max(dim=0)
        if best_sim.item() > self._config.cache_match_threshold:
            return valid_values[best_idx], best_sim.item()
        return None, 0.0

    def _write_cache(self, prediction: torch.Tensor) -> None:
        """Write a prediction into the cache ring buffer."""
        idx = self._cache_idx % self._config.cache_size
        self.cache_keys[idx] = prediction
        self.cache_values[idx] = prediction  # value = predicted observation
        self.cache_valid[idx] = True
        self._cache_idx += 1

    # ------------------------------------------------------------------
    # Imagination / planning
    # ------------------------------------------------------------------

    def imagine_trajectory(
        self, initial_state: WorldModelState, n_steps: int = 4
    ) -> list[WorldModelState]:
        """Imagine a trajectory using prior-only rollout (no observations).

        Args:
            initial_state: Starting RSSM state.
            n_steps: Number of imagination steps.

        Returns:
            List of imagined WorldModelState objects.
        """
        trajectory: list[WorldModelState] = []
        h = initial_state.deterministic.clone()
        z = initial_state.stochastic.clone()

        with torch.no_grad():
            for _ in range(n_steps):
                # Use prior only (no observation correction)
                prior_logits = self.prior_net(h).view(
                    self._n_categories, self._n_classes
                )
                z_sampled = self._sample_stochastic(prior_logits)
                z = z_sampled.reshape(-1)

                latent = torch.cat([h, z])
                predicted_obs = self.decoder(latent)
                reward = self.reward_head(latent).item()
                cont = self.continue_head(latent).item()

                # Advance deterministic state (use predicted obs as input)
                gru_input = torch.cat([z, predicted_obs])
                h = self.gru(gru_input.unsqueeze(0), h.unsqueeze(0)).squeeze(0)

                trajectory.append(
                    WorldModelState(
                        deterministic=h.clone(),
                        stochastic=z.clone(),
                        prediction_error=0.0,  # no ground truth
                        kl_divergence=0.0,
                        reconstruction_error=0.0,
                    )
                )

        return trajectory

    def project_multi_step(
        self, observation: torch.Tensor, n_steps: int | None = None
    ) -> list[torch.Tensor]:
        """Step with observation, then imagine forward.

        Args:
            observation: (d_obs,) current observation.
            n_steps: Projection horizon (defaults to config).

        Returns:
            List of decoded predicted observations.
        """
        n = n_steps or self._config.n_projection_steps
        state = self.step(observation)
        trajectory = self.imagine_trajectory(state, n)

        predictions: list[torch.Tensor] = []
        with torch.no_grad():
            for s in trajectory:
                latent = torch.cat([s.deterministic, s.stochastic])
                predictions.append(self.decoder(latent))
        return predictions

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset RSSM state and invalidate pre-activation cache."""
        self._h.zero_()
        self._z.zero_()
        self.cache_valid.fill_(False)
        self._cache_idx = 0
        self._pred_error_ema = 1.0
        self._kl_ema = 1.0
        self._recon_error_ema = 1.0
        self._prev_spatial_position = None
        self._spatial_loss_ema = 1.0
        logger.debug("World model state reset")

    @property
    def current_state(self) -> WorldModelState:
        """Return the current RSSM state (read-only snapshot)."""
        return WorldModelState(
            deterministic=self._h.detach().clone(),
            stochastic=self._z.detach().clone(),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_stochastic(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample from categorical distribution with straight-through gradient.

        Args:
            logits: (n_categories, n_classes) unnormalized logits.

        Returns:
            (n_categories, n_classes) one-hot samples with soft gradients.
        """
        probs = torch.softmax(logits, dim=-1)  # (n_cat, n_cls)
        # Sample categorical indices
        dist = torch.distributions.Categorical(probs=probs)
        indices = dist.sample()  # (n_cat,)
        one_hot = F.one_hot(indices, self._n_classes).float()  # (n_cat, n_cls)
        # Straight-through: one_hot in forward, probs in backward
        return (one_hot - probs).detach() + probs

    def _kl_divergence(
        self,
        posterior_logits: torch.Tensor,
        prior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between posterior and prior with free nats.

        Args:
            posterior_logits: (n_categories, n_classes) posterior logits.
            prior_logits: (n_categories, n_classes) prior logits.

        Returns:
            Scalar KL divergence (clamped by free nats per category).
        """
        posterior = torch.softmax(posterior_logits, dim=-1)
        prior = torch.softmax(prior_logits, dim=-1)

        # Per-category KL
        kl_per_cat = (
            posterior * (torch.log(posterior + 1e-8) - torch.log(prior + 1e-8))
        ).sum(dim=-1)  # (n_categories,)

        # Free nats: clamp minimum per category
        free_nats_per_cat = self._config.kl_free_nats / self._n_categories
        kl_clamped = torch.clamp(kl_per_cat, min=free_nats_per_cat)

        return kl_clamped.sum()

    def _symlog(self, x: torch.Tensor) -> torch.Tensor:
        """Symlog transform: sign(x) * log(|x| + 1)."""
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

    def _symlog_mse(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Scale-invariant MSE loss in symlog space."""
        return F.mse_loss(self._symlog(pred), self._symlog(target))
