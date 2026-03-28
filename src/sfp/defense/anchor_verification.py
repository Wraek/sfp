"""Defense Layer 5 extension: Anchor concept verification.

Maintains a set of anchor concepts with known embeddings and expected basin
assignments. Periodically verifies that these anchors haven't drifted from
their expected positions in the manifold.
"""

from __future__ import annotations

import torch

from sfp.memory.essential import EssentialMemory
from sfp.utils.logging import get_logger

logger = get_logger("defense.anchor")


class AnchorVerifier:
    """Verifies that anchor concepts remain in their expected manifold positions.

    Anchor exemplars are stored in a read-only, cryptographically signed partition.
    At each verification pass, we check:
      1. Each anchor is still assigned to its expected basin.
      2. The embedding distance to the expected basin center hasn't drifted.
      3. Pairwise distances between anchors maintain expected relationships.

    Args:
        anchor_exemplars: (N, d_model) tensor of anchor embeddings.
        expected_basins: (N,) tensor of expected basin IDs.
        drift_threshold: Maximum allowable cosine drift before alerting.
        pairwise_tolerance: Maximum relative change in pairwise distances.
    """

    def __init__(
        self,
        anchor_exemplars: torch.Tensor,
        expected_basins: torch.Tensor,
        drift_threshold: float = 0.3,
        pairwise_tolerance: float = 0.2,
    ) -> None:
        self._anchors = anchor_exemplars.clone()
        self._expected_basins = expected_basins.clone()
        self._drift_threshold = drift_threshold
        self._pairwise_tolerance = pairwise_tolerance

        # Store initial pairwise distances for comparison
        self._initial_pairwise: torch.Tensor | None = None
        if anchor_exemplars.shape[0] >= 2:
            self._initial_pairwise = torch.cdist(
                anchor_exemplars.float().unsqueeze(0),
                anchor_exemplars.float().unsqueeze(0),
            ).squeeze(0)

    def verify(self, tier2: EssentialMemory) -> list[str]:
        """Run anchor verification against the current Tier 2 state.

        Returns:
            List of violation messages. Empty list = all anchors verified OK.
        """
        violations: list[str] = []

        if tier2.n_active == 0:
            return ["No active basins — cannot verify anchors"]

        n_anchors = self._anchors.shape[0]
        device = tier2.keys.device

        current_basins: list[int] = []

        for i in range(n_anchors):
            anchor = self._anchors[i].to(device)
            expected_bid = self._expected_basins[i].item()

            # Retrieve from Tier 2
            with torch.no_grad():
                _, basin_id, attn = tier2.retrieve(anchor)

            actual_bid = basin_id.item() if basin_id.dim() == 0 else basin_id[0].item()
            current_basins.append(actual_bid)

            # Check 1: Basin assignment
            if actual_bid != expected_bid:
                violations.append(
                    f"Anchor {i}: basin shifted from {expected_bid} to {actual_bid}"
                )

            # Check 2: Drift from expected basin center
            if tier2.active_mask[actual_bid]:
                basin_key = tier2.keys[actual_bid]
                cos_sim = torch.nn.functional.cosine_similarity(
                    anchor.unsqueeze(0), basin_key.unsqueeze(0)
                ).item()
                drift = 1.0 - cos_sim

                if drift > self._drift_threshold:
                    violations.append(
                        f"Anchor {i}: excessive drift from basin {actual_bid} "
                        f"(cosine_drift={drift:.4f}, threshold={self._drift_threshold})"
                    )

        # Check 3: Pairwise distance stability
        if self._initial_pairwise is not None and n_anchors >= 2:
            current_pairwise = torch.cdist(
                self._anchors.float().to(device).unsqueeze(0),
                self._anchors.float().to(device).unsqueeze(0),
            ).squeeze(0)

            initial = self._initial_pairwise.to(device)
            relative_change = (current_pairwise - initial).abs() / (initial + 1e-8)

            # Check off-diagonal elements
            for i in range(n_anchors):
                for j in range(i + 1, n_anchors):
                    change = relative_change[i, j].item()
                    if change > self._pairwise_tolerance:
                        violations.append(
                            f"Anchor pair ({i}, {j}): pairwise distance changed by "
                            f"{change * 100:.1f}% (tolerance={self._pairwise_tolerance * 100:.1f}%)"
                        )

        if violations:
            logger.warning(
                "Anchor verification: %d violations detected", len(violations)
            )

        return violations

    @property
    def n_anchors(self) -> int:
        return self._anchors.shape[0]
