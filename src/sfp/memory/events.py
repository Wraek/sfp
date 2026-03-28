"""Event system for Tier 3 promotion authorization.

Implements an event-driven authorization model where promotion candidates
generate PromotionRequest events. External code registers handlers that
receive requests and return approval/denial decisions.
"""

from __future__ import annotations

from collections.abc import Callable

from sfp.types import AuthorizationHandler, PromotionRequest
from sfp.utils.logging import get_logger

logger = get_logger("memory.events")


class PromotionEventEmitter:
    """Manages Tier 3 promotion authorization via an event system.

    Handlers are called in registration order. The first handler to return
    a definitive answer (True or False) determines the outcome. If no handler
    is registered, the default policy applies.

    Default policy: auto-approve if all criteria are met (configurable).
    """

    def __init__(self, default_approve: bool = False) -> None:
        self._handlers: list[Callable[[PromotionRequest], bool | None]] = []
        self._default_approve = default_approve

    def register(self, handler: Callable[[PromotionRequest], bool | None]) -> None:
        """Register an authorization handler.

        Handlers receive a PromotionRequest and should return:
          - True to approve
          - False to deny
          - None to defer to the next handler
        """
        self._handlers.append(handler)

    def unregister(self, handler: Callable[[PromotionRequest], bool | None]) -> None:
        """Remove a previously registered handler."""
        self._handlers.remove(handler)

    def emit(self, request: PromotionRequest) -> bool:
        """Emit a promotion request and collect the authorization decision.

        Args:
            request: The promotion request containing candidate information.

        Returns:
            True if approved, False if denied.
        """
        for handler in self._handlers:
            result = handler(request)
            if result is True:
                logger.info(
                    "Promotion approved for basin %d (confidence=%.3f, episodes=%d)",
                    request.basin_id,
                    request.confidence,
                    request.episode_count,
                )
                return True
            if result is False:
                logger.info(
                    "Promotion denied for basin %d by handler",
                    request.basin_id,
                )
                return False
            # None = defer to next handler

        # No handler gave a definitive answer — use default policy
        if self._default_approve:
            logger.info(
                "Promotion auto-approved for basin %d (no handler objected)",
                request.basin_id,
            )
            return True

        logger.info(
            "Promotion denied for basin %d (no handler approved, default=deny)",
            request.basin_id,
        )
        return False

    @property
    def handler_count(self) -> int:
        return len(self._handlers)


class CriteriaAuthorizationHandler:
    """Default handler that approves promotions meeting all criteria.

    This handler checks the promotion request against the Tier 3 config
    requirements and approves if all are met. It can be used as a baseline
    handler that external code can override by registering higher-priority handlers.
    """

    def __init__(
        self,
        min_confidence: float = 0.9,
        min_episode_count: int = 1000,
        min_modalities: int = 2,
        min_age_days: float = 7.0,
    ) -> None:
        self._min_confidence = min_confidence
        self._min_episode_count = min_episode_count
        self._min_modalities = min_modalities
        self._min_age_days = min_age_days

    def __call__(self, request: PromotionRequest) -> bool | None:
        """Evaluate a promotion request against criteria.

        Returns True if all criteria are met, None otherwise (deferring to the
        next handler rather than actively denying).
        """
        if request.confidence < self._min_confidence:
            return None
        if request.episode_count < self._min_episode_count:
            return None
        if request.modality_count < self._min_modalities:
            return None
        if request.age_days < self._min_age_days:
            return None
        return True
