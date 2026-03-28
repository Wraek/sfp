"""Encoder registry for discovering and instantiating input encoders."""

from __future__ import annotations

from typing import Any

from sfp.exceptions import AdapterNotFoundError
from sfp.input.encoder import BaseInputEncoder
from sfp.utils.logging import get_logger

logger = get_logger("input.registry")


class EncoderRegistry:
    """Registry for input encoder backends.

    Provides name-based discovery and instantiation of encoders.
    Built-in encoders are registered at import time.
    """

    _encoders: dict[str, type] = {}

    @classmethod
    def register(cls, name: str, encoder_class: type) -> None:
        """Register an encoder class under a name.

        Args:
            name: Short identifier (e.g., "sentence-transformers").
            encoder_class: Class implementing BaseInputEncoder interface.
        """
        if not (
            hasattr(encoder_class, "output_dim")
            and hasattr(encoder_class, "encode")
        ):
            raise TypeError(
                f"{encoder_class.__name__} must have output_dim and encode"
            )
        cls._encoders[name] = encoder_class
        logger.debug("Registered encoder: %s -> %s", name, encoder_class.__name__)

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> BaseInputEncoder:
        """Instantiate a registered encoder by name.

        Args:
            name: Registered encoder name.
            **kwargs: Passed to the encoder constructor.

        Returns:
            An instantiated encoder.

        Raises:
            AdapterNotFoundError: If name is not registered.
        """
        if name not in cls._encoders:
            available = ", ".join(cls.list_available())
            raise AdapterNotFoundError(
                f"Unknown encoder '{name}'. Available: {available}"
            )
        return cls._encoders[name](**kwargs)

    @classmethod
    def list_available(cls) -> list[str]:
        """Return sorted list of registered encoder names."""
        return sorted(cls._encoders.keys())


# Register built-in encoders
def _register_builtins() -> None:
    from sfp.input.adapters import (
        CLIPAdapter,
        PrecomputedAdapter,
        SentenceTransformerAdapter,
    )
    from sfp.input.bytelevel import ByteLevelEncoder

    EncoderRegistry.register("sentence-transformers", SentenceTransformerAdapter)
    EncoderRegistry.register("clip", CLIPAdapter)
    EncoderRegistry.register("precomputed", PrecomputedAdapter)
    EncoderRegistry.register("byte-level", ByteLevelEncoder)


_register_builtins()
