"""Structured logging setup for SFP."""

from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the sfp namespace.

    Usage:
        logger = get_logger("core.streaming")
        # produces logger named "sfp.core.streaming"
    """
    return logging.getLogger(f"sfp.{name}")
