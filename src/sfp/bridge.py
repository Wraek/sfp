"""Bridge protocol and dynamic loader for external integrations.

Defines :class:`BridgeProtocol` — the contract that any bridge module must
implement — and :class:`BridgeLoader` which handles discovering, importing,
and validating bridge classes at runtime.

External bridge modules (e.g. sfpmc) are loaded dynamically so SFP has no
compile-time dependency on them.  The bridge module path is configurable
and persisted in ``~/.sfp/settings.json``.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from sfp.interface import SFPInterface
from sfp.utils.logging import get_logger

logger = get_logger("bridge")

_REQUIRED_METHODS = ("start", "stop", "status")
_REQUIRED_PROPERTIES = ("is_running",)


@runtime_checkable
class BridgeProtocol(Protocol):
    """Contract that a bridge class must satisfy.

    The SFP GUI will instantiate the bridge class, then call
    :meth:`start` with an :class:`SFPInterface` reference.  The bridge
    should start its own daemon thread(s) and return immediately.

    Example implementation::

        class BridgeServer:
            def start(self, interface: SFPInterface) -> None:
                self._interface = interface
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

            def stop(self) -> None:
                self._running = False
                self._thread.join(timeout=5.0)

            @property
            def is_running(self) -> bool:
                return self._thread.is_alive()

            def status(self) -> dict[str, Any]:
                return {"state": "listening", "address": "127.0.0.1:17700"}
    """

    def start(self, interface: SFPInterface) -> None:
        """Start the bridge, storing the interface reference.

        Must return promptly — long-running work should happen on
        daemon thread(s) started internally.
        """
        ...

    def stop(self) -> None:
        """Shut down the bridge gracefully.

        Should close connections, stop threads, and release resources.
        """
        ...

    @property
    def is_running(self) -> bool:
        """Whether the bridge is actively running."""
        ...

    def status(self) -> dict[str, Any]:
        """Return a status summary dict.

        Expected keys (at minimum): ``"state"`` (str).
        Additional keys are bridge-specific.
        """
        ...


class BridgeLoadError(Exception):
    """Raised when a bridge module cannot be loaded or validated."""


class BridgeLoader:
    """Discovers, imports, and validates bridge classes at runtime.

    Supports two discovery modes:

    1. **Direct import** — if the entry point module is already installed
       (e.g. via ``pip install sfpmc``), it is imported directly.
    2. **Path-based import** — if a ``module_path`` is provided and direct
       import fails, the path is added to ``sys.path`` and the import is
       retried.

    Args:
        module_path: Filesystem path to the bridge project root.
            Added to ``sys.path`` if direct import fails.
        entry_point: Dotted path to the bridge class, in the format
            ``"module.submodule:ClassName"`` (e.g.
            ``"bridge.bridge_server:BridgeServer"``).
    """

    def __init__(
        self,
        entry_point: str,
        module_path: str | None = None,
    ) -> None:
        self._entry_point = entry_point
        self._module_path = module_path

    def load(self) -> type:
        """Import and validate the bridge class.

        Returns:
            The bridge class (not an instance).

        Raises:
            BridgeLoadError: If import fails or the class does not
                satisfy :class:`BridgeProtocol`.
        """
        module_dotted, class_name = self._parse_entry_point()
        cls = self._import_class(module_dotted, class_name)
        self._validate(cls)
        return cls

    def _parse_entry_point(self) -> tuple[str, str]:
        """Parse ``"module.path:ClassName"`` into components."""
        if ":" not in self._entry_point:
            raise BridgeLoadError(
                f"Entry point must use 'module.path:ClassName' format, "
                f"got {self._entry_point!r}"
            )
        module_dotted, class_name = self._entry_point.rsplit(":", 1)
        if not module_dotted or not class_name:
            raise BridgeLoadError(
                f"Invalid entry point: {self._entry_point!r}"
            )
        return module_dotted, class_name

    def _import_class(self, module_dotted: str, class_name: str) -> type:
        """Try direct import, then path-based import."""
        # Attempt 1: direct import (works if pip-installed)
        try:
            mod = importlib.import_module(module_dotted)
            cls = getattr(mod, class_name)
            logger.info(
                "Loaded bridge %s via direct import", self._entry_point,
            )
            return cls
        except (ImportError, AttributeError):
            pass

        # Attempt 2: add module_path to sys.path and retry
        if self._module_path is None:
            raise BridgeLoadError(
                f"Cannot import {module_dotted!r} and no module_path configured"
            )

        resolved = Path(self._module_path).resolve()
        if not resolved.is_dir():
            raise BridgeLoadError(
                f"Module path does not exist or is not a directory: {resolved}"
            )

        path_str = str(resolved)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            logger.info("Added %s to sys.path", path_str)

        try:
            mod = importlib.import_module(module_dotted)
            cls = getattr(mod, class_name)
            logger.info(
                "Loaded bridge %s via path %s",
                self._entry_point,
                path_str,
            )
            return cls
        except ImportError as exc:
            raise BridgeLoadError(
                f"Failed to import {module_dotted!r} from {path_str}: {exc}"
            ) from exc
        except AttributeError as exc:
            raise BridgeLoadError(
                f"Module {module_dotted!r} has no class {class_name!r}: {exc}"
            ) from exc

    def _validate(self, cls: type) -> None:
        """Check that the class has the required methods and properties."""
        missing = []
        for name in _REQUIRED_METHODS:
            if not callable(getattr(cls, name, None)):
                missing.append(f"method {name}()")

        for name in _REQUIRED_PROPERTIES:
            # Check it exists on the class (property or attribute)
            if not hasattr(cls, name):
                missing.append(f"property {name}")

        if missing:
            raise BridgeLoadError(
                f"Bridge class {cls.__name__} missing: {', '.join(missing)}"
            )
