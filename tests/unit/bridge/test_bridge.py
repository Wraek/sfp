"""Tests for BridgeProtocol and BridgeLoader."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from sfp.bridge import BridgeLoadError, BridgeLoader, BridgeProtocol
from sfp.interface import SFPInterface


# ── Mock bridge classes ─────────────────────────────────────────────


class _GoodBridge:
    """Satisfies BridgeProtocol."""

    def __init__(self) -> None:
        self._running = False
        self._interface: SFPInterface | None = None

    def start(self, interface: SFPInterface) -> None:
        self._interface = interface
        self._running = True

    def stop(self) -> None:
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def status(self) -> dict[str, Any]:
        return {"state": "testing"}


class _MissingStart:
    """Missing start method."""

    def stop(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {}


class _MissingIsRunning:
    """Missing is_running property."""

    def start(self, interface: SFPInterface) -> None:
        pass

    def stop(self) -> None:
        pass

    def status(self) -> dict[str, Any]:
        return {}


class _MissingStop:
    """Missing stop method."""

    def start(self, interface: SFPInterface) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return False

    def status(self) -> dict[str, Any]:
        return {}


class _MissingStatus:
    """Missing status method."""

    def start(self, interface: SFPInterface) -> None:
        pass

    def stop(self) -> None:
        pass

    @property
    def is_running(self) -> bool:
        return False


# ── Protocol tests ──────────────────────────────────────────────────


class TestBridgeProtocol:
    """Verify runtime_checkable protocol."""

    def test_good_bridge_is_protocol(self) -> None:
        assert isinstance(_GoodBridge(), BridgeProtocol)

    def test_importable_from_package(self) -> None:
        from sfp import BridgeProtocol as BP
        assert BP is BridgeProtocol

    def test_bridge_load_error_importable(self) -> None:
        from sfp import BridgeLoadError as BLE
        assert BLE is BridgeLoadError

    def test_bridge_loader_importable(self) -> None:
        from sfp import BridgeLoader as BL
        assert BL is BridgeLoader


# ── Loader: entry point parsing ─────────────────────────────────────


class TestEntryPointParsing:
    """BridgeLoader._parse_entry_point validation."""

    def test_valid_entry_point(self) -> None:
        loader = BridgeLoader("bridge.server:BridgeServer")
        mod, cls = loader._parse_entry_point()
        assert mod == "bridge.server"
        assert cls == "BridgeServer"

    def test_simple_entry_point(self) -> None:
        loader = BridgeLoader("mymod:MyClass")
        mod, cls = loader._parse_entry_point()
        assert mod == "mymod"
        assert cls == "MyClass"

    def test_no_colon_raises(self) -> None:
        loader = BridgeLoader("bridge.server.BridgeServer")
        with pytest.raises(BridgeLoadError, match="module.path:ClassName"):
            loader._parse_entry_point()

    def test_empty_module_raises(self) -> None:
        loader = BridgeLoader(":BridgeServer")
        with pytest.raises(BridgeLoadError, match="Invalid entry point"):
            loader._parse_entry_point()

    def test_empty_class_raises(self) -> None:
        loader = BridgeLoader("bridge.server:")
        with pytest.raises(BridgeLoadError, match="Invalid entry point"):
            loader._parse_entry_point()


# ── Loader: validation ──────────────────────────────────────────────


class TestValidation:
    """BridgeLoader._validate checks for required methods/properties."""

    def test_good_bridge_passes(self) -> None:
        loader = BridgeLoader("x:X")
        loader._validate(_GoodBridge)  # Should not raise

    def test_missing_start_raises(self) -> None:
        loader = BridgeLoader("x:X")
        with pytest.raises(BridgeLoadError, match="start"):
            loader._validate(_MissingStart)

    def test_missing_stop_raises(self) -> None:
        loader = BridgeLoader("x:X")
        with pytest.raises(BridgeLoadError, match="stop"):
            loader._validate(_MissingStop)

    def test_missing_status_raises(self) -> None:
        loader = BridgeLoader("x:X")
        with pytest.raises(BridgeLoadError, match="status"):
            loader._validate(_MissingStatus)

    def test_missing_is_running_raises(self) -> None:
        loader = BridgeLoader("x:X")
        with pytest.raises(BridgeLoadError, match="is_running"):
            loader._validate(_MissingIsRunning)


# ── Loader: dynamic import ──────────────────────────────────────────


class TestDynamicImport:
    """BridgeLoader.load() with real filesystem modules."""

    def test_load_from_path(self, tmp_path: Path) -> None:
        """Write a bridge module to a temp dir and load it by path."""
        pkg = tmp_path / "testbridge"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "server.py").write_text(
            textwrap.dedent("""\
                class TestBridge:
                    def start(self, interface):
                        pass
                    def stop(self):
                        pass
                    @property
                    def is_running(self):
                        return False
                    def status(self):
                        return {"state": "test"}
            """)
        )

        loader = BridgeLoader(
            "testbridge.server:TestBridge",
            module_path=str(tmp_path),
        )
        cls = loader.load()
        assert cls.__name__ == "TestBridge"

        # Clean up sys.path and modules
        sys.path.remove(str(tmp_path))
        sys.modules.pop("testbridge.server", None)
        sys.modules.pop("testbridge", None)

    def test_no_path_and_not_installed_raises(self) -> None:
        loader = BridgeLoader("nonexistent.module:Foo")
        with pytest.raises(BridgeLoadError, match="no module_path configured"):
            loader.load()

    def test_bad_path_raises(self, tmp_path: Path) -> None:
        loader = BridgeLoader(
            "nonexistent.module:Foo",
            module_path=str(tmp_path / "does_not_exist"),
        )
        with pytest.raises(BridgeLoadError, match="does not exist"):
            loader.load()

    def test_import_error_raises(self, tmp_path: Path) -> None:
        """Module path exists but module doesn't."""
        loader = BridgeLoader(
            "missing_module:Foo",
            module_path=str(tmp_path),
        )
        with pytest.raises(BridgeLoadError, match="Failed to import"):
            loader.load()

    def test_missing_class_in_module_raises(self, tmp_path: Path) -> None:
        """Module exists but class doesn't."""
        (tmp_path / "has_module.py").write_text("class Other: pass\n")
        loader = BridgeLoader(
            "has_module:Missing",
            module_path=str(tmp_path),
        )
        with pytest.raises(BridgeLoadError, match="no class"):
            loader.load()

        sys.path.remove(str(tmp_path))
        sys.modules.pop("has_module", None)

    def test_validation_failure_after_import(self, tmp_path: Path) -> None:
        """Module loads but class is missing required methods."""
        (tmp_path / "bad_bridge.py").write_text(
            textwrap.dedent("""\
                class Incomplete:
                    def start(self, interface):
                        pass
            """)
        )
        loader = BridgeLoader(
            "bad_bridge:Incomplete",
            module_path=str(tmp_path),
        )
        with pytest.raises(BridgeLoadError, match="missing"):
            loader.load()

        sys.path.remove(str(tmp_path))
        sys.modules.pop("bad_bridge", None)
