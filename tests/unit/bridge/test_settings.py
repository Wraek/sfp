"""Tests for Settings persistence."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from sfp.gui.settings import Settings


@pytest.fixture()
def settings_path(tmp_path: Path) -> Path:
    """Return a temp settings file path."""
    return tmp_path / "settings.json"


@pytest.fixture()
def settings(settings_path: Path) -> Settings:
    """Create a Settings instance using a temp path."""
    with patch("sfp.gui.settings._settings_path", return_value=settings_path):
        return Settings()


class TestDefaults:
    """Verify default values when no file exists."""

    def test_bridge_module_path_default(self, settings: Settings) -> None:
        assert settings.bridge_module_path == ""

    def test_bridge_entry_point_default(self, settings: Settings) -> None:
        assert settings.bridge_entry_point == "bridge.bridge_server:BridgeServer"


class TestPersistence:
    """Settings are written to disk and survive reload."""

    def test_set_module_path(
        self, settings: Settings, settings_path: Path,
    ) -> None:
        settings.bridge_module_path = "/some/path"
        assert settings.bridge_module_path == "/some/path"
        # File was written
        assert settings_path.exists()
        data = json.loads(settings_path.read_text())
        assert data["bridge"]["module_path"] == "/some/path"

    def test_set_entry_point(
        self, settings: Settings, settings_path: Path,
    ) -> None:
        settings.bridge_entry_point = "my_mod:MyBridge"
        assert settings.bridge_entry_point == "my_mod:MyBridge"
        data = json.loads(settings_path.read_text())
        assert data["bridge"]["entry_point"] == "my_mod:MyBridge"

    def test_reload_preserves_values(self, settings_path: Path) -> None:
        # Write settings
        with patch("sfp.gui.settings._settings_path", return_value=settings_path):
            s1 = Settings()
            s1.bridge_module_path = "/my/bridge"
            s1.bridge_entry_point = "mod:Cls"

        # Reload
        with patch("sfp.gui.settings._settings_path", return_value=settings_path):
            s2 = Settings()
            assert s2.bridge_module_path == "/my/bridge"
            assert s2.bridge_entry_point == "mod:Cls"


class TestCorruptFile:
    """Graceful handling of corrupt settings files."""

    def test_corrupt_json_uses_defaults(self, settings_path: Path) -> None:
        settings_path.write_text("not json!!!")
        with patch("sfp.gui.settings._settings_path", return_value=settings_path):
            s = Settings()
        assert s.bridge_module_path == ""
        assert s.bridge_entry_point == "bridge.bridge_server:BridgeServer"

    def test_partial_json_merges(self, settings_path: Path) -> None:
        settings_path.write_text(json.dumps({"bridge": {"module_path": "/p"}}))
        with patch("sfp.gui.settings._settings_path", return_value=settings_path):
            s = Settings()
        assert s.bridge_module_path == "/p"
        # Entry point comes from defaults
        assert s.bridge_entry_point == "bridge.bridge_server:BridgeServer"
