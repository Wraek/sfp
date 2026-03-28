"""Persistent application settings stored in ``~/.sfp/settings.json``."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


def _settings_path() -> Path:
    """Return the settings file path, creating the parent dir if needed."""
    d = Path.home() / ".sfp"
    d.mkdir(parents=True, exist_ok=True)
    return d / "settings.json"


_DEFAULTS: dict[str, Any] = {
    "bridge": {
        "module_path": "",
        "entry_point": "bridge.bridge_server:BridgeServer",
    },
}


class Settings:
    """Read/write ``~/.sfp/settings.json`` with typed accessors."""

    def __init__(self) -> None:
        self._path = _settings_path()
        self._data: dict[str, Any] = copy.deepcopy(_DEFAULTS)
        self._load()

    def _load(self) -> None:
        """Load settings from disk, merging with defaults."""
        if self._path.exists():
            try:
                stored = json.loads(self._path.read_text(encoding="utf-8"))
                self._deep_merge(self._data, stored)
            except (json.JSONDecodeError, OSError):
                pass  # Corrupted or unreadable — use defaults

    def _save(self) -> None:
        """Persist current settings to disk."""
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _deep_merge(base: dict, overlay: dict) -> None:
        """Merge *overlay* into *base* in-place (one level deep)."""
        for key, value in overlay.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                base[key].update(value)
            else:
                base[key] = value

    # -- Bridge settings -------------------------------------------------

    @property
    def bridge_module_path(self) -> str:
        return self._data.get("bridge", {}).get("module_path", "")

    @bridge_module_path.setter
    def bridge_module_path(self, value: str) -> None:
        self._data.setdefault("bridge", {})["module_path"] = value
        self._save()

    @property
    def bridge_entry_point(self) -> str:
        return self._data.get("bridge", {}).get(
            "entry_point", "bridge.bridge_server:BridgeServer",
        )

    @bridge_entry_point.setter
    def bridge_entry_point(self, value: str) -> None:
        self._data.setdefault("bridge", {})["entry_point"] = value
        self._save()
