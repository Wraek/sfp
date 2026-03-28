"""Session manager — handles session lifecycle, file paths, and auto-save recovery."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def _sessions_dir() -> Path:
    """Return the sessions directory, creating it if needed."""
    d = Path.home() / ".sfp" / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


class SessionManager:
    """Manages session directories and metadata files."""

    def __init__(self) -> None:
        self._base = _sessions_dir()
        self._current_dir: Path | None = None
        self._current_name: str | None = None

    @property
    def current_name(self) -> str | None:
        return self._current_name

    @property
    def checkpoint_path(self) -> Path | None:
        if self._current_dir is None:
            return None
        return self._current_dir / "checkpoint.pt"

    @property
    def auto_save_path(self) -> Path | None:
        if self._current_dir is None:
            return None
        return self._current_dir / "autosave.pt"

    def new_session(
        self,
        name: str,
        preset: str,
        create_kwargs: dict[str, Any],
    ) -> Path:
        """Create a new session directory with metadata.

        Returns the session directory path.
        """
        # Sanitize name for filesystem
        safe_name = "".join(c if c.isalnum() or c in "-_ " else "_" for c in name).strip()
        if not safe_name:
            safe_name = f"session_{int(time.time())}"

        session_dir = self._base / safe_name
        session_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "name": name,
            "preset": preset,
            "create_kwargs": create_kwargs,
            "created": time.time(),
            "last_saved": None,
        }
        (session_dir / "session.json").write_text(json.dumps(meta, indent=2))

        self._current_dir = session_dir
        self._current_name = name
        return session_dir

    def open_session(self, session_dir: Path) -> dict[str, Any]:
        """Open an existing session and return its metadata."""
        meta_path = session_dir / "session.json"
        if not meta_path.exists():
            msg = f"No session.json in {session_dir}"
            raise FileNotFoundError(msg)
        meta = json.loads(meta_path.read_text())
        self._current_dir = session_dir
        self._current_name = meta.get("name", session_dir.name)
        return meta

    def mark_saved(self) -> None:
        """Update the last_saved timestamp in session metadata."""
        if self._current_dir is None:
            return
        meta_path = self._current_dir / "session.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            meta["last_saved"] = time.time()
            meta_path.write_text(json.dumps(meta, indent=2))

    def list_sessions(self) -> list[dict[str, Any]]:
        """Return metadata for all saved sessions."""
        sessions = []
        for d in sorted(self._base.iterdir()):
            if not d.is_dir():
                continue
            meta_path = d / "session.json"
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text())
                    meta["path"] = str(d)
                    meta["has_checkpoint"] = (d / "checkpoint.pt").exists()
                    meta["has_autosave"] = (d / "autosave.pt").exists()
                    sessions.append(meta)
                except (json.JSONDecodeError, OSError):
                    continue
        return sessions

    def check_recovery(self) -> Path | None:
        """Check if any session has an auto-save newer than its last explicit save.

        Returns the auto-save path if recovery is available, else None.
        """
        for d in self._base.iterdir():
            if not d.is_dir():
                continue
            autosave = d / "autosave.pt"
            if not autosave.exists():
                continue
            meta_path = d / "session.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                last_saved = meta.get("last_saved")
                autosave_time = autosave.stat().st_mtime
                if last_saved is None or autosave_time > last_saved:
                    return autosave
            except (json.JSONDecodeError, OSError):
                continue
        return None
