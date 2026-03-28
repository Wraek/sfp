"""Scrollable log panel for the SFP GUI."""

from __future__ import annotations

import logging
import time
import tkinter as tk
from tkinter import ttk


class LogPanel(ttk.LabelFrame):
    """Read-only scrollable log display with color-coded severity."""

    def __init__(self, parent: tk.Widget, **kwargs: object) -> None:
        super().__init__(parent, text="Log Output", **kwargs)

        self._text = tk.Text(
            self,
            height=12,
            state="disabled",
            wrap="word",
            font=("Consolas", 9),
            bg="#1e1e1e",
            fg="#d4d4d4",
            insertbackground="#d4d4d4",
        )
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        self._text.configure(yscrollcommand=scrollbar.set)

        self._text.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scrollbar.pack(side="right", fill="y", pady=4, padx=(0, 4))

        # Color tags
        self._text.tag_configure("info", foreground="#d4d4d4")
        self._text.tag_configure("warning", foreground="#e5c07b")
        self._text.tag_configure("error", foreground="#e06c75")
        self._text.tag_configure("success", foreground="#98c379")
        self._text.tag_configure("timestamp", foreground="#5c6370")

    def log(self, message: str, level: str = "info") -> None:
        """Append a timestamped log entry."""
        ts = time.strftime("%H:%M:%S")
        self._text.configure(state="normal")
        self._text.insert("end", f"[{ts}] ", "timestamp")
        self._text.insert("end", f"{message}\n", level)
        self._text.see("end")
        self._text.configure(state="disabled")

    def clear(self) -> None:
        """Clear all log entries."""
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


class GUILogHandler(logging.Handler):
    """Logging handler that routes SFP log records to a LogPanel.

    Must be called from the main thread (via root.after) since tkinter
    is not thread-safe.
    """

    def __init__(self, log_panel: LogPanel, root: tk.Tk) -> None:
        super().__init__()
        self._panel = log_panel
        self._root = root

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        level = "info"
        if record.levelno >= logging.ERROR:
            level = "error"
        elif record.levelno >= logging.WARNING:
            level = "warning"

        # Schedule on main thread
        try:
            self._root.after(0, self._panel.log, msg, level)
        except RuntimeError:
            pass  # Window already destroyed
