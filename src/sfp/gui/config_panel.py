"""Advanced configuration dialog for the SFP GUI.

Auto-generates form fields from frozen config dataclasses using
dataclasses.fields() introspection.
"""

from __future__ import annotations

import dataclasses
import tkinter as tk
from tkinter import ttk
from typing import Any, get_type_hints


class ConfigDialog(tk.Toplevel):
    """Tabbed dialog for editing advanced SFP configuration parameters."""

    def __init__(
        self,
        parent: tk.Widget,
        categories: dict[str, list[type]],
        overrides: dict[str, dict[str, Any]],
        on_apply: Any = None,
    ) -> None:
        super().__init__(parent)
        self.title("Advanced Configuration")
        self.geometry("700x500")
        self.resizable(True, True)
        self.transient(parent)

        self._categories = categories
        self._overrides = overrides
        self._on_apply = on_apply
        self._vars: dict[str, dict[str, tk.Variable]] = {}

        self._build_ui()

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        for cat_name, cls_list in self._categories.items():
            frame = ttk.Frame(notebook)
            notebook.add(frame, text=cat_name)

            canvas = tk.Canvas(frame, borderwidth=0, highlightthickness=0)
            scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
            inner = ttk.Frame(canvas)

            inner.bind(
                "<Configure>",
                lambda e, c=canvas: c.configure(scrollregion=c.bbox("all")),
            )
            canvas.create_window((0, 0), window=inner, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            row = 0
            for cls in cls_list:
                cls_name = cls.__name__
                header = ttk.Label(
                    inner, text=cls_name, font=("Segoe UI", 10, "bold")
                )
                header.grid(row=row, column=0, columnspan=2, sticky="w", padx=8, pady=(8, 2))
                row += 1

                defaults = self._overrides.get(cls_name, {})
                self._vars[cls_name] = {}

                for f in dataclasses.fields(cls):
                    val = defaults.get(f.name, f.default)
                    lbl = ttk.Label(inner, text=f.name)
                    lbl.grid(row=row, column=0, sticky="w", padx=(16, 4), pady=1)

                    var, widget = self._make_widget(inner, f, val)
                    widget.grid(row=row, column=1, sticky="ew", padx=(0, 8), pady=1)
                    self._vars[cls_name][f.name] = var
                    row += 1

                sep = ttk.Separator(inner, orient="horizontal")
                sep.grid(row=row, column=0, columnspan=2, sticky="ew", padx=8, pady=4)
                row += 1

            inner.columnconfigure(1, weight=1)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)

        ttk.Button(btn_frame, text="Apply", command=self._apply).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="Reset to Defaults", command=self._reset).pack(
            side="right", padx=4
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side="right", padx=4
        )

    def _make_widget(
        self, parent: tk.Widget, f: dataclasses.Field, value: Any  # type: ignore[type-arg]
    ) -> tuple[tk.Variable, tk.Widget]:
        """Create an appropriate widget for a dataclass field."""
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(parent, variable=var)
            return var, widget

        if isinstance(value, int):
            var = tk.IntVar(value=value)
            widget = ttk.Spinbox(
                parent, textvariable=var, from_=-999999, to=999999, width=15
            )
            return var, widget

        if isinstance(value, float):
            var = tk.DoubleVar(value=value)
            widget = ttk.Entry(parent, textvariable=var, width=18)
            return var, widget

        # String or other — use Entry
        var = tk.StringVar(value=str(value))
        widget = ttk.Entry(parent, textvariable=var, width=18)
        return var, widget

    def _apply(self) -> None:
        """Collect all values and call the apply callback."""
        result: dict[str, dict[str, Any]] = {}
        for cls_name, fields_vars in self._vars.items():
            result[cls_name] = {}
            for fname, var in fields_vars.items():
                try:
                    result[cls_name][fname] = var.get()
                except (tk.TclError, ValueError):
                    pass  # Keep default if invalid
        if self._on_apply:
            self._on_apply(result)
        self.destroy()

    def _reset(self) -> None:
        """Reset all fields to their dataclass defaults."""
        for cls_list in self._categories.values():
            for cls in cls_list:
                cls_name = cls.__name__
                if cls_name not in self._vars:
                    continue
                for f in dataclasses.fields(cls):
                    if f.name in self._vars[cls_name]:
                        var = self._vars[cls_name][f.name]
                        try:
                            var.set(f.default)
                        except (tk.TclError, TypeError):
                            pass
