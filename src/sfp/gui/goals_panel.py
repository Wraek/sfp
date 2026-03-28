"""Goals display panel for the SFP GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


class GoalsPanel(ttk.LabelFrame):
    """Displays all goals with status, progress, priority, and controls."""

    def __init__(
        self,
        parent: tk.Widget,
        on_remove: callable | None = None,
        on_pause: callable | None = None,
        on_resume: callable | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(parent, text="Goals", **kwargs)
        self._on_remove = on_remove
        self._on_pause = on_pause
        self._on_resume = on_resume
        self._build()

    def _build(self) -> None:
        columns = ("id", "name", "status", "progress", "priority", "urgency")
        self._tree = ttk.Treeview(
            self,
            columns=columns,
            show="headings",
            height=5,
            selectmode="browse",
        )

        self._tree.heading("id", text="ID")
        self._tree.heading("name", text="Name")
        self._tree.heading("status", text="Status")
        self._tree.heading("progress", text="Progress")
        self._tree.heading("priority", text="Pri")
        self._tree.heading("urgency", text="Urg")

        self._tree.column("id", width=30, minwidth=30, stretch=False)
        self._tree.column("name", width=120, minwidth=60)
        self._tree.column("status", width=70, minwidth=50, stretch=False)
        self._tree.column("progress", width=60, minwidth=40, stretch=False)
        self._tree.column("priority", width=45, minwidth=35, stretch=False)
        self._tree.column("urgency", width=45, minwidth=35, stretch=False)

        # Scrollbar
        scroll = ttk.Scrollbar(self, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)

        self._tree.pack(side="left", fill="both", expand=True, padx=(4, 0), pady=4)
        scroll.pack(side="left", fill="y", pady=4)

        # Status color tags
        self._tree.tag_configure("active", foreground="")
        self._tree.tag_configure("completed", foreground="#228B22")
        self._tree.tag_configure("paused", foreground="#888888")
        self._tree.tag_configure("blocked", foreground="#CC8800")
        self._tree.tag_configure("failed", foreground="#CC0000")
        self._tree.tag_configure("expired", foreground="#CC0000")

        # Button column
        btn_frame = ttk.Frame(self)
        btn_frame.pack(side="right", fill="y", padx=(2, 4), pady=4)

        ttk.Button(btn_frame, text="Remove", width=8, command=self._on_remove_click).pack(
            pady=(0, 2),
        )
        ttk.Button(btn_frame, text="Pause", width=8, command=self._on_pause_click).pack(
            pady=2,
        )
        ttk.Button(btn_frame, text="Resume", width=8, command=self._on_resume_click).pack(
            pady=2,
        )

    def _selected_goal_id(self) -> int | None:
        sel = self._tree.selection()
        if not sel:
            return None
        item = self._tree.item(sel[0])
        try:
            return int(item["values"][0])
        except (IndexError, ValueError, TypeError):
            return None

    def _on_remove_click(self) -> None:
        gid = self._selected_goal_id()
        if gid is not None and self._on_remove is not None:
            self._on_remove(gid)

    def _on_pause_click(self) -> None:
        gid = self._selected_goal_id()
        if gid is not None and self._on_pause is not None:
            self._on_pause(gid)

    def _on_resume_click(self) -> None:
        gid = self._selected_goal_id()
        if gid is not None and self._on_resume is not None:
            self._on_resume(gid)

    def update_goals(self, goals: list[dict[str, Any]]) -> None:
        """Refresh the treeview from list_goals() data."""
        # Preserve selection
        sel_id = self._selected_goal_id()

        # Clear existing rows
        for item in self._tree.get_children():
            self._tree.delete(item)

        reselect_iid = None
        for g in goals:
            gid = g.get("id", "?")
            name = g.get("description") or f"Goal #{gid}"
            if len(name) > 30:
                name = name[:27] + "..."
            status = g.get("status", "?")
            progress = f"{g.get('progress', 0):.0%}"
            priority = f"{g.get('priority', 0):.2f}"
            urgency = f"{g.get('urgency', 0):.2f}"

            tag = status.lower()
            iid = self._tree.insert(
                "",
                "end",
                values=(gid, name, status, progress, priority, urgency),
                tags=(tag,),
            )
            if gid == sel_id:
                reselect_iid = iid

        # Restore selection
        if reselect_iid is not None:
            self._tree.selection_set(reselect_iid)
