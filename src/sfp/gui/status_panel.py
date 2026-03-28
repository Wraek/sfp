"""Live status display panel for the SFP GUI."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


class StatusPanel(ttk.LabelFrame):
    """Displays live system metrics: state, steps, memory, tiers, modules."""

    def __init__(self, parent: tk.Widget, **kwargs: object) -> None:
        super().__init__(parent, text="System Status", **kwargs)

        self._labels: dict[str, ttk.Label] = {}
        self._create_rows()

    def _create_rows(self) -> None:
        rows = [
            ("state", "State:", "Idle"),
            ("steps", "Steps:", "0"),
            ("memory", "Memory:", "—"),
            ("surprise", "Last Surprise:", "—"),
            ("tiers", "Tiers:", "—"),
            ("modules", "Modules:", "—"),
            ("bridge", "Bridge:", "Stopped"),
        ]
        for i, (key, label_text, default) in enumerate(rows):
            lbl = ttk.Label(self, text=label_text, font=("Segoe UI", 9, "bold"))
            lbl.grid(row=i, column=0, sticky="w", padx=(8, 4), pady=2)
            val = ttk.Label(self, text=default, font=("Segoe UI", 9))
            val.grid(row=i, column=1, sticky="w", padx=(0, 8), pady=2)
            self._labels[key] = val

        self.columnconfigure(1, weight=1)

    def set_state(self, state: str) -> None:
        self._labels["state"].configure(text=state)

    def update_from_health(self, report: dict[str, Any]) -> None:
        """Update display from a health_report() dict."""
        self._labels["steps"].configure(text=str(report.get("step_count", "—")))

        # Tiers summary
        tier_parts = []
        t0 = report.get("tier0", {})
        if t0:
            tier_parts.append(f"T0:{t0.get('param_count', '?')}p")
        t1 = report.get("tier1", {})
        if t1:
            t1_text = f"T1:{t1.get('total_count', 0)}ep"
            sp = t1.get("spatial_count", 0)
            if sp > 0:
                t1_text += f"({sp}sp)"
            tier_parts.append(t1_text)
        t2 = report.get("tier2", {})
        if t2:
            tier_parts.append(f"T2:{t2.get('n_active', 0)}bas")
        t3 = report.get("tier3", {})
        if t3:
            tier_parts.append(f"T3:{t3.get('n_active', 0)}ax")
        if tier_parts:
            self._labels["tiers"].configure(text="  ".join(tier_parts))

        # Modules
        mod_parts = []
        if "world_model" in report:
            wm = report["world_model"]
            wm_text = "WM:on"
            sl = wm.get("spatial_loss", 0)
            if sl > 0:
                wm_text += f"(sp:{sl:.3f})"
            mod_parts.append(wm_text)
        if "scene_graph" in report:
            sg = report["scene_graph"]
            mod_parts.append(f"SG:{sg.get('n_nodes', 0)}n")
        if "goals" in report:
            g = report["goals"]
            mod_parts.append(f"Goals:{g.get('active_count', 0)}")
        if "metacognition" in report:
            mod_parts.append("Meta:on")
        if "valence" in report:
            mod_parts.append("Val:on")
        if "replay" in report:
            mod_parts.append("Replay:on")
        self._labels["modules"].configure(
            text="  ".join(mod_parts) if mod_parts else "none"
        )

    def update_memory(self, footprint: dict[str, Any]) -> None:
        """Update memory display from memory_footprint() dict."""
        total = footprint.get("total", 0)
        if total > 0:
            mb = total / (1024 * 1024)
            self._labels["memory"].configure(text=f"{mb:.1f} MB")
        else:
            self._labels["memory"].configure(text="—")

    def update_surprise(self, metric: dict[str, Any]) -> None:
        """Update last surprise metric display."""
        gn = metric.get("grad_norm", 0)
        loss = metric.get("loss", 0)
        updated = metric.get("updated", False)
        flag = "updated" if updated else "skipped"
        self._labels["surprise"].configure(
            text=f"grad={gn:.4f}  loss={loss:.4f}  ({flag})"
        )

    def update_bridge(self, status: dict[str, Any]) -> None:
        """Update bridge status from bridge.status() dict."""
        state = status.get("state", "stopped").capitalize()
        parts = [state]

        mode = status.get("mode")
        if mode:
            parts.append(mode)

        ticks = status.get("inference_ticks", 0)
        if ticks > 0:
            parts.append(f"{ticks}t")

        ent = status.get("entities_tracked", 0)
        if ent > 0:
            parts.append(f"{ent}ent")

        if status.get("has_depth"):
            parts.append("D")

        tmp = status.get("temporal_tokens", 0)
        if tmp > 0:
            parts.append(f"{tmp}tmp")

        tok = status.get("tokens_per_tick", 0)
        if tok > 0:
            parts.append(f"{tok}tok")

        obs = status.get("obs_received", 0)
        if obs > 0:
            ok = status.get("process_ok", 0)
            err = status.get("process_err", 0)
            parts.append(f"{ok}/{obs}proc")
            if err > 0:
                parts.append(f"{err}err")

        self._labels["bridge"].configure(text="  ".join(parts))
