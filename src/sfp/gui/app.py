"""Main SFP desktop application window."""

from __future__ import annotations

import logging
import tkinter as tk
import tkinter.simpledialog
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from sfp.bridge import BridgeLoadError, BridgeLoader
from sfp.gui.config_panel import ConfigDialog
from sfp.gui.goals_panel import GoalsPanel
from sfp.gui.log_panel import GUILogHandler, LogPanel
from sfp.gui.presets import CONFIG_CATEGORIES, PRESETS, get_default_overrides
from sfp.gui.session import SessionManager
from sfp.gui.settings import Settings
from sfp.gui.status_panel import StatusPanel
from sfp.gui.worker import SFPWorker


class SFPApp(tk.Tk):
    """Main application window for the SFP desktop GUI."""

    def __init__(self) -> None:
        super().__init__()
        self.title("SFP - Semantic Field Processor")
        self.geometry("900x620")
        self.minsize(700, 500)

        # State
        self._session_mgr = SessionManager()
        self._settings = Settings()
        self._worker = SFPWorker()
        self._preset_var = tk.StringVar(value="minimal")
        self._config_overrides: dict[str, dict[str, Any]] = get_default_overrides()
        self._pending_callbacks: dict[int, str] = {}  # cid → action name
        self._bridge: Any = None  # Active bridge instance
        self._health_poll_id: str | None = None  # after() ID for health polling
        self._goals_poll_id: str | None = None  # after() ID for goals polling

        self._build_menu()
        self._build_layout()
        self._setup_logging()

        # Start worker thread
        self._worker.start()

        # Start polling worker results
        self._poll_results()

        # Check for crash recovery
        self.after(500, self._check_recovery)

        # Graceful shutdown
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── Menu ────────────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="New Session...", command=self._on_new_session)
        file_menu.add_command(label="Load Session...", command=self._on_load_session)
        file_menu.add_command(label="Save Session", command=self._on_save_session)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        session_menu = tk.Menu(menubar, tearoff=0)
        session_menu.add_command(label="Process (Random)", command=self._on_process)
        session_menu.add_command(label="Query (Random)", command=self._on_query)
        session_menu.add_command(label="Consolidate", command=self._on_consolidate)
        session_menu.add_command(label="Reset Working Memory", command=self._on_reset)
        session_menu.add_command(label="Health Report", command=self._on_health)
        menubar.add_cascade(label="Session", menu=session_menu)

        bridge_menu = tk.Menu(menubar, tearoff=0)
        bridge_menu.add_command(
            label="Configure Bridge...", command=self._on_bridge_configure,
        )
        bridge_menu.add_command(
            label="Start Bridge", command=self._on_bridge_start,
        )
        bridge_menu.add_command(
            label="Stop Bridge", command=self._on_bridge_stop,
        )
        menubar.add_cascade(label="Bridge", menu=bridge_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About", command=self._on_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ── Layout ──────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        # Main paned window: left controls | right content
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        # Left panel — session controls
        left = ttk.Frame(paned, width=220)
        paned.add(left, weight=0)

        # Right panel — status, actions, log
        right = ttk.Frame(paned)
        paned.add(right, weight=1)

        self._build_left_panel(left)
        self._build_right_panel(right)

    def _build_left_panel(self, parent: ttk.Frame) -> None:
        # Session controls
        ctrl = ttk.LabelFrame(parent, text="Session")
        ctrl.pack(fill="x", padx=4, pady=(4, 2))

        ttk.Button(ctrl, text="New Session", command=self._on_new_session).pack(
            fill="x", padx=4, pady=2
        )
        ttk.Button(ctrl, text="Load Session", command=self._on_load_session).pack(
            fill="x", padx=4, pady=2
        )
        ttk.Button(ctrl, text="Save", command=self._on_save_session).pack(
            fill="x", padx=4, pady=2
        )
        ttk.Button(ctrl, text="Stop && Close", command=self._on_close).pack(
            fill="x", padx=4, pady=(2, 4)
        )

        # Preset selector
        preset_frame = ttk.LabelFrame(parent, text="Preset")
        preset_frame.pack(fill="x", padx=4, pady=2)

        for name in PRESETS:
            ttk.Radiobutton(
                preset_frame,
                text=name.capitalize(),
                variable=self._preset_var,
                value=name,
            ).pack(anchor="w", padx=8, pady=1)

        ttk.Button(
            preset_frame, text="Advanced...", command=self._on_advanced
        ).pack(fill="x", padx=4, pady=(2, 4))

        # Size selector
        size_frame = ttk.LabelFrame(parent, text="Field Size")
        size_frame.pack(fill="x", padx=4, pady=2)
        self._size_var = tk.StringVar(value="small")
        for s in ("tiny", "small", "medium", "large"):
            ttk.Radiobutton(
                size_frame, text=s.capitalize(), variable=self._size_var, value=s
            ).pack(anchor="w", padx=8, pady=1)

        # Bridge controls
        bridge_frame = ttk.LabelFrame(parent, text="Bridge")
        bridge_frame.pack(fill="x", padx=4, pady=2)

        self._bridge_status_var = tk.StringVar(value="Stopped")
        ttk.Label(
            bridge_frame, textvariable=self._bridge_status_var,
        ).pack(anchor="w", padx=8, pady=(4, 0))

        self._bridge_btn = ttk.Button(
            bridge_frame, text="Start Bridge", command=self._on_bridge_start,
        )
        self._bridge_btn.pack(fill="x", padx=4, pady=2)

        ttk.Button(
            bridge_frame, text="Configure...", command=self._on_bridge_configure,
        ).pack(fill="x", padx=4, pady=(0, 4))

    def _build_right_panel(self, parent: ttk.Frame) -> None:
        # Status panel
        self._status = StatusPanel(parent)
        self._status.pack(fill="x", padx=4, pady=(4, 2))

        # Action buttons
        actions = ttk.LabelFrame(parent, text="Processing")
        actions.pack(fill="x", padx=4, pady=2)

        btn_row = ttk.Frame(actions)
        btn_row.pack(fill="x", padx=4, pady=4)

        ttk.Button(btn_row, text="Process", command=self._on_process).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Query", command=self._on_query).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Consolidate", command=self._on_consolidate).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Reset T0", command=self._on_reset).pack(
            side="left", padx=2
        )
        ttk.Button(btn_row, text="Health", command=self._on_health).pack(
            side="left", padx=2
        )

        # Batch processing
        batch_row = ttk.Frame(actions)
        batch_row.pack(fill="x", padx=4, pady=(0, 4))

        ttk.Label(batch_row, text="Batch:").pack(side="left", padx=(0, 4))
        self._batch_var = tk.IntVar(value=10)
        ttk.Spinbox(
            batch_row, textvariable=self._batch_var, from_=1, to=10000, width=8
        ).pack(side="left", padx=2)
        ttk.Button(
            batch_row, text="Run Batch", command=self._on_batch_process
        ).pack(side="left", padx=2)

        # Goals panel
        self._goals_panel = GoalsPanel(
            parent,
            on_remove=self._on_goal_remove,
            on_pause=self._on_goal_pause,
            on_resume=self._on_goal_resume,
        )
        self._goals_panel.pack(fill="x", padx=4, pady=2)

        # Log panel
        self._log = LogPanel(parent)
        self._log.pack(fill="both", expand=True, padx=4, pady=(2, 4))

    # ── Logging ─────────────────────────────────────────────────────

    def _setup_logging(self) -> None:
        """Route SFP log records to the log panel."""
        handler = GUILogHandler(self._log, self)
        handler.setFormatter(logging.Formatter("%(name)s: %(message)s"))
        logging.getLogger("sfp").addHandler(handler)
        logging.getLogger("sfp-bridge").addHandler(handler)

    # ── Worker result polling ───────────────────────────────────────

    def _poll_results(self) -> None:
        """Poll the worker for completed results and dispatch to callbacks."""
        for cid, success, result in self._worker.poll_results():
            action = self._pending_callbacks.pop(cid, "unknown")
            if success:
                self._handle_result(action, result)
            else:
                self._log.log(f"Error [{action}]: {result}", "error")
                self._status.set_state("Error")

        self.after(100, self._poll_results)

    def _handle_result(self, action: str, result: Any) -> None:
        """Handle a successful worker result."""
        if action == "create_session":
            self._log.log(str(result), "success")
            self._status.set_state("Ready")
            # Configure auto-save
            asp = self._session_mgr.auto_save_path
            if asp:
                self._worker.submit("set_auto_save", path=str(asp))
            # Start periodic health + goals refresh
            self._poll_health()
            self._poll_goals()

        elif action == "load_session":
            self._log.log(str(result), "success")
            self._status.set_state("Ready")
            asp = self._session_mgr.auto_save_path
            if asp:
                self._worker.submit("set_auto_save", path=str(asp))
            # Start periodic health + goals refresh
            self._poll_health()
            self._poll_goals()

        elif action == "save_session":
            self._log.log(str(result), "success")
            self._session_mgr.mark_saved()

        elif action == "process":
            self._status.update_surprise(result)
            flag = "updated" if result["updated"] else "skipped"
            self._log.log(
                f"Process: grad={result['grad_norm']:.4f} "
                f"loss={result['loss']:.4f} ({flag})"
            )
            self._status.set_state("Ready")

        elif action == "query":
            self._log.log(
                f"Query: hops={result.get('hops', '?')} "
                f"basins={result.get('basins_visited', '?')} "
                f"conf={result.get('confidence', '?'):.3f}"
            )
            self._status.set_state("Ready")

        elif action == "consolidate":
            self._log.log(str(result), "success")
            self._status.set_state("Ready")

        elif action == "reset":
            self._log.log(str(result), "success")

        elif action == "health_report":
            self._status.update_from_health(result)

        elif action == "memory_footprint":
            self._status.update_memory(result)

        elif action == "list_goals":
            self._goals_panel.update_goals(result)

        elif action in ("remove_goal", "pause_goal", "resume_goal"):
            self._log.log(str(result))

        elif action == "batch_process":
            self._status.update_surprise(result)
            self._batch_remaining -= 1
            if self._batch_remaining > 0:
                cid = self._worker.submit("process")
                self._pending_callbacks[cid] = "batch_process"
            else:
                self._log.log(f"Batch complete ({self._batch_total} steps)", "success")
                self._status.set_state("Ready")
                cid = self._worker.submit("health_report")
                self._pending_callbacks[cid] = "health_report"

    # ── Action handlers ─────────────────────────────────────────────

    def _get_create_kwargs(self) -> dict[str, Any]:
        """Build create_field() kwargs from current preset + overrides."""
        preset = self._preset_var.get()
        kwargs = dict(PRESETS.get(preset, PRESETS["minimal"]))
        kwargs["size"] = self._size_var.get()
        return kwargs

    def _on_new_session(self) -> None:
        """Create a new SFP session."""
        # Simple name dialog
        name = tk.simpledialog.askstring(
            "New Session", "Session name:", parent=self
        )
        if not name:
            return

        preset = self._preset_var.get()
        create_kwargs = self._get_create_kwargs()

        self._session_mgr.new_session(name, preset, create_kwargs)
        self._log.log(f"Creating session '{name}' (preset={preset}, size={create_kwargs['size']})")
        self._status.set_state("Initializing...")

        cid = self._worker.submit("create_session", create_kwargs=create_kwargs)
        self._pending_callbacks[cid] = "create_session"

    def _on_load_session(self) -> None:
        """Load a session from a list or file."""
        sessions = self._session_mgr.list_sessions()

        if sessions:
            # Show session picker dialog
            dialog = _SessionPickerDialog(self, sessions)
            self.wait_window(dialog)
            if dialog.result is None:
                return
            session_path = Path(dialog.result)
        else:
            # Fallback to file dialog
            path = filedialog.askopenfilename(
                title="Load Session Checkpoint",
                filetypes=[("Checkpoint", "*.pt"), ("All files", "*.*")],
                parent=self,
            )
            if not path:
                return
            session_path = Path(path).parent

        # Open session
        meta = self._session_mgr.open_session(session_path)
        checkpoint = session_path / "checkpoint.pt"
        if not checkpoint.exists():
            # Try auto-save
            checkpoint = session_path / "autosave.pt"
        if not checkpoint.exists():
            self._log.log(f"No checkpoint found in {session_path}", "error")
            return

        self._log.log(f"Loading session '{meta.get('name', 'unknown')}'...")
        self._status.set_state("Loading...")
        cid = self._worker.submit("load_session", path=str(checkpoint))
        self._pending_callbacks[cid] = "load_session"

    def _on_save_session(self) -> None:
        """Save the current session."""
        if not self._worker.has_processor:
            self._log.log("No active session to save", "warning")
            return

        path = self._session_mgr.checkpoint_path
        if path is None:
            # No session dir — ask for path
            path_str = filedialog.asksaveasfilename(
                title="Save Checkpoint",
                defaultextension=".pt",
                filetypes=[("Checkpoint", "*.pt")],
                parent=self,
            )
            if not path_str:
                return
            path = Path(path_str)

        self._log.log("Saving session...")
        self._status.set_state("Saving...")
        meta = {
            "session_name": self._session_mgr.current_name or "unnamed",
            "preset": self._preset_var.get(),
        }
        cid = self._worker.submit("save_session", path=str(path), metadata=meta)
        self._pending_callbacks[cid] = "save_session"

    def _on_process(self) -> None:
        """Process a random tensor."""
        if not self._worker.has_processor:
            self._log.log("No active session", "warning")
            return
        self._status.set_state("Processing...")
        cid = self._worker.submit("process")
        self._pending_callbacks[cid] = "process"

    def _on_query(self) -> None:
        """Query the system (read-only)."""
        if not self._worker.has_processor:
            self._log.log("No active session", "warning")
            return
        self._status.set_state("Querying...")
        cid = self._worker.submit("query")
        self._pending_callbacks[cid] = "query"

    def _on_consolidate(self) -> None:
        """Trigger manual consolidation."""
        if not self._worker.has_processor:
            self._log.log("No active session", "warning")
            return
        self._status.set_state("Consolidating...")
        cid = self._worker.submit("consolidate")
        self._pending_callbacks[cid] = "consolidate"

    def _on_reset(self) -> None:
        """Reset Tier 0 working memory."""
        if not self._worker.has_processor:
            self._log.log("No active session", "warning")
            return
        cid = self._worker.submit("reset_session")
        self._pending_callbacks[cid] = "reset"

    def _on_health(self) -> None:
        """Refresh health report."""
        if not self._worker.has_processor:
            return
        cid = self._worker.submit("health_report")
        self._pending_callbacks[cid] = "health_report"
        cid2 = self._worker.submit("memory_footprint")
        self._pending_callbacks[cid2] = "memory_footprint"

    def _on_batch_process(self) -> None:
        """Run multiple process steps."""
        if not self._worker.has_processor:
            self._log.log("No active session", "warning")
            return
        count = self._batch_var.get()
        self._batch_remaining = count
        self._batch_total = count
        self._status.set_state(f"Batch: 0/{count}")
        self._log.log(f"Starting batch of {count} process steps...")
        cid = self._worker.submit("process")
        self._pending_callbacks[cid] = "batch_process"

    def _on_advanced(self) -> None:
        """Open the advanced config dialog."""
        ConfigDialog(
            self,
            CONFIG_CATEGORIES,
            self._config_overrides,
            on_apply=self._apply_overrides,
        )

    def _apply_overrides(self, overrides: dict[str, dict[str, Any]]) -> None:
        """Store config overrides from the advanced dialog."""
        self._config_overrides = overrides
        self._log.log("Advanced config updated (applies to next new session)")

    # ── Bridge handlers ────────────────────────────────────────────

    def _on_bridge_configure(self) -> None:
        """Open the bridge configuration dialog."""
        _BridgeConfigDialog(self, self._settings)

    def _on_bridge_start(self) -> None:
        """Load and start the bridge."""
        if self._bridge is not None and self._bridge.is_running:
            self._log.log("Bridge is already running", "warning")
            return

        interface = self._worker.get_interface()
        if interface is None:
            self._log.log(
                "Bridge requires an active hierarchical session", "warning",
            )
            return

        entry_point = self._settings.bridge_entry_point
        module_path = self._settings.bridge_module_path or None

        if not entry_point:
            self._log.log("No bridge entry point configured", "warning")
            self._on_bridge_configure()
            return

        try:
            loader = BridgeLoader(entry_point, module_path)
            bridge_cls = loader.load()
            self._bridge = bridge_cls()
            self._bridge.start(interface)
            self._log.log(f"Bridge started: {bridge_cls.__name__}", "success")
            self._bridge_status_var.set("Running")
            self._bridge_btn.configure(
                text="Stop Bridge", command=self._on_bridge_stop,
            )
            self._poll_bridge_status()
        except BridgeLoadError as exc:
            self._log.log(f"Bridge load failed: {exc}", "error")
        except Exception as exc:
            self._log.log(f"Bridge start failed: {exc}", "error")

    def _on_bridge_stop(self) -> None:
        """Stop the running bridge."""
        if self._bridge is None:
            return
        try:
            self._bridge.stop()
            self._log.log("Bridge stopped", "success")
        except Exception as exc:
            self._log.log(f"Bridge stop error: {exc}", "error")
        finally:
            self._bridge = None
            self._bridge_status_var.set("Stopped")
            self._status.update_bridge({"state": "stopped"})
            self._bridge_btn.configure(
                text="Start Bridge", command=self._on_bridge_start,
            )

    def _poll_bridge_status(self) -> None:
        """Periodically update bridge status in the UI."""
        if self._bridge is None:
            return
        try:
            if self._bridge.is_running:
                status = self._bridge.status()
                state = status.get("state", "running")
                self._bridge_status_var.set(state.capitalize())
                self._status.update_bridge(status)
                self.after(2000, self._poll_bridge_status)
            else:
                self._bridge_status_var.set("Stopped")
                self._status.update_bridge({"state": "stopped"})
                self._bridge_btn.configure(
                    text="Start Bridge", command=self._on_bridge_start,
                )
                self._bridge = None
        except Exception:
            self._bridge_status_var.set("Error")

    def _poll_health(self) -> None:
        """Periodically refresh health & memory stats while a session is active."""
        if self._worker is None or not self._worker.has_processor:
            self._health_poll_id = None
            return
        cid = self._worker.submit("health_report")
        self._pending_callbacks[cid] = "health_report"
        cid2 = self._worker.submit("memory_footprint")
        self._pending_callbacks[cid2] = "memory_footprint"
        self._health_poll_id = self.after(2000, self._poll_health)

    def _stop_health_poll(self) -> None:
        """Cancel the periodic health polling loop."""
        if self._health_poll_id is not None:
            self.after_cancel(self._health_poll_id)
            self._health_poll_id = None

    def _poll_goals(self) -> None:
        """Periodically refresh the goals panel while a session is active."""
        if self._worker is None or not self._worker.has_processor:
            self._goals_poll_id = None
            return
        cid = self._worker.submit("list_goals")
        self._pending_callbacks[cid] = "list_goals"
        self._goals_poll_id = self.after(2000, self._poll_goals)

    def _stop_goals_poll(self) -> None:
        """Cancel the periodic goals polling loop."""
        if self._goals_poll_id is not None:
            self.after_cancel(self._goals_poll_id)
            self._goals_poll_id = None

    def _on_goal_remove(self, goal_id: int) -> None:
        """Remove the selected goal."""
        cid = self._worker.submit("remove_goal", goal_id=goal_id)
        self._pending_callbacks[cid] = "remove_goal"

    def _on_goal_pause(self, goal_id: int) -> None:
        """Pause the selected goal."""
        cid = self._worker.submit("pause_goal", goal_id=goal_id)
        self._pending_callbacks[cid] = "pause_goal"

    def _on_goal_resume(self, goal_id: int) -> None:
        """Resume the selected goal."""
        cid = self._worker.submit("resume_goal", goal_id=goal_id)
        self._pending_callbacks[cid] = "resume_goal"

    def _on_about(self) -> None:
        from sfp._version import __version__

        messagebox.showinfo(
            "About SFP",
            f"SFP - Semantic Field Processor\n"
            f"Version {__version__}\n\n"
            f"Knowledge encoded as the shape of neural network weights.\n"
            f"Data transforms the manifold and moves on.",
            parent=self,
        )

    def _check_recovery(self) -> None:
        """Check for auto-save recovery on startup."""
        recovery = self._session_mgr.check_recovery()
        if recovery is not None:
            if messagebox.askyesno(
                "Crash Recovery",
                f"Found unsaved session data at:\n{recovery}\n\nRecover?",
                parent=self,
            ):
                session_dir = recovery.parent
                self._session_mgr.open_session(session_dir)
                self._log.log("Recovering session from auto-save...")
                self._status.set_state("Loading...")
                cid = self._worker.submit("load_session", path=str(recovery))
                self._pending_callbacks[cid] = "load_session"

    def _on_close(self) -> None:
        """Graceful shutdown: save if active, then exit."""
        if self._worker.has_processor:
            if messagebox.askyesno(
                "Save Session?",
                "Save session before exiting?",
                parent=self,
            ):
                self._on_save_session()
                # Give worker a moment to save
                self.after(2000, self._shutdown)
                return
        self._shutdown()

    def _shutdown(self) -> None:
        self._stop_health_poll()
        self._stop_goals_poll()
        if self._bridge is not None:
            try:
                self._bridge.stop()
            except Exception:
                pass
            self._bridge = None
        self._worker.stop()
        self.destroy()


class _SessionPickerDialog(tk.Toplevel):
    """Simple dialog to pick from a list of saved sessions."""

    def __init__(self, parent: tk.Widget, sessions: list[dict[str, Any]]) -> None:
        super().__init__(parent)
        self.title("Load Session")
        self.geometry("400x300")
        self.transient(parent)
        self.result: str | None = None

        self._sessions = sessions

        listbox = tk.Listbox(self, font=("Segoe UI", 10))
        listbox.pack(fill="both", expand=True, padx=8, pady=(8, 0))

        for s in sessions:
            name = s.get("name", "unnamed")
            preset = s.get("preset", "?")
            has_ckpt = "saved" if s.get("has_checkpoint") else "unsaved"
            listbox.insert("end", f"{name} [{preset}] ({has_ckpt})")

        self._listbox = listbox

        btn_frame = ttk.Frame(self)
        btn_frame.pack(fill="x", padx=8, pady=8)
        ttk.Button(btn_frame, text="Open", command=self._on_open).pack(side="right", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side="right", padx=4)

    def _on_open(self) -> None:
        sel = self._listbox.curselection()
        if sel:
            idx = sel[0]
            self.result = self._sessions[idx].get("path")
        self.destroy()


class _BridgeConfigDialog(tk.Toplevel):
    """Dialog to configure the bridge module path and entry point."""

    def __init__(self, parent: tk.Widget, settings: Settings) -> None:
        super().__init__(parent)
        self.title("Configure Bridge")
        self.geometry("500x180")
        self.transient(parent)
        self._settings = settings

        # Module path
        ttk.Label(self, text="Module Path:").grid(
            row=0, column=0, sticky="w", padx=8, pady=(12, 2),
        )
        self._path_var = tk.StringVar(value=settings.bridge_module_path)
        path_frame = ttk.Frame(self)
        path_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=2)
        path_frame.columnconfigure(0, weight=1)
        ttk.Entry(path_frame, textvariable=self._path_var).grid(
            row=0, column=0, sticky="ew",
        )
        ttk.Button(path_frame, text="Browse...", command=self._browse).grid(
            row=0, column=1, padx=(4, 0),
        )

        # Entry point
        ttk.Label(self, text="Entry Point (module:Class):").grid(
            row=2, column=0, sticky="w", padx=8, pady=(8, 2),
        )
        self._entry_var = tk.StringVar(value=settings.bridge_entry_point)
        ttk.Entry(self, textvariable=self._entry_var).grid(
            row=3, column=0, sticky="ew", padx=8, pady=2,
        )

        self.columnconfigure(0, weight=1)

        # Buttons
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=4, column=0, sticky="e", padx=8, pady=(12, 8))
        ttk.Button(btn_frame, text="Save", command=self._save).pack(
            side="right", padx=4,
        )
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(
            side="right", padx=4,
        )

    def _browse(self) -> None:
        path = filedialog.askdirectory(
            title="Select Bridge Module Directory", parent=self,
        )
        if path:
            self._path_var.set(path)

    def _save(self) -> None:
        self._settings.bridge_module_path = self._path_var.get().strip()
        self._settings.bridge_entry_point = self._entry_var.get().strip()
        self.destroy()


def main() -> None:
    """Entry point for the SFP GUI."""
    app = SFPApp()
    app.mainloop()


if __name__ == "__main__":
    main()
