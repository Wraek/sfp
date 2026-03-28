"""Background worker thread for SFP operations.

Tkinter must run on the main thread. All SFP operations (create, process,
save, etc.) run on a daemon background thread. Communication uses two queues:
command_queue (main → worker) and result_queue (worker → main).
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from typing import Any, Callable

import torch

from sfp.storage.serialization import SessionCheckpoint
from sfp.utils.logging import get_logger

logger = get_logger("gui.worker")


class SFPWorker:
    """Manages the SFP processor on a background thread."""

    def __init__(self) -> None:
        self._command_queue: queue.Queue[tuple[str, dict[str, Any], int]] = queue.Queue()
        self._result_queue: queue.Queue[tuple[int, bool, Any]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._processor: Any = None
        self._interface: Any = None  # SFPInterface wrapper (for bridges)
        self._create_kwargs: dict[str, Any] = {}
        self._running = False
        self._callback_counter = 0
        self._auto_save_path: str | None = None
        self._auto_save_interval: float = 300.0  # 5 minutes
        self._last_auto_save: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running and self._thread is not None and self._thread.is_alive()

    @property
    def has_processor(self) -> bool:
        return self._processor is not None

    def get_interface(self) -> Any:
        """Return the SFPInterface wrapper, or ``None``.

        Only available when the processor is a
        ``HierarchicalMemoryProcessor``.  The bridge uses this for
        thread-safe access to the running processor.
        """
        return self._interface

    def start(self) -> None:
        """Start the worker thread."""
        if self.is_running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._running = False

    def submit(self, command: str, **kwargs: Any) -> int:
        """Submit a command and return its callback ID."""
        self._callback_counter += 1
        cid = self._callback_counter
        self._command_queue.put((command, kwargs, cid))
        return cid

    def poll_results(self) -> list[tuple[int, bool, Any]]:
        """Non-blocking poll for completed results."""
        results = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def _run_loop(self) -> None:
        """Main worker loop — processes commands from the queue."""
        while self._running:
            # Check auto-save
            if (
                self._processor is not None
                and self._auto_save_path is not None
                and time.time() - self._last_auto_save > self._auto_save_interval
            ):
                self._do_auto_save()

            try:
                command, kwargs, cid = self._command_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self._dispatch(command, kwargs)
                self._result_queue.put((cid, True, result))
            except Exception as exc:
                tb = traceback.format_exc()
                self._result_queue.put((cid, False, f"{exc}\n{tb}"))

    def _dispatch(self, command: str, kwargs: dict[str, Any]) -> Any:
        """Dispatch a command to the appropriate handler."""
        handler = getattr(self, f"_cmd_{command}", None)
        if handler is None:
            msg = f"Unknown command: {command}"
            raise ValueError(msg)
        return handler(**kwargs)

    # ── Commands ────────────────────────────────────────────────────

    def _cmd_create_session(self, create_kwargs: dict[str, Any]) -> str:
        """Create a new SFP session from create_field() kwargs."""
        import sfp
        from sfp.memory.processor import HierarchicalMemoryProcessor

        self._create_kwargs = dict(create_kwargs)
        self._processor = sfp.create_field(**create_kwargs)
        if isinstance(self._processor, HierarchicalMemoryProcessor):
            self._interface = sfp.SFPInterface(self._processor)
        else:
            self._interface = None
        self._last_auto_save = time.time()
        ptype = type(self._processor).__name__
        return f"Session created: {ptype}"

    def _cmd_load_session(self, path: str, device: str = "auto") -> str:
        """Load a session from a checkpoint file."""
        from sfp.memory.processor import HierarchicalMemoryProcessor

        processor, create_kwargs, metadata = SessionCheckpoint.load(path, device=device)
        self._processor = processor
        self._create_kwargs = create_kwargs
        if isinstance(self._processor, HierarchicalMemoryProcessor):
            import sfp
            self._interface = sfp.SFPInterface(self._processor)
        else:
            self._interface = None
        self._last_auto_save = time.time()
        name = metadata.get("session_name", "unnamed")
        return f"Session loaded: {name}"

    def _cmd_save_session(self, path: str, metadata: dict[str, Any] | None = None) -> str:
        """Save the current session to a checkpoint file."""
        if self._processor is None:
            return "No active session to save"
        SessionCheckpoint.save(path, self._processor, self._create_kwargs, metadata)
        return f"Session saved to {path}"

    def _cmd_process(self, dim: int | None = None) -> dict[str, Any]:
        """Process a random tensor through the system."""
        if self._processor is None:
            msg = "No active session"
            raise RuntimeError(msg)

        if dim is None:
            # Infer dim from processor
            from sfp.core.streaming import StreamingProcessor
            from sfp.memory.processor import HierarchicalMemoryProcessor

            if isinstance(self._processor, HierarchicalMemoryProcessor):
                dim = self._processor._tier0.field.config.dim
            elif isinstance(self._processor, StreamingProcessor):
                dim = self._processor.field.config.dim
            else:
                dim = self._processor.config.dim

        x = torch.randn(dim)
        metric = self._processor.process(x)
        return {
            "grad_norm": metric.grad_norm,
            "loss": metric.loss,
            "updated": metric.updated,
        }

    def _cmd_query(self, dim: int | None = None) -> dict[str, Any]:
        """Query the system (read-only, no weight update)."""
        if self._processor is None:
            msg = "No active session"
            raise RuntimeError(msg)

        from sfp.memory.processor import HierarchicalMemoryProcessor

        if not isinstance(self._processor, HierarchicalMemoryProcessor):
            return {"error": "Query requires hierarchical mode"}

        if dim is None:
            dim = self._processor._tier0.field.config.dim

        x = torch.randn(dim)
        result = self._processor.query(x)
        return {
            "hops": result.hops,
            "basins_visited": len(result.visited_basins),
            "confidence": result.confidence,
        }

    def _cmd_consolidate(self) -> str:
        """Trigger manual consolidation."""
        if self._processor is None:
            msg = "No active session"
            raise RuntimeError(msg)

        from sfp.memory.processor import HierarchicalMemoryProcessor

        if not isinstance(self._processor, HierarchicalMemoryProcessor):
            return "Consolidation requires hierarchical mode"

        self._processor.consolidate()
        return "Consolidation complete"

    def _cmd_reset_session(self) -> str:
        """Reset Tier 0 working memory."""
        if self._processor is None:
            msg = "No active session"
            raise RuntimeError(msg)

        from sfp.memory.processor import HierarchicalMemoryProcessor

        if isinstance(self._processor, HierarchicalMemoryProcessor):
            self._processor.reset_session()
            return "Tier 0 working memory reset"
        return "Reset requires hierarchical mode"

    def _cmd_health_report(self) -> dict[str, Any]:
        """Get health report from the processor."""
        if self._processor is None:
            return {"status": "no_session"}

        from sfp.memory.processor import HierarchicalMemoryProcessor

        if isinstance(self._processor, HierarchicalMemoryProcessor):
            return self._processor.health_report()

        from sfp.core.streaming import StreamingProcessor

        if isinstance(self._processor, StreamingProcessor):
            return {
                "step_count": self._processor._step_count,
                "surprise_history_len": len(self._processor._history),
            }

        return {"status": "bare_field"}

    def _cmd_memory_footprint(self) -> dict[str, Any]:
        """Get memory footprint from the processor."""
        if self._processor is None:
            return {"total": 0}

        from sfp.memory.processor import HierarchicalMemoryProcessor

        if isinstance(self._processor, HierarchicalMemoryProcessor):
            return self._processor.memory_footprint()
        return {"total": 0}

    def _cmd_list_goals(self) -> list[dict[str, Any]]:
        """List all goals via the SFPInterface."""
        if self._interface is None:
            return []
        return self._interface.list_goals()

    def _cmd_remove_goal(self, goal_id: int) -> str:
        """Remove a goal by ID."""
        if self._interface is None:
            return "No active session"
        ok = self._interface.remove_goal(goal_id)
        return f"Goal {goal_id} {'removed' if ok else 'not found'}"

    def _cmd_pause_goal(self, goal_id: int) -> str:
        """Pause an active goal."""
        if self._interface is None:
            return "No active session"
        self._interface.pause_goal(goal_id)
        return f"Goal {goal_id} paused"

    def _cmd_resume_goal(self, goal_id: int) -> str:
        """Resume a paused goal."""
        if self._interface is None:
            return "No active session"
        self._interface.resume_goal(goal_id)
        return f"Goal {goal_id} resumed"

    def _cmd_set_auto_save(self, path: str, interval: float = 300.0) -> str:
        """Configure auto-save path and interval."""
        self._auto_save_path = path
        self._auto_save_interval = interval
        return f"Auto-save configured: {path} every {interval}s"

    def _do_auto_save(self) -> None:
        """Perform an auto-save (called from the worker loop)."""
        if self._processor is None or self._auto_save_path is None:
            return
        try:
            SessionCheckpoint.save(
                self._auto_save_path,
                self._processor,
                self._create_kwargs,
                {"auto_save": True, "timestamp": time.time()},
            )
            self._last_auto_save = time.time()
            logger.info("Auto-saved session")
        except Exception:
            logger.warning("Auto-save failed", exc_info=True)
