"""
Centralised background-task manager with queue, pause/resume and cancel support.

Usage from any module's ``serve.py``::

    def my_heavy_work(ctx, files, model):
        for i, f in enumerate(files):
            ctx.check()                       # cooperative cancel / pause
            result = model.process(f)
            ctx.update((i + 1) / len(files),  # progress 0 → 1
                       f"Processing {i+1}/{len(files)}")

    task_id = app.task_manager.submit(
        name="Generate image embeddings",
        fn=my_heavy_work,
        args=(files, model),
    )
"""

from __future__ import annotations

import dataclasses
import threading
import time
import traceback
import uuid
from collections import deque
from typing import Any, Callable

from flask import Flask
from flask_socketio import SocketIO


# ------------------------------------------------------------------
# Exceptions
# ------------------------------------------------------------------

class TaskCancelled(Exception):
    """Raised inside a task when the user requests cancellation."""


# ------------------------------------------------------------------
# TaskContext  — the object every task function receives as 1st arg
# ------------------------------------------------------------------

class TaskContext:
    """Co-operative handle passed to the task callable.

    The task must call :meth:`check` periodically (e.g. once per loop
    iteration).  It may call :meth:`update` to report progress.
    """

    def __init__(self, task: Task, manager: TaskManager):
        self._task = task
        self._manager = manager
        self._last_emit: float = 0.0
        self._throttle: float = 0.25  # seconds

    # -- public API used inside task functions -------------------------

    def check(self) -> None:
        """Block while paused; raise :class:`TaskCancelled` if cancelled."""
        while True:
            if self._task.cancel_event.is_set():
                raise TaskCancelled()
            # wait() returns True immediately when the event is set (= not paused)
            if self._task.pause_event.wait(timeout=0.25):
                break
            # Still paused — loop to re-check cancel_event

    def update(self, progress: float, message: str = "") -> None:
        """Report progress (0.0 – 1.0) and an optional status string."""
        self._task.progress = max(0.0, min(1.0, progress))
        if message:
            self._task.message = message
        now = time.time()
        if now - self._last_emit >= self._throttle:
            self._last_emit = now
            self._manager._broadcast()


# ------------------------------------------------------------------
# Task dataclass
# ------------------------------------------------------------------

@dataclasses.dataclass
class Task:
    id: str
    name: str
    status: str               # queued | running | paused | completed | failed | cancelled
    progress: float           # 0.0 – 1.0
    message: str
    created_at: float
    started_at: float | None
    finished_at: float | None
    cancel_event: threading.Event
    pause_event: threading.Event   # SET = running, CLEARED = paused
    fn: Callable
    args: tuple
    kwargs: dict
    error: str | None

    # -- serialisation for the frontend --------------------------------

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "progress": round(self.progress, 4),
            "message": self.message,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


# ------------------------------------------------------------------
# TaskManager
# ------------------------------------------------------------------

_HISTORY_LIMIT = 50


class TaskManager:
    """Single-worker background queue with socket-based status updates."""

    def __init__(self, socketio: SocketIO, app: Flask):
        self.socketio = socketio
        self.app = app
        self._lock = threading.Lock()
        self._queue: deque[Task] = deque()
        self._active: Task | None = None
        self._history: deque[dict] = deque(maxlen=_HISTORY_LIMIT)
        self._work_available = threading.Event()

        # Start the single consumer thread
        t = threading.Thread(target=self._worker, daemon=True, name="task-manager-worker")
        t.start()

        # Register socket events
        self._register_socket_events()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, name: str, fn: Callable, args: tuple = (), kwargs: dict | None = None) -> str:
        """Enqueue a new task.  Returns the task id."""
        task = Task(
            id=uuid.uuid4().hex,
            name=name,
            status="queued",
            progress=0.0,
            message="Waiting in queue…",
            created_at=time.time(),
            started_at=None,
            finished_at=None,
            cancel_event=threading.Event(),
            pause_event=threading.Event(),
            fn=fn,
            args=args,
            kwargs=kwargs or {},
            error=None,
        )
        task.pause_event.set()  # not paused by default
        with self._lock:
            self._queue.append(task)
        self._work_available.set()
        self._broadcast()
        return task.id

    def get_state(self) -> dict:
        """Return a snapshot of the full manager state."""
        with self._lock:
            active = self._active.to_dict() if self._active else None
            queued = [t.to_dict() for t in self._queue]
            history = list(self._history)
        return {"active": active, "queued": queued, "history": history}

    def wait_for_task(self, task_id: str, poll_interval: float = 2.0) -> None:
        """Block the calling thread until *task_id* is no longer active or queued.

        Returns immediately if the task id is not found (already completed,
        cancelled, or was never submitted).  Safe to call from any thread.
        """
        while True:
            with self._lock:
                is_active = self._active is not None and self._active.id == task_id
                is_queued = any(t.id == task_id for t in self._queue)
            if not is_active and not is_queued:
                return
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # Socket events
    # ------------------------------------------------------------------

    def _register_socket_events(self) -> None:
        sio = self.socketio

        @sio.on("task_manager_get_state")
        def _on_get_state(data=None):
            return self.get_state()

        @sio.on("task_manager_cancel")
        def _on_cancel(data):
            self._cancel_task(data.get("task_id", ""))

        @sio.on("task_manager_pause")
        def _on_pause(data):
            self._pause_task(data.get("task_id", ""))

        @sio.on("task_manager_resume")
        def _on_resume(data):
            self._resume_task(data.get("task_id", ""))

        @sio.on("task_manager_remove")
        def _on_remove(data):
            self._remove_queued(data.get("task_id", ""))

        @sio.on("task_manager_submit_test")
        def _on_submit_test(data):
            self._submit_test_task(data)

    # ------------------------------------------------------------------
    # Task control helpers
    # ------------------------------------------------------------------

    def _cancel_task(self, task_id: str) -> None:
        with self._lock:
            if self._active and self._active.id == task_id:
                self._active.cancel_event.set()
                self._active.pause_event.set()  # unblock if paused
                return
            for t in self._queue:
                if t.id == task_id:
                    t.status = "cancelled"
                    t.finished_at = time.time()
                    break
        # Remove cancelled items from queue
        self._flush_cancelled()
        self._broadcast()

    def _pause_task(self, task_id: str) -> None:
        with self._lock:
            if self._active and self._active.id == task_id:
                self._active.pause_event.clear()
                self._active.status = "paused"
        self._broadcast()

    def _resume_task(self, task_id: str) -> None:
        with self._lock:
            if self._active and self._active.id == task_id:
                self._active.status = "running"
                self._active.pause_event.set()
        self._broadcast()

    def _remove_queued(self, task_id: str) -> None:
        with self._lock:
            self._queue = deque(t for t in self._queue if t.id != task_id)
        self._broadcast()

    def _flush_cancelled(self) -> None:
        with self._lock:
            removed = [t for t in self._queue if t.status == "cancelled"]
            self._queue = deque(t for t in self._queue if t.status != "cancelled")
        for t in removed:
            self._history.appendleft(t.to_dict())

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            self._work_available.wait()
            task = self._pick_next()
            if task is None:
                self._work_available.clear()
                continue
            self._run_task(task)

    def _pick_next(self) -> Task | None:
        with self._lock:
            if self._queue:
                return self._queue.popleft()
            self._work_available.clear()
            return None

    def _run_task(self, task: Task) -> None:
        with self._lock:
            self._active = task
            task.status = "running"
            task.started_at = time.time()
            task.message = "Running…"
        self._broadcast()

        ctx = TaskContext(task, self)
        try:
            with self.app.app_context():
                task.fn(ctx, *task.args, **task.kwargs)
            task.status = "completed"
            task.progress = 1.0
            task.message = "Completed"
        except TaskCancelled:
            task.status = "cancelled"
            task.message = "Cancelled by user"
        except Exception as exc:
            task.status = "failed"
            task.error = traceback.format_exc()
            task.message = f"Error: {exc}"
        finally:
            task.finished_at = time.time()
            with self._lock:
                self._active = None
                self._history.appendleft(task.to_dict())
            self._broadcast()
            # Nudge worker to check for the next item
            if self._queue:
                self._work_available.set()

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def _broadcast(self) -> None:
        self.socketio.emit("task_manager_update", self.get_state())

    # ------------------------------------------------------------------
    # Built-in test / toy tasks
    # ------------------------------------------------------------------

    def _submit_test_task(self, data: dict) -> None:
        kind = data.get("type", "counter")
        if kind == "counter":
            count = int(data.get("count", 20))
            self.submit(f"Test: count to {count}", _toy_counter, args=(count,))
        elif kind == "fail":
            self.submit("Test: failing task", _toy_fail)
        elif kind == "instant":
            self.submit("Test: instant task", _toy_instant)


# ------------------------------------------------------------------
# Toy task functions (for manual testing)
# ------------------------------------------------------------------

def _toy_counter(ctx: TaskContext, count: int) -> None:
    """Count to *count* with 0.5 s per step."""
    for i in range(count):
        ctx.check()
        time.sleep(0.5)
        ctx.update((i + 1) / count, f"Step {i + 1}/{count}")


def _toy_fail(ctx: TaskContext) -> None:
    """Succeed for 3 steps then raise."""
    for i in range(3):
        ctx.check()
        time.sleep(0.4)
        ctx.update((i + 1) / 5, f"Step {i + 1}/5")
    raise RuntimeError("Simulated failure for testing")


def _toy_instant(ctx: TaskContext) -> None:
    """Complete immediately."""
    ctx.update(1.0, "Done instantly")
