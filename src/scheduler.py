"""Periodic background scheduler with Task Manager UI integration.

Usage::

    from src.scheduler import Scheduler

    # Starting a scheduler is just creating an instance.
    Scheduler(
        app,
        interval_minutes=cfg.my_module.update_interval_minutes,
        fn=_check_and_submit_work,
        name='MyModule: do work',
        check_fn=lambda: len(pending_items()) > 0,
    )

The scheduler daemon thread starts immediately on instantiation.  If
*interval_minutes* is falsy the instance is inert (no thread, not visible
in the UI) — this is the intended way to disable a scheduler via config.
"""
from __future__ import annotations

import threading
import time
import uuid
from typing import Callable, Optional


class Scheduler:
    """Background periodic task with pause/resume and Task Manager UI visibility.

    The cycle is: initial sleep → check → fire → wait for submitted task →
    sleep → …  The interval is a *cooldown* that begins only after the
    previous task completes, so long-running tasks never cause duplicate queuing.

    *check_fn* is called inside an ``app.app_context()`` before *fn*.  If it
    returns a falsy value the cycle is skipped and the cooldown restarts
    silently without submitting any task.  Defaults to ``lambda: True``.

    *fn* is called with no arguments inside an ``app.app_context()`` and should
    return the task id (str) submitted to ``app.task_manager``, or ``None``.

    Pausing freezes the countdown; resuming continues from where it left off.
    While paused, *fn* is never called.

    If *interval_minutes* is falsy the instance is inert: no thread is started
    and it is not registered in the UI.  This is the intended way to disable a
    scheduler via config.
    """

    # Minimum seconds between throttled UI broadcasts during cooldown.
    _BROADCAST_INTERVAL = 5.0

    def __init__(
        self,
        app,
        interval_minutes: float,
        fn: Callable,
        name: str = '',
        check_fn: Optional[Callable] = None,
        start_immediately: bool = False,
    ) -> None:
        self.id = uuid.uuid4().hex
        self.name = name or 'Scheduler'
        self.interval_minutes = float(interval_minutes) if interval_minutes else 0.0

        self._paused: bool = False
        # Wall-clock timestamp of the next scheduled fire; None = running or unknown.
        self._next_run_at: float | None = None
        self._manager = None   # set by TaskManager.register_scheduler
        self._lock = threading.Lock()
        self._last_broadcast: float = 0.0

        if not interval_minutes:
            return  # disabled: no thread, not registered in the UI

        self._app = app
        self._fn = fn
        self._check_fn = check_fn if check_fn is not None else lambda: True
        self._interval_seconds: float = interval_minutes * 60
        self._start_immediately = start_immediately

        if hasattr(app, 'task_manager'):
            app.task_manager.register_scheduler(self)

        threading.Thread(
            target=self._loop,
            daemon=True,
            name=f'scheduler-{name}',
        ).start()

    # ---- public control (called via socket events) --------------------

    def pause(self) -> None:
        with self._lock:
            if self._paused:
                return
            self._paused = True
            # Freeze _next_run_at in place so the UI displays the frozen value.
        if self._manager:
            self._manager._broadcast()

    def resume(self) -> None:
        with self._lock:
            if not self._paused:
                return
            self._paused = False
            if self._next_run_at is not None:
                # _next_run_at was frozen at pause time; shift it forward so the
                # remaining gap is preserved relative to now.
                frozen_remaining = self._next_run_at - time.time()
                if frozen_remaining < 0:
                    frozen_remaining = 0.0
                self._next_run_at = time.time() + frozen_remaining
        if self._manager:
            self._manager._broadcast()

    def get_state(self) -> dict:
        with self._lock:
            return {
                'id': self.id,
                'name': self.name,
                'interval_minutes': self.interval_minutes,
                'paused': self._paused,
                'next_run_at': self._next_run_at,
            }

    # ---- internal helpers called by the loop -------------------------

    def _set_cooldown(self, remaining_seconds: float) -> None:
        with self._lock:
            if not self._paused:
                self._next_run_at = time.time() + remaining_seconds
        if self._manager:
            self._manager._broadcast()

    def _update_remaining(self, remaining_seconds: float) -> None:
        """Throttled update to keep the UI countdown accurate."""
        now = time.time()
        with self._lock:
            if self._paused:
                return
            self._next_run_at = now + max(0.0, remaining_seconds)
            if now - self._last_broadcast < self._BROADCAST_INTERVAL:
                return
            self._last_broadcast = now
        if self._manager:
            self._manager._broadcast()

    def _set_running(self) -> None:
        """Clear the countdown just before fn() fires (shows 'Running…' in UI)."""
        with self._lock:
            self._next_run_at = None
        if self._manager:
            self._manager._broadcast()

    # ---- private: thread loop ----------------------------------------

    def _sleep_with_pause(self, total_seconds: float) -> None:
        """Sleep for *total_seconds*, freezing elapsed time while paused."""
        remaining = total_seconds
        last_tick = time.time()
        self._set_cooldown(total_seconds)
        while remaining > 0:
            time.sleep(0.3)
            now = time.time()
            elapsed = now - last_tick
            last_tick = now
            if not self._paused:
                remaining -= elapsed
                self._update_remaining(remaining)

    def _loop(self) -> None:
        if not self._start_immediately:
            self._sleep_with_pause(self._interval_seconds)

        while True:
            # Wait out any pause before checking / firing.
            while self._paused:
                time.sleep(0.3)

            with self._app.app_context():
                should_run = self._check_fn()

            if should_run:
                self._set_running()
                with self._app.app_context():
                    task_id = self._fn()
                if task_id and hasattr(self._app, 'task_manager'):
                    self._app.task_manager.wait_for_task(task_id)

            self._sleep_with_pause(self._interval_seconds)
