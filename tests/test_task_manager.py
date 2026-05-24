"""
Tests for src/task_manager.py (TaskManager, TaskContext, Task)

TaskManager requires a Flask app + SocketIO instance. We create minimal stubs
so no network or GPU is needed.

Covers:
  - Task submitted and executed in order (FIFO queue)
  - Multiple tasks run sequentially (single worker)
  - ctx.check() raises TaskCancelled when cancel requested
  - Cancelling a running task stops it within one check() cycle
  - Cancelling a queued-but-not-started task marks it cancelled immediately
  - Pausing a task blocks execution; resuming unblocks it
  - Exception inside a task sets status='failed' and does not crash the worker
  - Completed tasks appear in get_history()
  - wait_for_task() blocks until the task finishes
"""
import time
import threading
import pytest
from unittest.mock import MagicMock, patch

from src.task_manager import TaskManager, TaskCancelled


# ---------------------------------------------------------------------------
# Fixtures — minimal Flask/SocketIO stubs
# ---------------------------------------------------------------------------

def _make_task_manager():
    """Return a TaskManager backed by mock Flask app and SocketIO."""
    mock_socketio = MagicMock()
    mock_socketio.emit = MagicMock()

    mock_app = MagicMock()
    # app_context() must be a usable context manager
    mock_app.app_context.return_value.__enter__ = MagicMock(return_value=None)
    mock_app.app_context.return_value.__exit__  = MagicMock(return_value=False)

    tm = TaskManager(socketio=mock_socketio, app=mock_app)
    return tm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTaskManagerBasics:
    def test_submit_returns_task_id(self):
        tm = _make_task_manager()
        def noop(ctx): pass
        task_id = tm.submit('noop', noop)
        assert isinstance(task_id, str) and len(task_id) > 0

    def test_task_executes_and_completes(self):
        tm = _make_task_manager()
        done = threading.Event()

        def simple_task(ctx):
            done.set()

        task_id = tm.submit('simple', simple_task)
        assert done.wait(timeout=5), "Task never executed"
        tm.wait_for_task(task_id, poll_interval=0.05)

        state = tm.get_state()
        history_ids = [t['id'] for t in state['history']]
        assert task_id in history_ids

        completed_task = next(t for t in state['history'] if t['id'] == task_id)
        assert completed_task['status'] == 'completed'

    def test_tasks_execute_in_fifo_order(self):
        tm = _make_task_manager()
        order = []
        barrier = threading.Barrier(2)

        def task_a(ctx):
            order.append('A')

        def task_b(ctx):
            order.append('B')

        # Submit both; worker is single-threaded so A should run first
        id_a = tm.submit('A', task_a)
        id_b = tm.submit('B', task_b)
        tm.wait_for_task(id_b, poll_interval=0.05)

        assert order == ['A', 'B'], f"Expected FIFO order A→B, got {order}"

    def test_exception_in_task_sets_failed_status(self):
        tm = _make_task_manager()

        def failing_task(ctx):
            raise RuntimeError("boom")

        task_id = tm.submit('failing', failing_task)
        tm.wait_for_task(task_id, poll_interval=0.05)

        state = tm.get_state()
        failed = next(t for t in state['history'] if t['id'] == task_id)
        assert failed['status'] == 'failed'
        assert 'boom' in failed.get('error', '')

    def test_exception_does_not_crash_worker(self):
        """Worker must keep processing tasks after a task raises."""
        tm = _make_task_manager()
        done = threading.Event()

        def failing_task(ctx):
            raise ValueError("intentional")

        def followup_task(ctx):
            done.set()

        tm.submit('fail', failing_task)
        id_followup = tm.submit('followup', followup_task)
        assert done.wait(timeout=5), "Worker did not process task after a failure"

    def test_completed_task_in_history(self):
        tm = _make_task_manager()

        def task(ctx):
            ctx.update(0.5, 'halfway')

        task_id = tm.submit('history_test', task)
        tm.wait_for_task(task_id, poll_interval=0.05)

        ids = [t['id'] for t in tm.get_state()['history']]
        assert task_id in ids


class TestTaskManagerCancel:
    def test_cancel_running_task(self):
        tm = _make_task_manager()
        started = threading.Event()
        cancelled = threading.Event()

        def long_task(ctx):
            started.set()
            for _ in range(200):
                ctx.check()  # will raise TaskCancelled when cancelled
                time.sleep(0.02)

        task_id = tm.submit('long', long_task)
        assert started.wait(timeout=5), "Task never started"
        tm._cancel_task(task_id)
        tm.wait_for_task(task_id, poll_interval=0.05)

        state = tm.get_state()
        entry = next((t for t in state['history'] if t['id'] == task_id), None)
        assert entry is not None
        assert entry['status'] == 'cancelled'

    def test_cancel_queued_task(self):
        """Cancelling a task that hasn't started yet should not execute it."""
        tm = _make_task_manager()
        executed = threading.Event()

        # Block the worker with a slow first task
        slow_started = threading.Event()

        def slow_task(ctx):
            slow_started.set()
            time.sleep(0.3)

        def never_task(ctx):
            executed.set()

        tm.submit('slow', slow_task)
        assert slow_started.wait(timeout=5)

        id_never = tm.submit('never', never_task)
        tm._cancel_task(id_never)

        # Wait a bit then confirm it was never executed
        time.sleep(0.05)
        assert not executed.is_set(), "Cancelled queued task should not have executed"


class TestTaskManagerPauseResume:
    def test_pause_blocks_progress(self):
        tm = _make_task_manager()
        steps = []
        task_running = threading.Event()

        def pausable_task(ctx):
            task_running.set()
            for i in range(5):
                ctx.check()
                steps.append(i)
                time.sleep(0.05)

        task_id = tm.submit('pausable', pausable_task)
        assert task_running.wait(timeout=5)
        time.sleep(0.06)   # let at least one step complete
        steps_before = len(steps)
        tm._pause_task(task_id)
        time.sleep(0.15)   # while paused, no new steps should appear
        steps_while_paused = len(steps)
        tm._resume_task(task_id)
        tm.wait_for_task(task_id, poll_interval=0.05)

        # Steps should have stopped increasing during the pause window
        assert steps_while_paused - steps_before <= 1, (
            f"Expected progress to stop while paused. "
            f"Before: {steps_before}, During pause: {steps_while_paused}"
        )

    def test_resume_allows_completion(self):
        tm = _make_task_manager()
        done = threading.Event()
        inside_loop = threading.Event()

        def task(ctx):
            # Loop so that pause() is guaranteed to take effect before done is set.
            for _ in range(200):
                inside_loop.set()
                ctx.check()   # blocks here while paused
                time.sleep(0.01)
            done.set()

        task_id = tm.submit('resume_test', task)
        # Wait until the task is spinning inside its check() loop
        assert inside_loop.wait(timeout=5)
        tm._pause_task(task_id)
        time.sleep(0.15)  # long enough for several loop iterations to be blocked
        assert not done.is_set(), "Task should still be blocked while paused"
        tm._resume_task(task_id)
        assert done.wait(timeout=5), "Task should complete after resume"


class TestTaskManagerGetState:
    def test_get_state_structure(self):
        tm = _make_task_manager()
        state = tm.get_state()
        assert 'active' in state
        assert 'queued' in state
        assert 'history' in state
        assert 'schedulers' in state

    def test_active_is_none_when_idle(self):
        tm = _make_task_manager()
        # Give worker a moment to be truly idle
        time.sleep(0.1)
        assert tm.get_state()['active'] is None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
