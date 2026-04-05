import threading
import time


def schedule_task(app, interval_minutes: float, name: str, fn) -> None:
    """Start a daemon thread that submits *fn* to the TaskManager every *interval_minutes*.

    The task is only queued if no task with the same *name* is already active
    or queued, preventing pile-ups when a run takes longer than the interval.
    *interval_minutes* must be a positive number; passing 0 or None is a no-op.
    """
    if not interval_minutes:
        return

    def _loop():
        print(f"[Scheduler] Starting '{name}' with interval {interval_minutes} minutes")
        while True:
            time.sleep(interval_minutes * 60)
            state = app.task_manager.get_state()
            active = state['active']
            already = (
                (active and active.get('name') == name) or
                any(t.get('name') == name for t in state['queued'])
            )
            if not already:
                app.task_manager.submit(name, fn)

    threading.Thread(target=_loop, daemon=True, name=f"scheduler:{name}").start()
