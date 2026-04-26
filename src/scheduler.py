import threading
import time


def schedule_task(app, interval_minutes: float, fn) -> None:
    """Start a daemon thread that calls *fn* every *interval_minutes* minutes.

    The cycle is: initial sleep → fire → wait for the submitted task to finish
    → sleep → fire → …  This means the interval is a *cooldown* that starts
    only after the previous task completes, so long-running tasks never cause
    duplicate queuing.

    *fn* is called with no arguments inside an ``app.app_context()`` and should
    return the task id (str) if it submitted a task to ``app.task_manager``, or
    None / nothing if it decided there was no work to do.

    *interval_minutes* must be a positive number; passing 0 or None is a no-op.
    """
    if not interval_minutes:
        return

    def _loop():
        while True:
            time.sleep(interval_minutes * 60)
            with app.app_context():
                task_id = fn()
            # If a task was submitted, block until it leaves the active/queued state
            # before starting the next cooldown interval.
            if task_id and hasattr(app, 'task_manager'):
                app.task_manager.wait_for_task(task_id)

    threading.Thread(target=_loop, daemon=True, name="scheduler").start()
