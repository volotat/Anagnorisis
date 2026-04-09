import threading
import time


def schedule_task(app, interval_minutes: float, fn) -> None:
    """Start a daemon thread that calls *fn* every *interval_minutes* minutes.

    *fn* is called with no arguments inside an ``app.app_context()`` so it can
    perform DB queries and use Flask extensions freely.  Whatever *fn* decides
    to do (submit a task, skip, log, etc.) is entirely its own concern.

    *interval_minutes* must be a positive number; passing 0 or None is a no-op.
    """
    if not interval_minutes:
        return

    def _loop():
        while True:
            time.sleep(interval_minutes * 60)
            with app.app_context():
                fn()

    threading.Thread(target=_loop, daemon=True, name="scheduler").start()
