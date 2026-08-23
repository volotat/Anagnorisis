"""Reporting progress without knowing who is listening.

Long operations here take a *context* with two methods: ``check()``, which
raises or blocks if the caller wants to stop, and ``update(fraction, message)``.
That is deliberately the shape the Flask app's ``TaskContext`` already has, so
the app can hand its own object straight in and its work stays pausable from the
Task Manager. The command line passes ``PrintProgress``; tests pass nothing.

The point is that the engine never imports a socket, a queue or a task manager
to say how far along it is.
"""
import sys
import time


class NullProgress:
    """Reports nowhere. The default, so progress is never a required argument."""

    def check(self) -> None:
        return None

    def update(self, progress: float = 0.0, message: str = "") -> None:
        return None


class PrintProgress:
    """Reports to a terminal, one line rewritten in place.

    Throttled because describing a file takes seconds and a redraw per file is
    plenty; a progress bar that outpaces the work is just noise.
    """

    def __init__(self, stream=sys.stderr, min_interval: float = 0.2):
        self._stream = stream
        self._min_interval = min_interval
        self._last = 0.0

    def check(self) -> None:
        return None

    def update(self, progress: float = 0.0, message: str = "") -> None:
        now = time.monotonic()
        if now - self._last < self._min_interval and progress < 1.0:
            return
        self._last = now
        pct = f"{progress * 100:5.1f}%"
        self._stream.write(f"\r{pct}  {message[:100]:<100}")
        self._stream.flush()

    def done(self, message: str = "") -> None:
        self._stream.write(f"\r{message[:110]:<110}\n")
        self._stream.flush()


# ---------------------------------------------------------------------------
# Adapters for callers that report through a status callable
#
# These predate NullProgress/PrintProgress and serve a different shape: the app
# hands them a function that pushes a line to the browser, and they decide how
# often to call it. Same idea, opposite direction — the engine still does not
# know who is listening.
# ---------------------------------------------------------------------------

class SortingProgressCallback:
    def __init__(self, show_status_function=None, operation_name="Sorting"):
        self.last_shown_time = 0
        self.start_time = time.time()
        self.show_status_function = show_status_function
        self.operation_name = operation_name

    def __call__(self, num_processed, num_total):
        current_time = time.time()
        if current_time - self.last_shown_time >= 1:  
            # Calculate the percentage of processed items
            percent = (num_processed / num_total) * 100

            # Show the status
            self.show_status_function(f"{self.operation_name} {percent:.2f}% ({num_processed}/{num_total} files)")
            self.last_shown_time = current_time


###########################################
# Embedding Gathering Progress Callback

class EmbeddingGatheringCallback:
    def __init__(self, show_status_function=None, name=""):
        self.last_shown_time = 0
        self.start_time = time.time()
        self.show_status_function = show_status_function
        self.name = name

    def __call__(self, num_extracted, num_total):
        current_time = time.time()
        if current_time - self.last_shown_time >= 1:
            # Calculate the percentage of processed files
            percent = (num_extracted / num_total) * 100

            # Show the status
            self.show_status_function(f"Resolving {self.name} embeddings for {num_extracted}/{num_total} ({percent:.2f}%) files.")
            self.last_shown_time = current_time


###########################################
# Arbitrary Progress Callback

class ArbitraryProgressCallback:
    def __init__(self, show_status_function=None, name=""):
        self.last_shown_time = 0
        self.start_time = time.time()
        self.show_status_function = show_status_function
        self.name = name

    def __call__(self, msg):
        current_time = time.time()
        if current_time - self.last_shown_time >= 1:
            # Show the status
            self.show_status_function(msg)
            self.last_shown_time = current_time
