"""Subprocess proxy for the UniversalEvaluator (TransformerEvaluator).

The evaluator model lives entirely inside a dedicated subprocess so that:
  - The main Flask process never initialises a CUDA context.
  - VRAM is released ~120 s after the last evaluation call (idle timeout).
  - Training also runs in the subprocess; progress is streamed back via the
    shared output queue so the UI progress bar keeps updating in real time.

Public interface (proxy class, thread-safe):
    load(model_path)           — load weights from disk; mirrors .hash
    predict(X)                 — np.ndarray of predicted ratings
    reinitialize()             — reset model to random weights
    save(model_path)           — persist current weights to disk
    evaluate(...)              — (train_acc, test_acc) without weight updates
    train_full(...)            — full epoch loop; streams progress via callback
    .hash                      — MD5 of loaded .pt file, or None
    .mape_bias                 — constant 2 (same as TransformerEvaluator)
"""

import os
import time
import traceback
import threading
import multiprocessing
import queue

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Worker implementation (everything below runs in the subprocess only)
# ─────────────────────────────────────────────────────────────────────────────

class _UniversalEvaluatorImpl:
    """
    Owns the actual TransformerEvaluator and executes all torch / CUDA ops.
    Instantiated once per worker process.
    """

    def __init__(self):
        # Deferred imports: these happen in the subprocess, not the main process.
        import torch
        from src.scoring_models import TransformerEvaluator, configure_device

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        configure_device(device)
        print(f"[UniversalEvaluator Worker] Initialised on {device}")

        self.evaluator = TransformerEvaluator(
            embedding_dim=1024, rate_classes=11, name="UniversalEvaluator"
        )

    def load(self, model_path: str) -> dict:
        self.evaluator.load(model_path)
        return {'hash': self.evaluator.hash, 'mape_bias': self.evaluator.mape_bias}

    def save(self, model_path: str) -> bool:
        self.evaluator.save(model_path)
        return True

    def reinitialize(self) -> bool:
        self.evaluator.reinitialize()
        return True

    def predict(self, X) -> np.ndarray:
        return self.evaluator.predict(X)

    def evaluate(self, X_train, y_train, X_test, y_test, batch_size: int = 16):
        return self.evaluator.evaluate(X_train, y_train, X_test, y_test, batch_size)

    def train_full(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        sample_weights,
        total_epochs: int,
        batch_size: int,
        time_budget_seconds,
        model_save_path: str,
        output_queue,
    ) -> dict:
        """
        Run the full training loop.  Progress messages are streamed to
        *output_queue* as ``('progress', {...})`` tuples.  A final summary
        dict is returned; the caller is responsible for emitting
        ``('success', summary)`` afterwards.
        """
        import time as _time

        best_train_accuracy = 0.0
        best_test_accuracy = 0.0
        best_epoch = 0

        # Epoch-0 evaluation (before any weight updates)
        train_acc, test_acc = self.evaluator.evaluate(
            X_train, y_train, X_test, y_test, batch_size=batch_size
        )
        output_queue.put(('progress', {
            'type': 'initial_eval',
            'train_acc': train_acc,
            'test_acc': test_acc,
        }))

        training_start = _time.time()

        for epoch in range(total_epochs):
            train_acc, test_acc = self.evaluator.train(
                X_train, y_train, X_test, y_test,
                batch_size=batch_size,
                sample_weights=sample_weights,
            )

            output_queue.put(('progress', {
                'type': 'epoch',
                'epoch': epoch,
                'total_epochs': total_epochs,
                'train_acc': train_acc,
                'test_acc': test_acc,
            }))

            if test_acc > best_test_accuracy:
                best_train_accuracy = train_acc
                best_test_accuracy = test_acc
                best_epoch = epoch + 1
                self.evaluator.save(model_save_path)

            if time_budget_seconds is not None:
                if _time.time() - training_start >= time_budget_seconds:
                    print(
                        f"[UniversalEvaluator Worker] Time budget of "
                        f"{time_budget_seconds}s reached after epoch {epoch + 1}."
                    )
                    break

        # Reload the best checkpoint so subsequent predict() calls use it.
        if best_epoch > 0 and os.path.exists(model_save_path):
            self.evaluator.load(model_save_path)

        return {
            'best_epoch': best_epoch,
            'best_train_accuracy': best_train_accuracy,
            'best_test_accuracy': best_test_accuracy,
            'hash': self.evaluator.hash,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Worker loop
# ─────────────────────────────────────────────────────────────────────────────

import setproctitle  # noqa: E402 — available in requirements.txt


def _worker_loop(input_queue, output_queue):
    setproctitle.setproctitle("Anagnorisis-UniversalEvaluator")

    try:
        impl = _UniversalEvaluatorImpl()
    except Exception as exc:
        traceback.print_exc()
        output_queue.put(('error', exc))
        return

    while True:
        try:
            task = input_queue.get()
            if task is None:
                break

            command, args, kwargs = task

            if command == 'train_full':
                # train_full streams progress; inject the live output_queue.
                try:
                    result = impl.train_full(*args, **kwargs, output_queue=output_queue)
                    output_queue.put(('success', result))
                except Exception as exc:
                    traceback.print_exc()
                    output_queue.put(('error', exc))
            else:
                try:
                    method = getattr(impl, command)
                    result = method(*args, **kwargs)
                    output_queue.put(('success', result))
                except Exception as exc:
                    traceback.print_exc()
                    output_queue.put(('error', exc))

        except Exception as exc:
            traceback.print_exc()
            output_queue.put(('error', exc))


# ─────────────────────────────────────────────────────────────────────────────
# Proxy class (runs in the main process only)
# ─────────────────────────────────────────────────────────────────────────────

class UniversalEvaluator:
    """
    Singleton proxy for the UniversalEvaluator subprocess.

    All heavy operations (model forward pass, training) are routed to a
    worker subprocess so that no CUDA context is ever created in the main
    Flask process.  The subprocess is started lazily on first use and
    terminated automatically after 120 s of inactivity to free VRAM.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._lock = threading.Lock()

        # Mirrored attributes (updated from subprocess results)
        self.hash = None
        self.mape_bias = 2  # constant — no subprocess call needed

        # Stored for transparent subprocess restart
        self._last_loaded_path: str = None

        # Idle management
        self._last_used_time: float = 0.0
        self._idle_timeout: int = 120  # seconds
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(
            target=self._monitor_idle, daemon=True
        )
        self._monitor_thread.start()

        self._initialized = True

    # ── idle management ──────────────────────────────────────────────────────

    def _monitor_idle(self):
        while not self._shutdown_event.is_set():
            time.sleep(5)
            with self._lock:
                if self._process is not None and self._process.is_alive():
                    if (
                        self._last_used_time > 0
                        and time.time() - self._last_used_time > self._idle_timeout
                    ):
                        print(
                            f"UniversalEvaluator: Idle for {self._idle_timeout}s. "
                            "Terminating subprocess to free GPU."
                        )
                        self._terminate_process()

    def _terminate_process(self):
        """Gracefully (then forcefully) kill the worker process."""
        if self._process:
            try:
                self._input_queue.put(None)
                self._process.join(timeout=1)
            except Exception:
                pass
            if self._process.is_alive():
                print("UniversalEvaluator: Force killing subprocess...")
                self._process.terminate()
                self._process.join()
            self._process = None
            self._input_queue = None
            self._output_queue = None
            import gc
            gc.collect()

    def unload(self):
        """Immediately terminate the worker subprocess to free GPU memory."""
        with self._lock:
            self._terminate_process()
        print("UniversalEvaluator: Subprocess terminated (hash preserved).")

    def _ensure_process_running(self):
        """
        Start the subprocess if it is not running.
        Must be called while holding ``self._lock``.
        If the process was previously using a loaded model, it is re-loaded
        transparently in the new subprocess.
        """
        if self._process is None or not self._process.is_alive():
            print("UniversalEvaluator: Starting worker subprocess...")
            ctx = multiprocessing.get_context('spawn')
            self._input_queue = ctx.Queue()
            self._output_queue = ctx.Queue()
            self._process = ctx.Process(
                target=_worker_loop,
                args=(self._input_queue, self._output_queue),
                name="Anagnorisis-UniversalEvaluator",
            )
            self._process.start()

            # Re-load previously loaded model if path is known
            if self._last_loaded_path and os.path.exists(self._last_loaded_path):
                print("UniversalEvaluator: Re-loading model in new subprocess...")
                result = self._send_command_internal('load', (self._last_loaded_path,), {})
                self.hash = result.get('hash', self.hash)

    def _send_command_internal(self, command, args, kwargs):
        """Send a command and wait for a single reply.  Lock must be held."""
        self._input_queue.put((command, args, kwargs))
        while True:
            try:
                status, result = self._output_queue.get(timeout=5)
                break
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    exit_code = self._process.exitcode if self._process else None
                    self._terminate_process()
                    raise RuntimeError(
                        f"UniversalEvaluator subprocess died unexpectedly during "
                        f"'{command}' (exit code: {exit_code})."
                    )
                # Still alive — keep waiting.
        if status == 'error':
            raise result
        return result

    def _execute(self, command, *args, **kwargs):
        """Execute a command and return the result (holds lock for duration)."""
        with self._lock:
            self._ensure_process_running()
            result = self._send_command_internal(command, args, kwargs)
            self._last_used_time = time.time()
            return result

    def _execute_training(self, command, *args, progress_callback=None, **kwargs):
        """
        Execute a long-running command that streams ``('progress', data)``
        messages before emitting a terminal ``('success', result)`` or
        ``('error', exc)``.

        For each progress message, *progress_callback(data)* is called
        in the main process (useful for live UI updates).
        """
        with self._lock:
            self._ensure_process_running()
            self._input_queue.put((command, args, kwargs))

            while True:
                try:
                    status, data = self._output_queue.get(timeout=5)
                except queue.Empty:
                    if self._process is None or not self._process.is_alive():
                        exit_code = self._process.exitcode if self._process else None
                        self._terminate_process()
                        raise RuntimeError(
                            f"UniversalEvaluator subprocess died during training "
                            f"(exit code: {exit_code})."
                        )
                    continue  # Still alive — keep waiting for next progress message.

                if status == 'progress':
                    if progress_callback is not None:
                        progress_callback(data)
                elif status == 'success':
                    self._last_used_time = time.time()
                    return data
                elif status == 'error':
                    raise data

    # ── public interface ─────────────────────────────────────────────────────

    def load(self, model_path: str):
        """Load model weights from *model_path* and mirror ``.hash``."""
        self._last_loaded_path = model_path
        result = self._execute('load', model_path)
        self.hash = result.get('hash')

    def save(self, model_path: str):
        """Persist current model weights to *model_path*."""
        self._execute('save', model_path)

    def reinitialize(self):
        """Reset model to random weights (discards any loaded checkpoint)."""
        self._execute('reinitialize')

    def predict(self, X) -> np.ndarray:
        """
        Predict ratings for a list of variable-length embedding sequences.

        Parameters
        ----------
        X : list of np.ndarray  — each of shape [S_i, embedding_dim]

        Returns
        -------
        np.ndarray of shape [len(X)] with predicted float ratings
        """
        return self._execute('predict', X)

    def evaluate(self, X_train, y_train, X_test, y_test, batch_size: int = 16):
        """Evaluate on both splits without updating model weights."""
        return self._execute('evaluate', X_train, y_train, X_test, y_test, batch_size)

    def train_full(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        sample_weights,
        total_epochs: int,
        batch_size: int,
        time_budget_seconds,
        model_save_path: str,
        progress_callback=None,
    ) -> dict:
        """
        Run the full training loop inside the subprocess.

        Progress is relayed to *progress_callback(data)* where ``data`` is:
          ``{'type': 'initial_eval', 'train_acc': float, 'test_acc': float}``
          ``{'type': 'epoch', 'epoch': int, 'total_epochs': int,
              'train_acc': float, 'test_acc': float}``

        Returns a summary dict with keys:
          ``best_epoch``, ``best_train_accuracy``, ``best_test_accuracy``, ``hash``

        After completion, ``.hash`` is updated and subsequent ``predict()``
        calls will use the best checkpoint weights.
        """
        result = self._execute_training(
            'train_full',
            X_train, y_train, X_test, y_test,
            sample_weights, total_epochs, batch_size,
            time_budget_seconds, model_save_path,
            progress_callback=progress_callback,
        )
        # Mirror the hash so serve.py hash checks stay valid after training.
        self.hash = result.get('hash', self.hash)
        self._last_loaded_path = model_save_path
        return result

    def __del__(self):
        self._shutdown_event.set()
        self._terminate_process()


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (run directly: python src/universal_evaluator.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    print("UniversalEvaluator self-test")
    print("=" * 50)

    evaluator = UniversalEvaluator()
    print("Proxy created (subprocess not yet started).")

    # Create dummy embeddings (5 samples, 3 chunks each, 1024-dim)
    rng = np.random.default_rng(42)
    X_dummy = [rng.standard_normal((3, 1024)).astype(np.float32) for _ in range(5)]
    y_dummy = [5.0, 7.0, 3.0, 8.0, 2.0]

    # Predict before any training (random weights → garbage output, just tests IPC)
    print("\n[Test] predict() on random weights...")
    preds = evaluator.predict(X_dummy)
    print(f"  predictions: {preds}")
    assert isinstance(preds, np.ndarray) and len(preds) == 5, "predict() shape mismatch"
    print("  ✓ predict() OK")

    # Quick train_full test with 3 epochs
    print("\n[Test] train_full() for 3 epochs...")
    save_path = '/tmp/_universal_evaluator_test.pt'

    progress_log = []

    def on_progress(data):
        progress_log.append(data)
        print(f"  progress: {data}")

    result = evaluator.train_full(
        X_dummy, y_dummy, X_dummy, y_dummy,
        sample_weights=None,
        total_epochs=3,
        batch_size=4,
        time_budget_seconds=None,
        model_save_path=save_path,
        progress_callback=on_progress,
    )
    print(f"  result: {result}")
    assert 'best_epoch' in result, "train_full() result missing 'best_epoch'"
    assert len(progress_log) == 4, f"Expected 4 progress messages (1 initial + 3 epochs), got {len(progress_log)}"
    print("  ✓ train_full() OK")

    # Verify hash is mirrored
    assert evaluator.hash is not None, "hash should be set after train_full"
    print(f"  hash: {evaluator.hash}")
    print("  ✓ hash mirroring OK")

    # Idle timeout test
    print("\n[Test] subprocess is running...")
    assert evaluator._process is not None and evaluator._process.is_alive()
    print("  ✓ subprocess alive")

    print("\nAll tests passed!")
    sys.exit(0)
