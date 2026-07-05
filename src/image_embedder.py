import os
import time
import traceback
import threading
import multiprocessing
import queue
import hashlib
from typing import List

import torch
import numpy as np
import src.virtual_file_system as vfs

# --- The Worker Implementation (Runs in separate process) ---

class _ImageEmbedderImpl:
    """
    The actual implementation that runs inside the subprocess.
    It holds the SigLIP model and CUDA context.
    Converts images into L2-normalised embedding vectors.
    """
    def __init__(self, cfg):
        # Heavy imports are deferred to the subprocess to avoid initialising
        # CUDA in the main process.
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.embedding_dim = None
        self.model_hash = None

    def initiate(self, models_folder: str):
        if self.model is not None:
            return

        model_name = self.cfg.image_embedder.model_name
        if not model_name:
            raise ValueError("cfg.image_embedder.model_name is not specified.")

        model_folder_name = model_name.replace('/', '__')
        local_model_path = os.path.join(models_folder, model_folder_name)

        self._ensure_model_downloaded(models_folder, model_name)
        self._load_model_and_processor(local_model_path)
        self.model_hash = self._calculate_model_hash()
        print(f"ImageEmbedder (Worker): Initiated. Embedding dim: {self.embedding_dim} on {self.device}")

    def _calculate_model_hash(self) -> str:
        print("ImageEmbedder (Worker): Calculating model hash...")
        try:
            md5 = hashlib.md5()
            for k, v in sorted(self.model.state_dict().items()):
                md5.update(k.encode('utf-8'))
                md5.update(str(v.shape).encode('utf-8'))
                flat_v = v.view(-1)
                sample = flat_v[:100].tolist()
                md5.update(str(sample).encode('utf-8'))
            return md5.hexdigest()
        except Exception as e:
            print(f"Error calculating model hash: {e}")
            return "unknown_hash"

    def _ensure_model_downloaded(self, models_folder: str, model_name: str):
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig
        import glob

        self.model_name = model_name
        local_model_path = os.path.join(models_folder, model_name.replace('/', '__'))
        config_file_path = os.path.join(local_model_path, 'config.json')

        def download(force=False):
            print(f"{'Re-downloading' if force else 'Downloading'} model '{model_name}' to '{local_model_path}'...")
            snapshot_download(
                repo_id=model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                force_download=force,
                resume_download=True,
            )
            print(f"Model '{model_name}' downloaded successfully.")

        model_exists = os.path.exists(config_file_path)
        weights_exist = False
        if model_exists:
            weight_files = ['pytorch_model.bin', 'model.safetensors']
            weights_exist = any(os.path.exists(os.path.join(local_model_path, wf)) for wf in weight_files)
            if not weights_exist:
                weights_exist = bool(
                    glob.glob(os.path.join(local_model_path, 'pytorch_model-*.bin')) or
                    glob.glob(os.path.join(local_model_path, 'model-*.safetensors'))
                )

        if not model_exists or not weights_exist:
            try:
                download(force=False)
            except Exception as e:
                raise RuntimeError(f"Failed to download model '{model_name}': {e}") from e
        else:
            print(f"Found existing model '{model_name}' at '{local_model_path}'. Verifying integrity...")
            try:
                cfg_check = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                del cfg_check
                print(f"Model '{model_name}' integrity check passed.")
            except Exception as e:
                print(f"WARNING: Local model at '{local_model_path}' seems corrupted. Re-downloading...")
                try:
                    download(force=True)
                except Exception as download_e:
                    raise RuntimeError(f"Failed to re-download model '{model_name}': {download_e}") from download_e

    def _load_model_and_processor(self, local_path: str):
        from transformers import AutoModel, AutoProcessor

        try:
            print("ImageEmbedder (Worker): Loading SigLIP model...")
            self.model = AutoModel.from_pretrained(local_path, local_files_only=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
            self.embedding_dim = self.model.config.text_config.hidden_size
        except Exception as e:
            raise RuntimeError(f"Failed to load SigLIP model from '{local_path}': {e}") from e

    def _read_image(self, image_path: str):
        import cv2
        import imageio

        try:
            if image_path.lower().endswith('.gif'):
                gif = imageio.mimread(image_path)
                image = np.array(gif[0])
            else:
                image = cv2.imread(image_path)

            if image is None:
                return None

            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError(f"Unexpected image shape: {image.shape}")
            return image
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            return None

    def embed_image(self, image_path: str) -> np.ndarray:
        """
        Returns a L2-normalised image embedding of shape (1, embedding_dim).
        """
        if self.model is None:
            raise RuntimeError("ImageEmbedder not initiated.")

        local_path, temp = vfs.resolve_to_local_path(image_path)
        try:
            image = self._read_image(local_path)
            if image is None:
                raise ValueError(f"Failed to read image: {image_path}")

            try:
                inputs = self.processor(images=[image], padding="max_length", return_tensors="pt").to(self.device)
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.cpu().numpy()  # shape (1, embedding_dim)
            except Exception as e:
                print(f"Error embedding image {image_path}: {e}")
                traceback.print_exc()
                raise
        finally:
            if temp and os.path.exists(temp):
                try: os.unlink(temp)
                except OSError: pass

    def embed_text(self, text: str) -> np.ndarray:
        """
        Returns a L2-normalised text embedding of shape (embedding_dim,).
        """
        if self.model is None:
            raise RuntimeError("ImageEmbedder not initiated.")

        try:
            inputs = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            return features.squeeze(0).cpu().numpy()  # shape (embedding_dim,)
        except Exception as e:
            print(f"Error embedding text: {e}")
            traceback.print_exc()
            raise

    def compare(self, image_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        """
        Computes cosine similarities between image_embeddings (N, D) and query_embedding (D,).
        Returns list of N float scores.
        """
        if image_embeddings is None or query_embedding is None:
            return [0.0]
        img_emb = np.array(image_embeddings, dtype=np.float32)
        qry_emb = np.array(query_embedding, dtype=np.float32)
        norm_q = qry_emb / (np.linalg.norm(qry_emb) + 1e-9)
        norms = np.linalg.norm(img_emb, axis=-1, keepdims=True) + 1e-9
        norm_i = img_emb / norms
        scores = np.dot(norm_i, norm_q).flatten()
        return scores.tolist()


# Set the process name for system tools (nvidia-smi, top, ps)
import setproctitle


def _worker_loop(input_queue, output_queue, cfg):
    """The loop running in the separate process."""
    setproctitle.setproctitle("Anagnorisis-ImageEmbedder")

    try:
        embedder = _ImageEmbedderImpl(cfg)

        while True:
            try:
                task = input_queue.get()
                if task is None:  # sentinel to stop
                    break

                command, args, kwargs = task

                if command == 'initiate':
                    embedder.initiate(*args, **kwargs)
                    result = {
                        'embedding_dim': embedder.embedding_dim,
                        'device_type': embedder.device.type,
                        'model_hash': embedder.model_hash,
                    }
                    output_queue.put(('success', result))

                elif hasattr(embedder, command):
                    method = getattr(embedder, command)
                    result = method(*args, **kwargs)
                    output_queue.put(('success', result))

                else:
                    output_queue.put(('error', ValueError(f"Unknown command: {command}")))

            except Exception as e:
                traceback.print_exc()
                output_queue.put(('error', e))

    except Exception as e:
        print(f"Critical error in ImageEmbedder worker process: {e}")
        traceback.print_exc()


# --- The Proxy Class (Runs in main process) ---

class ImageEmbedder:
    """
    A singleton proxy class that manages a subprocess for image embedding
    using SigLIP.  The subprocess is terminated after a period of inactivity
    so that GPU memory is not held for the lifetime of the main service.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImageEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return

        if cfg is None:
            raise ValueError("ImageEmbedder requires a configuration object (cfg) on first initialization.")

        self.cfg = cfg
        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._lock = threading.Lock()

        # State mirroring
        self.embedding_dim = None
        self.device = torch.device('cpu')  # Updated to actual device after initiate()
        self.model_hash = None
        self._models_folder = None

        # Idle management
        self._last_used_time = 0
        self._idle_timeout = 120  # seconds
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_idle, daemon=True)
        self._monitor_thread.start()

        self._initialized = True

    def _monitor_idle(self):
        """Background thread to kill the subprocess when idle."""
        while not self._shutdown_event.is_set():
            time.sleep(5)
            with self._lock:
                if self._process is not None and self._process.is_alive():
                    if self._last_used_time > 0 and time.time() - self._last_used_time > self._idle_timeout:
                        print(f"ImageEmbedder: Idle for {self._idle_timeout}s. Terminating subprocess to free GPU.")
                        self._terminate_process()

    def _terminate_process(self):
        """Terminates the worker process immediately."""
        if self._process:
            try:
                self._input_queue.put(None)
                self._process.join(timeout=1)
            except Exception:
                pass

            if self._process.is_alive():
                print("ImageEmbedder: Force killing subprocess...")
                self._process.terminate()
                self._process.join()

            self._process = None
            self._input_queue = None
            self._output_queue = None

            import gc
            gc.collect()

    def unload(self):
        """
        Immediately terminate the worker subprocess to free GPU memory.
        embedding_dim, model_hash and _models_folder are preserved so the
        process restarts transparently on the next call.
        """
        with self._lock:
            self._terminate_process()
        print("ImageEmbedder: Unloaded subprocess (model_hash preserved for restart).")

    def _ensure_process_running(self):
        """Starts the subprocess if it is not currently running. Must be called within self._lock."""
        if self._process is None or not self._process.is_alive():
            print("ImageEmbedder: Starting worker subprocess...")
            ctx = multiprocessing.get_context('spawn')
            self._input_queue = ctx.Queue()
            self._output_queue = ctx.Queue()
            self._process = ctx.Process(
                target=_worker_loop,
                args=(self._input_queue, self._output_queue, self.cfg),
                name="Anagnorisis-ImageEmbedder",
            )
            self._process.start()

            if self._models_folder:
                print("ImageEmbedder: Re-initiating model in new subprocess...")
                self._send_command_internal('initiate', (self._models_folder,), {})

    def _send_command_internal(self, command, args, kwargs):
        """Send a command to the worker and wait for the result. Assumes lock is held."""
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
                        f"ImageEmbedder subprocess died unexpectedly during "
                        f"'{command}' (exit code: {exit_code})."
                    )
                # Still alive — keep waiting.

        if status == 'error':
            raise result
        return result

    def _execute(self, command, *args, **kwargs):
        """Thread-safe wrapper to execute a command in the worker subprocess."""
        with self._lock:
            self._ensure_process_running()
            result = self._send_command_internal(command, args, kwargs)
            self._last_used_time = time.time()
            return result

    # --- Public Interface ---

    def initiate(self, models_folder: str):
        self._models_folder = models_folder
        res = self._execute('initiate', models_folder)
        self.embedding_dim = res['embedding_dim']
        self.model_hash = res.get('model_hash', 'unknown_hash')
        self.device = torch.device(res['device_type'])
        self.unload()

    def embed_image(self, image_path: str) -> np.ndarray:
        return self._execute('embed_image', image_path)

    def embed_text(self, text: str) -> np.ndarray:
        return self._execute('embed_text', text)

    def compare(self, image_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        return self._execute('compare', image_embeddings, query_embedding)

    # --- Cleanup ---
    def __del__(self):
        self._shutdown_event.set()
        self._terminate_process()


if __name__ == '__main__':
    import os
    import sys
    import time
    import numpy as np
    from omegaconf import OmegaConf

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(models_path, exist_ok=True)

    mock_cfg = OmegaConf.create({
        'image_embedder': {
            'model_name': 'google/siglip-base-patch16-224',
        },
        'main': {
            'device': 'cuda',
        }
    })

    # ---------------------------------------------------------------------------
    # Locate test images
    # ---------------------------------------------------------------------------
    test_image_dir = os.path.join(script_dir, '..', 'modules', 'images', 'engine_test_data')
    test_images = []
    if os.path.isdir(test_image_dir):
        for fname in sorted(os.listdir(test_image_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(test_image_dir, fname))

    if len(test_images) < 2:
        print("WARNING: fewer than 2 test images found in engine_test_data — semantic test will be skipped.")

    # ---------------------------------------------------------------------------
    # Initialise proxy
    # ---------------------------------------------------------------------------
    print("Initializing ImageEmbedder proxy...")
    embedder = ImageEmbedder(cfg=mock_cfg)
    embedder.initiate(models_folder=models_path)
    print(f"ImageEmbedder ready. embedding_dim={embedder.embedding_dim}, model_hash={embedder.model_hash[:8]}")

    # ---------------------------------------------------------------------------
    # Pre-flight checks
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Running pre-flight checks...")

    # embed_image
    if test_images:
        result = embedder.embed_image(test_images[0])
        assert isinstance(result, np.ndarray), f"embed_image should return np.ndarray, got {type(result)}"
        assert result.shape == (1, embedder.embedding_dim), \
            f"embed_image returned wrong shape: {result.shape}, expected (1, {embedder.embedding_dim})"
        print(f"✅ embed_image: correct type and shape {result.shape}")
    else:
        print("⚠️  embed_image: skipped (no test images)")

    # embed_text
    text_emb = embedder.embed_text("a photo of a cat sitting on a sofa")
    assert isinstance(text_emb, np.ndarray), f"embed_text should return np.ndarray, got {type(text_emb)}"
    assert text_emb.shape == (embedder.embedding_dim,), \
        f"embed_text returned wrong shape: {text_emb.shape}, expected ({embedder.embedding_dim},)"
    print(f"✅ embed_text: correct type and shape {text_emb.shape}")

    # ---------------------------------------------------------------------------
    # Semantic similarity test
    # ---------------------------------------------------------------------------
    if len(test_images) >= 2:
        print("\n" + "=" * 50)
        print("Running semantic similarity test...")

        img_embs = np.vstack([embedder.embed_image(p) for p in test_images[:4]])
        query_emb = embedder.embed_text("a cute cat")
        scores = embedder.compare(img_embs, query_emb)
        assert isinstance(scores, list), f"compare should return list, got {type(scores)}"
        assert len(scores) == len(test_images[:4]), "compare returned wrong number of scores"
        print(f"Scores for query 'a cute cat': {[f'{s:.3f}' for s in scores]}")
        print("✅ compare: correct type and count")
    else:
        print("⚠️  Semantic similarity test: skipped (need >= 2 test images)")

    # ---------------------------------------------------------------------------
    # Process lifecycle test
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Running process lifecycle test...")

    # Force a call so the process is alive
    _ = embedder.embed_text("test lifecycle")
    with embedder._lock:
        pid_before = embedder._process.pid if embedder._process else None
    print(f"Process PID before idle timeout: {pid_before}")

    # Manually override last-used time to trigger early idle termination
    embedder._last_used_time = time.time() - (embedder._idle_timeout + 10)
    print(f"Waiting up to 15s for idle monitor to fire...")
    for _ in range(30):
        time.sleep(0.5)
        with embedder._lock:
            if embedder._process is None:
                break
    with embedder._lock:
        process_killed = embedder._process is None
    assert process_killed, "Idle monitor should have terminated the subprocess."
    print("✅ Idle monitor correctly terminated the subprocess.")

    # Next call should transparently restart and succeed
    text_emb2 = embedder.embed_text("process restarted successfully")
    with embedder._lock:
        pid_after = embedder._process.pid if embedder._process else None
    assert isinstance(text_emb2, np.ndarray), "embed_text should work after restart"
    assert pid_before != pid_after, "PID should differ after restart"
    print(f"Process PID after restart: {pid_after}")
    print("✅ Process lifecycle test passed (process restarted transparently).")

    # ---------------------------------------------------------------------------
    # Done
    # ---------------------------------------------------------------------------
    print("\n--- ImageEmbedder test completed successfully ---")
