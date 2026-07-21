import os
import time
import traceback
import threading
import multiprocessing
import queue
import hashlib
from typing import List

import torch
import torchaudio
import numpy as np
import src.virtual_file_system as vfs

_SHARED_CLAP = None
def get_shared_audio_embedder(cfg, models_folder):
    global _SHARED_CLAP
    if _SHARED_CLAP is None:
        _SHARED_CLAP = AudioEmbedder(cfg)
        _SHARED_CLAP.initiate(models_folder)
    return _SHARED_CLAP

# --- The Worker Implementation (Runs in separate process) ---

class _AudioEmbedderImpl:
    """
    The actual implementation that runs inside the subprocess.
    It holds the CLAP model and CUDA context.
    Converts audio files and text into embedding vectors.
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

        model_name = self.cfg.audio_embedder.model_name
        if not model_name:
            raise ValueError("cfg.audio_embedder.model_name is not specified.")

        model_folder_name = model_name.replace('/', '__')
        local_model_path = os.path.join(models_folder, model_folder_name)

        self._ensure_model_downloaded(models_folder, model_name)
        self._load_model_and_processor(local_model_path)
        self.model_hash = self._calculate_model_hash()
        print(f"AudioEmbedder (Worker): Initiated. Embedding dim: {self.embedding_dim} on {self.device}")

    def _calculate_model_hash(self) -> str:
        print("AudioEmbedder (Worker): Calculating model hash...")
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
        from transformers import ClapModel, ClapProcessor

        try:
            print("AudioEmbedder (Worker): Loading CLAP model...")
            self.model = ClapModel.from_pretrained(local_path, local_files_only=True)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.processor = ClapProcessor.from_pretrained(local_path, local_files_only=True)
            self.embedding_dim = self.model.config.text_config.projection_dim
        except Exception as e:
            raise RuntimeError(f"Failed to load CLAP model from '{local_path}': {e}") from e

    def _read_audio(self, audio_path: str):
        import torchaudio

        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None, None

    def embed_audio(self, audio_path: str) -> np.ndarray:
        """Returns a L2-normalised audio embedding of shape (1, embedding_dim)."""
        if self.model is None:
            raise RuntimeError("AudioEmbedder not initiated.")

        local_path, temp = vfs.resolve_to_local_path(audio_path)
        try:
            waveform, sample_rate = self._read_audio(local_path)
            if waveform is None:
                raise ValueError(f"Failed to read audio file: {audio_path}")
            try:
                # Resample to 48 kHz mono on CPU
                if sample_rate != 48000:
                    waveform = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=48000
                    )(waveform)
                    sample_rate = 48000
                if waveform.dim() > 1 and waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=False)
                if waveform.dim() > 1:
                    waveform = waveform.squeeze()
                waveform_np = waveform.to(torch.float32).contiguous().cpu().numpy()
                
                proc = self.processor(
                    audio=[waveform_np],
                    sampling_rate=sample_rate,
                    return_tensors="pt",
                    padding=False,
                )
                inputs_audio = {
                    k: (v.pin_memory().to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                    for k, v in proc.items()
                }
                if "input_features" in inputs_audio:
                    inputs_audio["input_features"] = torch.nan_to_num(
                        inputs_audio["input_features"], nan=0.0, posinf=0.0, neginf=0.0
                    )

                with torch.no_grad():
                    out = self.model.get_audio_features(**inputs_audio)

                if isinstance(out, torch.Tensor):
                    features = out
                elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                    features = out.pooler_output
                elif hasattr(out, "audio_embeds") and out.audio_embeds is not None:
                    features = out.audio_embeds
                elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                    features = out.last_hidden_state[:, 0]
                else:
                    raise RuntimeError(
                        f"Unexpected get_audio_features() return type: {type(out).__name__}"
                    )

                if features.dim() == 1:
                    features = features.unsqueeze(0)
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                return features.cpu().numpy()  # shape (1, embedding_dim)
            except Exception as e:
                print(f"Error embedding audio {audio_path}: {e}")
                traceback.print_exc()
                raise
        finally:
            if temp and os.path.exists(temp):
                try: os.unlink(temp)
                except OSError: pass

    def embed_text(self, text: str) -> np.ndarray:
        """Returns a CLAP text embedding of shape (embedding_dim,)."""
        if self.model is None:
            raise RuntimeError("AudioEmbedder not initiated.")

        try:
            inputs = self.processor(text=text, padding=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model.get_text_features(**inputs)

            # transformers >= 5.5 may return BaseModelOutputWithPooling instead of a raw tensor
            if isinstance(out, torch.Tensor):
                features = out
            elif hasattr(out, "pooler_output") and out.pooler_output is not None:
                features = out.pooler_output
            elif hasattr(out, "text_embeds") and out.text_embeds is not None:
                features = out.text_embeds
            elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
                features = out.last_hidden_state[:, 0]
            else:
                raise RuntimeError(
                    f"Unexpected get_text_features() return type: {type(out).__name__}"
                )

            return features.squeeze(0).cpu().numpy()  # shape (embedding_dim,)
        except Exception as e:
            print(f"Error embedding text: {e}")
            traceback.print_exc()
            raise

    def compare(self, audio_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        """
        Computes cosine similarities between audio_embeddings (N, D) and query_embedding (D,).
        Returns list of N float scores.
        """
        if audio_embeddings is None or query_embedding is None:
            return [0.0]
        aud_emb = np.array(audio_embeddings, dtype=np.float32)
        qry_emb = np.array(query_embedding, dtype=np.float32)
        norm_q = qry_emb / (np.linalg.norm(qry_emb) + 1e-9)
        norms = np.linalg.norm(aud_emb, axis=-1, keepdims=True) + 1e-9
        norm_a = aud_emb / norms
        scores = np.dot(norm_a, norm_q).flatten()
        return scores.tolist()


# Set the process name for system tools (nvidia-smi, top, ps)
import setproctitle


def _worker_loop(input_queue, output_queue, cfg):
    """The loop running in the separate process."""
    setproctitle.setproctitle("Anagnorisis-AudioEmbedder")

    try:
        embedder = _AudioEmbedderImpl(cfg)

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
        print(f"Critical error in AudioEmbedder worker process: {e}")
        traceback.print_exc()


# --- The Proxy Class (Runs in main process) ---

class AudioEmbedder:
    """
    A singleton proxy class that manages a subprocess for audio embedding
    using CLAP.  The subprocess is terminated after a period of inactivity
    so that GPU memory is not held for the lifetime of the main service.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(AudioEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return

        if cfg is None:
            raise ValueError("AudioEmbedder requires a configuration object (cfg) on first initialization.")

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
                        print(f"AudioEmbedder: Idle for {self._idle_timeout}s. Terminating subprocess to free GPU.")
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
                print("AudioEmbedder: Force killing subprocess...")
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
        print("AudioEmbedder: Unloaded subprocess (model_hash preserved for restart).")

    def _ensure_process_running(self):
        """Starts the subprocess if it is not currently running. Must be called within self._lock."""
        if self._process is None or not self._process.is_alive():
            print("AudioEmbedder: Starting worker subprocess...")
            ctx = multiprocessing.get_context('spawn')
            self._input_queue = ctx.Queue()
            self._output_queue = ctx.Queue()
            self._process = ctx.Process(
                target=_worker_loop,
                args=(self._input_queue, self._output_queue, self.cfg),
                name="Anagnorisis-AudioEmbedder",
            )
            self._process.start()

            if self._models_folder:
                print("AudioEmbedder: Re-initiating model in new subprocess...")
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
                        f"AudioEmbedder subprocess died unexpectedly during "
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

    def embed_audio(self, audio_path: str) -> np.ndarray:
        return self._execute('embed_audio', audio_path)

    def embed_text(self, text: str) -> np.ndarray:
        return self._execute('embed_text', text)

    def compare(self, audio_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        return self._execute('compare', audio_embeddings, query_embedding)

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
        'audio_embedder': {
            'model_name': 'laion/clap-htsat-fused',
        },
        'main': {
            'device': 'cuda',
        }
    })

    # ---------------------------------------------------------------------------
    # Locate test audio files
    # ---------------------------------------------------------------------------
    test_audio_dir = os.path.join(script_dir, '..', 'modules', 'music', 'engine_test_data')
    test_audios = []
    if os.path.isdir(test_audio_dir):
        for fname in sorted(os.listdir(test_audio_dir)):
            if fname.lower().endswith(('.mp3', '.wav', '.flac', '.ogg')):
                test_audios.append(os.path.join(test_audio_dir, fname))

    if len(test_audios) < 2:
        print("WARNING: fewer than 2 test audio files found in engine_test_data — semantic test will be skipped.")

    # ---------------------------------------------------------------------------
    # Initialise proxy
    # ---------------------------------------------------------------------------
    print("Initializing AudioEmbedder proxy...")
    # embedder = AudioEmbedder(cfg=mock_cfg)
    # embedder.initiate(models_folder=models_path)
    embedder = get_shared_audio_embedder(mock_cfg, models_path)
    print(f"AudioEmbedder ready. embedding_dim={embedder.embedding_dim}, model_hash={embedder.model_hash[:8]}")

    # ---------------------------------------------------------------------------
    # Pre-flight checks
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 50)
    print("Running pre-flight checks...")

    # embed_audio
    if test_audios:
        result = embedder.embed_audio(test_audios[0])
        assert isinstance(result, np.ndarray), f"embed_audio should return np.ndarray, got {type(result)}"
        assert result.shape == (1, embedder.embedding_dim), \
            f"embed_audio returned wrong shape: {result.shape}, expected (1, {embedder.embedding_dim})"
        print(f"✅ embed_audio: correct type and shape {result.shape}")
    else:
        print("⚠️  embed_audio: skipped (no test audio files)")

    # embed_text
    text_emb = embedder.embed_text("jazz piano with a slow tempo")
    assert isinstance(text_emb, np.ndarray), f"embed_text should return np.ndarray, got {type(text_emb)}"
    assert text_emb.shape == (embedder.embedding_dim,), \
        f"embed_text returned wrong shape: {text_emb.shape}, expected ({embedder.embedding_dim},)"
    print(f"✅ embed_text: correct type and shape {text_emb.shape}")

    # ---------------------------------------------------------------------------
    # Semantic similarity test
    # ---------------------------------------------------------------------------
    if len(test_audios) >= 2:
        print("\n" + "=" * 50)
        print("Running semantic similarity test...")

        audio_embs = np.vstack([embedder.embed_audio(p) for p in test_audios[:4]])
        query_emb = embedder.embed_text("jazz piano")
        scores = embedder.compare(audio_embs, query_emb)
        assert isinstance(scores, list), f"compare should return list, got {type(scores)}"
        assert len(scores) == len(test_audios[:4]), "compare returned wrong number of scores"
        print(f"Scores for query 'jazz piano': {[f'{s:.3f}' for s in scores]}")
        print("✅ compare: correct type and count")
    else:
        print("⚠️  Semantic similarity test: skipped (need >= 2 test audio files)")

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
    print("\n--- AudioEmbedder test completed successfully ---")
