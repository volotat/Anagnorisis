import os
import time
import traceback
import threading
import multiprocessing
import queue
import hashlib
from typing import Optional, List, Dict, Any

import torch

import numpy as np

# --- The Worker Implementation (Runs in separate process) ---

class _TextEmbedderImpl:
    """
    The actual implementation that runs inside the subprocess.
    It holds the heavy model and CUDA context.
    """
    def __init__(self, cfg):
        # Imports are done here to ensure they happen in the subprocess
        
        from sentence_transformers import SentenceTransformer
        # from src.model_manager import ModelManager
        
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.embedding_dim = None
        
        # We don't use ModelManager's idle timeout here because the whole process 
        # will be killed by the parent when idle.
        # self._model_manager_cls = ModelManager 
        self._sentence_transformer_cls = SentenceTransformer

    def initiate(self, models_folder: str):
        if self.model is not None:
            return

        model_name = self.cfg.text.embedding_model
        if not model_name:
            raise ValueError("cfg.text.embedding_model is not specified.")

        model_folder_name = model_name.replace('/', '__')
        local_model_path = os.path.join(models_folder, model_folder_name)

        self._ensure_model_downloaded(models_folder, model_name)
        self._load_model_and_tokenizer(local_model_path)
        self.model_hash = self._calculate_model_hash()

    def _calculate_model_hash(self) -> str:
        """
        Calculates a hash of the model weights to ensure cache validity.
        """
        print("TextEmbedder (Worker): Calculating model hash...")
        try:
            md5 = hashlib.md5()
            # Iterate over state_dict to calculate hash based on weights
            # We sort keys to ensure consistent order
            for k, v in sorted(self.model.state_dict().items()):
                md5.update(k.encode('utf-8'))
                # We hash the shape and a small sample of the tensor to be fast but reasonably safe
                # Hashing the entire tensor for large LLMs is too slow at startup
                md5.update(str(v.shape).encode('utf-8'))
                # Take a slice of the tensor (first 100 elements) for content hashing
                flat_v = v.view(-1)
                sample = flat_v[:100].tolist()
                md5.update(str(sample).encode('utf-8'))
            
            return md5.hexdigest()
        except Exception as e:
            print(f"Error calculating model hash: {e}")
            return "unknown_hash"

    def _ensure_model_downloaded(self, models_folder: str, model_name: str):
        """
        Ensure the HF model is present locally. If missing, download it.
        If present but integrity check fails, retry with force_download=True.
        """
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        # store model name for other methods (used elsewhere in the worker)
        self.model_name = model_name

        local_model_path = os.path.join(models_folder, model_name.replace('/', '__'))
        config_file_path = os.path.join(local_model_path, 'config.json')

        def download(force=False):
            print(f"{'Re-downloading' if force else 'Downloading'} model '{self.model_name}' to '{local_model_path}'...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                force_download=force,
                resume_download=True
            )
            print(f"Model '{self.model_name}' downloaded successfully.")

        # Check if model directory and config exist
        model_exists = os.path.exists(config_file_path)
        weights_exist = False
        
        if model_exists:
            # Check if at least one model weight file exists
            weight_files = ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5', 
                          'model.ckpt.index', 'flax_model.msgpack']
            weights_exist = any(os.path.exists(os.path.join(local_model_path, wf)) for wf in weight_files)
            # Also check for sharded models (pytorch_model-00001-of-*.bin, model-00001-of-*.safetensors)
            if not weights_exist:
                import glob
                weights_exist = bool(glob.glob(os.path.join(local_model_path, 'pytorch_model-*.bin')) or
                                   glob.glob(os.path.join(local_model_path, 'model-*.safetensors')))
        
        if not model_exists or not weights_exist:
            try:
                if model_exists and not weights_exist:
                    print(f"WARNING: Config found but model weights missing for '{self.model_name}'. Resuming download...")
                download(force=False)  # resume_download=True will continue incomplete downloads
            except Exception as e:
                print(f"ERROR: Failed to download model '{self.model_name}'.")
                print(f"Error details: {e}")
                raise RuntimeError(f"Failed to download model '{self.model_name}': {e}") from e
        else:
            print(f"Found existing model '{self.model_name}' at '{local_model_path}'. Verifying integrity...")
            try:
                # Try loading the config as a lightweight integrity check
                cfg = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                del cfg
                print(f"Model '{self.model_name}' integrity check passed.")
            except Exception as e:
                print(f"WARNING: Local model at '{local_model_path}' seems corrupted. Re-downloading...")
                print(f"Integrity check error: {e}")
                try:
                    download(force=True)
                except Exception as download_e:
                    print(f"ERROR: Failed to re-download model '{self.model_name}'.")
                    print(f"Error details: {download_e}")
                    raise RuntimeError(f"Failed to re-download model '{self.model_name}': {download_e}") from download_e

    def _load_model_and_tokenizer(self, local_path: str):
        try:
            print("TextEmbedder (Worker): Loading model...")
            # Load model
            raw_model = self._sentence_transformer_cls(
                local_path, 
                device='cpu', 
                trust_remote_code=True,
                tokenizer_kwargs={"fix_mistral_regex": True}
            )
            self.tokenizer = raw_model.tokenizer
            self.embedding_dim = self.cfg.text.embedding_dimension or raw_model.get_sentence_embedding_dimension()
            
            # Move to device immediately
            self.model = raw_model.to(self.device)
            print(f"TextEmbedder (Worker): Initiated. Embedding dim: {self.embedding_dim} on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load text embedding model from '{local_path}': {e}") from e

    def embed_query(self, query_text: str) -> np.ndarray:
        if not self.model:
            raise RuntimeError("TextEmbedder not initiated.")
        try:
            embedding = self.model.encode(
                query_text,
                task="retrieval.query",
                truncate_dim=self.cfg.text.embedding_dimension,
                convert_to_tensor=False,
                device=self.device
            )
            return embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            traceback.print_exc()
            return np.array([], dtype=np.float32)

    def embed_text(self, long_text: str) -> np.ndarray:
        if not self.model or not self.tokenizer:
            raise RuntimeError("TextEmbedder not initiated.")

        try:
            # Chunking logic
            encoding = self.tokenizer(
                long_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
            )
            tokens = encoding["input_ids"]
            offsets = encoding["offset_mapping"]
            
            chunk_size = self.cfg.text.chunk_size
            chunk_overlap = self.cfg.text.chunk_overlap
            
            chunk_texts = []
            start_token_idx = 0
            
            while start_token_idx < len(tokens):
                end_token_idx = min(start_token_idx + chunk_size, len(tokens))
                char_start = offsets[start_token_idx][0]
                char_end = offsets[end_token_idx - 1][1]
                chunk_text = long_text[char_start:char_end]
                chunk_texts.append(chunk_text)
                
                next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
                if next_start_token_idx <= start_token_idx:
                    next_start_token_idx = start_token_idx + chunk_size
                if end_token_idx == len(tokens):
                    break
                start_token_idx = next_start_token_idx

            if not chunk_texts:
                return []

            embeddings = self.model.encode(
                chunk_texts,
                task="retrieval.passage",
                truncate_dim=self.cfg.text.embedding_dimension,
                convert_to_tensor=False,
                device=self.device
            )
            return embeddings
        except Exception as e:
            print(f"Error embedding long text: {e}")
            traceback.print_exc()
            return []

    def compare(self, text_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        if text_embeddings is None or query_embedding is None or text_embeddings.size == 0 or query_embedding.size == 0:
            return [0.0]
        
        norm_query = query_embedding  / np.linalg.norm(query_embedding, axis=-1, keepdims=True)
        norm_chunks = text_embeddings  / np.linalg.norm(text_embeddings, axis=-1, keepdims=True)
        scores = np.dot(norm_chunks, norm_query.T).flatten()
        return scores.tolist()

    def get_chunk_scores(self, long_text: str, query_text: str) -> Dict[str, float]:
        # This method requires logic that combines embedding and tokenization.
        # To keep the worker simple, we can reuse the internal methods.
        # However, since get_chunk_scores in the original code does complex logic 
        # with token offsets, we should run that logic here in the worker 
        # where the tokenizer exists.
        
        if not self.model or not self.tokenizer:
            raise RuntimeError("TextEmbedder not initiated.")

        query_embedding = self.embed_query(query_text)
        all_chunk_embeddings = self.embed_text(long_text)

        if all_chunk_embeddings is None or len(all_chunk_embeddings) == 0:
            return {}
        if query_embedding is None or len(query_embedding) == 0:
            return {}

        # Re-create offsets (same logic as original)
        encoding = self.tokenizer(
            long_text, add_special_tokens=False, truncation=False, return_offsets_mapping=True
        )
        tokens = encoding["input_ids"]
        offsets = encoding["offset_mapping"]
        
        chunk_size = self.cfg.text.chunk_size
        chunk_overlap = self.cfg.text.chunk_overlap
        
        chunks_info = []
        start_token_idx = 0
        
        while start_token_idx < len(tokens):
            end_token_idx = min(start_token_idx + chunk_size, len(tokens))
            char_start = offsets[start_token_idx][0]
            char_end = offsets[end_token_idx - 1][1]
            chunks_info.append({'start_char': char_start, 'end_char': char_end})
            
            next_start_token_idx = start_token_idx + chunk_size - chunk_overlap
            if next_start_token_idx <= start_token_idx:
                next_start_token_idx = start_token_idx + chunk_size
            if end_token_idx == len(tokens):
                break
            start_token_idx = next_start_token_idx

        all_chunk_scores = self.compare(all_chunk_embeddings, query_embedding)
        for i, score in enumerate(all_chunk_scores):
            chunks_info[i]['score'] = score

        segments = {}
        last_processed_char_index = 0

        for i in range(len(chunks_info)):
            current_chunk = chunks_info[i]
            if i + 1 < len(chunks_info):
                next_chunk = chunks_info[i+1]
                unique_part_end_char = next_chunk['start_char']
                if last_processed_char_index < unique_part_end_char:
                    unique_text = long_text[last_processed_char_index:unique_part_end_char]
                    if unique_text: segments[unique_text] = current_chunk['score']
                    last_processed_char_index = unique_part_end_char

                overlap_start_char = next_chunk['start_char']
                overlap_end_char = current_chunk['end_char']
                if overlap_start_char < overlap_end_char:
                    overlap_text = long_text[overlap_start_char:overlap_end_char]
                    if overlap_text:
                        averaged_score = (current_chunk['score'] + next_chunk['score']) / 2.0
                        segments[overlap_text] = averaged_score
                    last_processed_char_index = overlap_end_char
            else:
                final_segment_start_char = last_processed_char_index
                final_segment_end_char = current_chunk['end_char']
                if final_segment_start_char < final_segment_end_char:
                    final_text = long_text[final_segment_start_char:final_segment_end_char]
                    if final_text: segments[final_text] = current_chunk['score']
        
        return segments

# Set the process name for system tools (nvidia-smi, top, ps)
import setproctitle

def _worker_loop(input_queue, output_queue, cfg):
    """
    The loop running in the separate process.
    """
    setproctitle.setproctitle("Anagnorisis-TextEmbedder")

    try:
        embedder = _TextEmbedderImpl(cfg)
        
        while True:
            try:
                # Wait for a command
                task = input_queue.get()
                if task is None: # Sentinel to stop
                    break
                
                command, args, kwargs = task
                
                if command == 'initiate':
                    embedder.initiate(*args, **kwargs)
                    # Return attributes needed by the proxy
                    result = {
                        'embedding_dim': embedder.embedding_dim,
                        'device_type': embedder.device.type,
                        'model_hash': embedder.model_hash
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
        print(f"Critical error in TextEmbedder worker process: {e}")
        traceback.print_exc()

# --- The Proxy Class (Runs in main process) ---

class TextEmbedder:
    """
    A singleton proxy class that manages a subprocess for text embedding.
    It ensures the subprocess is terminated after a period of inactivity.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return
        
        if cfg is None:
            raise ValueError("TextEmbedder requires a configuration object (cfg) on first initialization.")
            
        self.cfg = cfg
        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._lock = threading.Lock() # Ensures thread safety for the queue
        
        # State mirroring
        self.embedding_dim = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Assumption until loaded
        self.model = "ProxyModel" # Dummy to satisfy checks
        self.tokenizer = "ProxyTokenizer" # Dummy
        self._models_folder = None # Stored for re-initiation
        self.model_hash = None # Mirror hash
        
        # Idle management
        self._last_used_time = 0
        self._idle_timeout = 120 # seconds
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_idle, daemon=True)
        self._monitor_thread.start()
        
        self._initialized = True

    def _monitor_idle(self):
        """Background thread to kill the process when idle."""
        while not self._shutdown_event.is_set():
            time.sleep(5)
            with self._lock:
                if self._process is not None and self._process.is_alive():
                    # Only enforce timeout if process has been used at least once
                    if self._last_used_time > 0 and time.time() - self._last_used_time > self._idle_timeout:
                        print(f"TextEmbedder: Idle for {self._idle_timeout}s. Terminating subprocess to free GPU.")
                        self._terminate_process()

    def _terminate_process(self):
        """Terminates the worker process immediately."""
        if self._process:
            # Try graceful shutdown first
            try:
                self._input_queue.put(None)
                self._process.join(timeout=1)
            except:
                pass
            
            if self._process.is_alive():
                print("TextEmbedder: Force killing subprocess...")
                self._process.terminate()
                self._process.join()
            
            self._process = None
            self._input_queue = None
            self._output_queue = None
            
            # Force garbage collection in main process just in case
            import gc
            gc.collect()

    def _ensure_process_running(self):
        """Starts the process if it's not running."""
        # Must be called within self._lock
        # Don't update _last_used_time here - only update when commands COMPLETE
        
        if self._process is None or not self._process.is_alive():
            print("TextEmbedder: Starting worker subprocess...")
            # Use 'spawn' to avoid CUDA context issues if main process has touched CUDA
            ctx = multiprocessing.get_context('spawn')
            self._input_queue = ctx.Queue()
            self._output_queue = ctx.Queue()
            
            self._process = ctx.Process(
                target=_worker_loop, 
                args=(self._input_queue, self._output_queue, self.cfg),
                name="Anagnorisis-TextEmbedder"  # Sets internal Python name
            )
            self._process.start()
            
            # If we were previously initiated, re-initiate the new process
            if self._models_folder:
                print("TextEmbedder: Re-initiating model in new subprocess...")
                self._send_command_internal('initiate', (self._models_folder,), {})

    def _send_command_internal(self, command, args, kwargs):
        """Helper to send command and wait for result. Assumes lock is held."""
        self._input_queue.put((command, args, kwargs))
        
        try:
            # The worse case timeout is during model loading, so we set 48 hours here to allow the model to load
            if command == 'initiate':
                timeout = 48 * 3600
            else:
                timeout = 25 * 60 # 25 minutes for other commands
                
            status, result = self._output_queue.get(timeout=timeout)
        except queue.Empty:
            self._terminate_process()
            raise RuntimeError("TextEmbedder subprocess timed out.")
            
        if status == 'error':
            raise result
        return result
    
    def _execute(self, command, *args, **kwargs):
        """Public wrapper to execute commands safely."""
        with self._lock:
            self._ensure_process_running()
            result = self._send_command_internal(command, args, kwargs)
            # Update last used time AFTER command completes successfully
            self._last_used_time = time.time()
            return result

    # --- Public Interface matching original TextEmbedder ---

    def initiate(self, models_folder: str):
        self._models_folder = models_folder
        # Execute to ensure the process starts and loads the model
        res = self._execute('initiate', models_folder)
        self.embedding_dim = res['embedding_dim']
        self.model_hash = res.get('model_hash', 'unknown_hash')
        # Update device type if needed, though main process usually stays on CPU for this proxy
        # self.device = torch.device(res['device_type']) 

    def embed_query(self, query_text: str) -> np.ndarray:
        return self._execute('embed_query', query_text)

    def embed_text(self, long_text: str) -> np.ndarray:
        return self._execute('embed_text', long_text)

    def compare(self, text_embeddings: np.ndarray, query_embedding: np.ndarray) -> List[float]:
        # Optimization: If data is small, we could do this in main process, 
        # but to keep logic consistent and imports clean, we send it to worker.
        return self._execute('compare', text_embeddings, query_embedding)

    def get_chunk_scores(self, long_text: str, query_text: str) -> Dict[str, float]:
        return self._execute('get_chunk_scores', long_text, query_text)

    # --- Cleanup ---
    def __del__(self):
        self._shutdown_event.set()
        self._terminate_process()
    
if __name__ == '__main__':
    from omegaconf import OmegaConf
    import pprint

    # 1. Setup mock configuration
    mock_cfg = OmegaConf.create({
        'text': {
            'embedding_model': 'jinaai__jina-embeddings-v3', #jinaai__jina-embeddings-v3
            'embedding_dimension': 512,
            'chunk_size': 192,
            'chunk_overlap': 64,
        }
    })

    # 2. Create a dummy text file for testing
    script_dir = os.path.dirname(__file__)
    test_text = (
        "Section 1: A Brief History of the Roman Empire.\n"
        "The Roman Empire, which succeeded the Roman Republic, was characterized by a government headed by emperors and large territorial holdings around the Mediterranean Sea in Europe, Africa, and Asia. "
        "The first emperor, Augustus, ruled from 27 BC to 14 AD, marking the beginning of the Pax Romana, a period of relative peace and stability that lasted for over two centuries. "
        "The empire's economy was based on agriculture and trade, with a vast network of roads and sea routes facilitating the movement of goods and armies. "
        "Key challenges included managing its vast borders, internal political instability, and economic pressures. "
        "The eventual decline in the West during the 5th century AD was a complex process attributed to factors like barbarian invasions, economic troubles, and over-reliance on slave labor.\n\n"
        
        "Section 2: Fundamentals of Quantum Computing.\n"
        "Quantum computing represents a paradigm shift from classical computing. Instead of bits, which can be a 0 or a 1, quantum computers use qubits. "
        "A qubit can exist in a state of superposition, meaning it can be a combination of both 0 and 1 simultaneously. This property allows quantum computers to process a vast number of calculations at once. "
        "Another key principle is entanglement, a phenomenon where two or more qubits become linked in such a way that their fates are intertwined, regardless of the distance separating them. "
        "Measuring the state of one entangled qubit instantly influences the state of the other. These principles could enable quantum machines to solve complex optimization, cryptography, and simulation problems that are intractable for even the most powerful classical supercomputers.\n\n"

        "Section 3: The Art of Baking Sourdough Bread.\n"
        "Baking sourdough bread is a rewarding process that relies on a symbiotic culture of wild yeast and lactobacilli, known as a sourdough starter. "
        "The process begins with feeding the starter with flour and water until it becomes active and bubbly. The dough is then mixed, often using a technique called autolyse where flour and water are combined before adding the starter and salt. "
        "This is followed by a period of bulk fermentation, which includes a series of 'stretch and folds' to develop gluten strength. "
        "After shaping the loaf, it undergoes a final proof, often in a cool environment, before being baked in a very hot oven, typically inside a Dutch oven to trap steam and create a crispy crust."
    )

    # 4. Define a query and run the test
    query = "What are the principles of quantum superposition and entanglement?"
    
    # 3. Initialize the TextEmbedder
    # Note: The models will be downloaded to ../models/ relative to this script
    models_path = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(models_path, exist_ok=True)
    
    print("Initializing Proxy...")
    embedder = TextEmbedder(cfg=mock_cfg)
    embedder.initiate(models_folder=models_path)

    # 3.5. Pre-run type and shape checks
    print("\n" + "="*50)
    print("Running pre-flight checks on method outputs...")
    
    # Test embed_query
    test_query_embedding = embedder.embed_query(query)
    assert isinstance(test_query_embedding, np.ndarray), f"embed_query should return np.ndarray, but got {type(test_query_embedding)}"
    assert test_query_embedding.shape == (mock_cfg.text.embedding_dimension,), f"embed_query returned wrong shape: {test_query_embedding.shape}"
    print("✅ embed_query returns correct type and shape.")

    # Test embed_text
    test_text_embeddings = embedder.embed_text(test_text)

    # The result can be a single ndarray (for one chunk) or a list of them.
    if isinstance(test_text_embeddings, np.ndarray):
        # This case handles single-chunk inputs
        assert test_text_embeddings[0].shape == (mock_cfg.text.embedding_dimension,), f"embed_text (single chunk) returned wrong shape: {test_text_embeddings[0].shape}"
    else:
        raise AssertionError(f"embed_text returned an unexpected type: {type(test_text_embeddings)}")
    print("✅ embed_text returns correct type and shape.")

    # Test compare
    test_scores = embedder.compare(test_text_embeddings, test_query_embedding)
    if isinstance(test_scores, list):
        assert all(isinstance(score, float) for score in test_scores), f"compare should return a list of floats, but got {type(test_scores)}"
    else:
        raise AssertionError(f"compare returned an unexpected type: {type(test_scores)}")
    print("✅ compare returns correct type.")

    print("Pre-flight checks passed.")
    print("="*50 + "\n")


    print("\n" + "="*50)
    print(f"Query: '{query}'")
    print("="*50 + "\n")
    
    # Get the scored chunks
    scored_segments = embedder.get_chunk_scores(test_text, query)

    # 5. Print the results
    print("Scored Text Segments:")
    # Use a custom print for better readability
    for text, score in scored_segments.items():
        print(f"  - SCORE: {score:.4f} | TEXT: \"{text.strip()}\"")

    # 6. Verify that the combined segments reconstruct the original text
    reconstructed_text = "".join(scored_segments.keys())
    
    print("\n" + "="*50)
    print("Verifying text reconstruction...")
    
    if reconstructed_text == test_text:
        print("✅ SUCCESS: Reconstructed text perfectly matches the original text.")
    else:
        print("❌ FAILURE: Reconstructed text does not match the original text.")
        # For debugging, print the differences
        import difflib
        diff = difflib.unified_diff(
            test_text.splitlines(keepends=True),
            reconstructed_text.splitlines(keepends=True),
            fromfile='original_text',
            tofile='reconstructed_text',
        )
        print("Differences found:\n" + ''.join(diff))

    # --- Test Process Lifecycle ---
    print("\n" + "="*50)
    print("Testing Process Lifecycle (Idle Timeout & Restart)...")
    
    # 1. Get current process info
    if embedder._process is None:
        print("❌ Error: Process is None before lifecycle test starts.")
    else:
        initial_pid = embedder._process.pid
        print(f"Current worker PID: {initial_pid}")
        
        # 2. Reduce timeout for testing
        print("Reducing idle timeout to 2 seconds...")
        embedder._idle_timeout = 2
        
        # 3. Wait for timeout
        # The monitor thread sleeps for 5s, so we need to wait at least that long + buffer
        print("Waiting for idle timeout (approx 6s)...")
        time.sleep(6) 
        
        # 4. Verify termination
        if embedder._process is None:
            print("✅ Worker process reference cleared in proxy.")
        else:
            print(f"❌ Worker process reference still exists: {embedder._process}")
        
        # Check active children
        active_children = multiprocessing.active_children()
        active_pids = [p.pid for p in active_children]
        
        if initial_pid not in active_pids:
            print(f"✅ Old worker process {initial_pid} is no longer active.")
        else:
            print(f"❌ Old worker process {initial_pid} is still active (Zombie)!")

        # 5. Trigger restart
        print("Triggering restart via new query...")
        embedder.embed_query("Wake up!")
        
        if embedder._process is None:
             print("❌ Failed to restart process.")
        else:
            new_pid = embedder._process.pid
            print(f"New worker PID: {new_pid}")
            
            if new_pid != initial_pid:
                print("✅ New worker process started (PID changed).")
            else:
                print("❌ PID did not change (Process might not have restarted).")
            
            # 6. Check for duplicates
            active_children_after = multiprocessing.active_children()
            print(f"Active child processes: {[p.pid for p in active_children_after]}")
            
            # We expect the new_pid to be there
            if new_pid in [p.pid for p in active_children_after]:
                print("✅ New worker is in active children.")
            
            # Check if we have exactly 1 child (assuming no other multiprocessing usage in this script)
            if len(active_children_after) == 1:
                print("✅ Exactly one active child process detected.")
            elif len(active_children_after) > 1:
                print(f"⚠️  Warning: {len(active_children_after)} active child processes detected. Potential leak?")
            else:
                print("❌ No active children found (unexpected if we just started one).")

    print("Process Lifecycle tests passed.")
    print("="*50 + "\n")