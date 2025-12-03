# src/base_search_engine.py
import os
import torch
import pickle
import hashlib
import flask
import io
import traceback
from abc import ABC, abstractmethod
from huggingface_hub import snapshot_download
from transformers import AutoConfig
from tqdm import tqdm
import pages.images.db_models as db_models
import pages.file_manager as file_manager
import numpy as np
from src.caching import TwoLevelCache
from src.model_manager import ModelManager # Assuming ModelManager is already implemented and works
from typing import List

class BaseSearchEngine(ABC):
    """
    Base class for all search engines (Image, Text, Music).
    Encapsulates common functionality like model loading, caching, and file processing.
    """
    
    _instance = None # For singleton pattern implementation in subclasses if desired

    def __new__(cls, *args, **kwargs):
        # Implement a basic singleton pattern here, allowing only one instance per subclass
        if cls._instance is None:
            cls._instance = super(BaseSearchEngine, cls).__new__(cls)
            cls._instance._initialized = False # Flag to control __init__ calls
        return cls._instance

    def __init__(self, cfg=None): # cfg might be needed by subclasses in their _load_model_and_processor
        if self._initialized:
            return

        self.device = None # Will be set during model loading
        self._model_manager = None # Will wrap the actual model
        self.processor = None
        self.is_busy = False # Flag to prevent concurrent processing
        self.model_hash = None
        self.embedding_dim = None

        self._batch_processing_size = 1

        # Caching mechanisms
        # self.cached_metadata = None

        cache_folder = os.path.join(cfg.main.cache_path, self.cache_prefix)
        self._fast_cache = TwoLevelCache(cache_dir=cache_folder, name=f"{self.cache_prefix}")

        self._initialized = True # Mark as initialized

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Abstract property: Returns the Hugging Face model name/ID (e.g., "google/siglip-base-patch16-224")."""
        pass
    
    @property
    @abstractmethod
    def cache_prefix(self) -> str:
        """Abstract property: Returns a unique prefix for cache files (e.g., 'images', 'text', 'music')."""
        pass
    
    @abstractmethod
    def _load_model_and_processor(self, local_model_path: str):
        """
        Abstract method: Loads the specific model and processor from the local path.
        Sets self.model (the raw model object), self.processor, and self.device.
        Important: Load the raw model to CPU before wrapping with ModelManager if needed.
        """
        pass
    
    @abstractmethod
    def _get_metadata(self, file_path):
        """
        Abstract method: Returns the metadata of a file
        specific to its modality (e.g., get_image_metadata, get_audiofile_data).
        """
        pass
    
    @abstractmethod
    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Abstract method: Processes a single file to generate its raw embeddings.
        This method should encapsulate modality-specific file reading and model inference.
        Returns a torch.Tensor of embeddings (e.g., [1, embedding_dim]).
        """
        pass
    
    @abstractmethod
    def _get_db_model_class(self):
        """
        Abstract method: Returns the SQLAlchemy database model class
        associated with this modality (e.g., ImagesLibrary, MusicLibrary, TextLibrary).
        """
        pass

    @abstractmethod
    def _get_model_hash_postfix(self) -> str:
        """
        Returns a short hash postfix based on the model name and version.
        Used for versioning embeddings.
        """
        return ""
    
    def _get_media_folder(self) -> str:
        """
        Returns the media folder path associated with this search engine.
        Used for file path validations and relative path calculations.
        """
        return ""

    def initiate(self, models_folder: str, cache_folder: str, **kwargs):
        """
        Initializes the search engine. This is the primary entry point for setup.
        Ensures model is downloaded, loads it, and sets up caching.
        """
        # Model is already initiated via ModelManager's lazy loading
        if self._model_manager is not None:
             return
        
        # Setup model path
        if self.model_name is None:
            model_folder_name = 'dummy_model' # Fallback if model_name is not set
        else:
            model_folder_name = self.model_name.replace('/', '__') # Sanitized name for local folder

        local_model_path = os.path.join(models_folder, model_folder_name)
        
        # Ensure directories exist
        os.makedirs(models_folder, exist_ok=True)
        
        # Download model if needed
        self._ensure_model_downloaded(local_model_path)
        
        # Load the actual model and processor (this method should put model on CPU initially)
        self._load_model_and_processor(local_model_path)
        
        # Wrap the model with ModelManager for lazy GPU loading/unloading
        # Assuming _load_model_and_processor set self.model and self.device
        self._model_manager = ModelManager(self.model, device=self.device)
        self.model = self._model_manager # Override self.model with the manager instance
        self.model_hash = self._get_model_hash_from_instance() # Get hash from the *actual* loaded model

        try:
            self.model.eval()
        except Exception:
            pass
        # if torch.cuda.is_available():
        #     torch.backends.cudnn.benchmark = True
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
        
        print(f"{self.__class__.__name__} initiated successfully.")

        # Force unload after wrapping as this usually cause the model to be loaded to GPU
        self._model_manager.unload_model()
    
    # def _ensure_model_downloaded(self, local_model_path: str):
    #     """
    #     Checks if the model exists locally; if not, downloads it from Hugging Face.
    #     """
    #     # Check for a common file like 'config.json' to indicate a potentially complete download
    #     # Different models might have different definitive files. 'config.json' is a good general one.
    #     config_file_path = os.path.join(local_model_path, 'config.json') 
    #     if not os.path.exists(config_file_path):
    #         print(f"Model '{self.model_name}' not found locally or incomplete at '{local_model_path}'. Downloading...")
    #         if self.model_name is None:
    #             print("ERROR: Model name is not set. Cannot download model.")
    #         else:    
    #             try:
    #                 snapshot_download(
    #                     repo_id=self.model_name,
    #                     local_dir=local_model_path,
    #                     local_dir_use_symlinks=False # Download actual files
    #                 )
    #                 print(f"Model '{self.model_name}' downloaded successfully.")
    #             except Exception as e:
    #                 print(f"ERROR: Failed to download model '{self.model_name}'. Please check your internet connection and permissions.")
    #                 print(f"Error details: {e}")
    #                 #raise RuntimeError(f"Failed to download required model: {self.model_name}") from e
    #     else:
    #         print(f"Found existing model '{self.model_name}' at '{local_model_path}'.")

    def _ensure_model_downloaded(self, local_model_path: str):
        """
        Checks if the model exists locally; if not, downloads it from Hugging Face.
        Retries with force_download=True if the local model is corrupted.
        """
        if self.model_name is None:
            print("ERROR: Model name is not set. Cannot download model.")
            return

        # 1. Check for basic existence (e.g., config.json)
        config_file_path = os.path.join(local_model_path, 'config.json')
        
        # Helper to perform the download
        def download(force=False):
            print(f"{'Re-downloading' if force else 'Downloading'} model '{self.model_name}' to '{local_model_path}'...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False, # Download actual files
                force_download=force,         # Force re-download if requested
                resume_download=True          # Resume if possible (unless forced)
            )
            print(f"Model '{self.model_name}' downloaded successfully.")

        if not os.path.exists(config_file_path):
            try:
                download(force=False)
            except Exception as e:
                print(f"ERROR: Failed to download model '{self.model_name}'.")
                print(f"Error details: {e}")
                # raise RuntimeError(...) # Optional: re-raise if critical
        else:
            print(f"Found existing model '{self.model_name}' at '{local_model_path}'. Verifying integrity...")
            # 2. Verify integrity by attempting a dry-run load (or just trust it until it fails)
            # Since we can't easily "dry-run" without loading the whole model, we rely on the
            # calling code (initiate) to catch loading errors.
            # BUT, we can check if the folder is empty or missing key files.
            
            # If you want to be very safe, you can try to load the config here:
            try:
                model = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                # Unload immediately
                del model
                print(f"Model '{self.model_name}' integrity check passed.")
            except Exception as e:
                print(f"WARNING: Local model at '{local_model_path}' seems corrupted. Re-downloading...")
                print(f"Integrity check error: {e}")
                try:
                    download(force=True)
                except Exception as download_e:
                    print(f"ERROR: Failed to re-download model '{self.model_name}'.")
                    print(f"Error details: {download_e}")
    
    def _get_model_hash_from_instance(self) -> str:
        """
        Calculates a hash of the currently loaded model's state dictionary.
        Used for versioning embeddings.
        """
        if self._model_manager is None: 
            raise Exception("Model manager is not initialized. Call initiate() first.")
        
        # Access the underlying model's state_dict via ModelManager
        state_dict = self._model_manager.state_dict()
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        model_hash = hashlib.md5(buffer.read()).hexdigest() + self._get_model_hash_postfix()
        return model_hash
    
    def get_hash_algorithm(self) -> str:
        """
        Returns the current hashing algorithm identifier used by this engine.
        This is useful for storing in the DB alongside file hashes.
        """
        return "md5:v1" # Currently only md5 is implemented; can be extended in subclasses
    
    def get_file_hash(self, file_path: str) -> str:
        """Returns the cached hash for a given file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # We expect file_path to be absolute not a relative one to preserve consistency across different callers
        # Check if file_path is not started from the media_folder path and log a warning in this case
        if not os.path.abspath(file_path).startswith(os.path.abspath(self._get_media_folder())):
            traceback.print_stack()
            print(f"WARNING: File '{file_path}' is not located within the media folder or not an absolute path.")

        # Get the last modified time of the file
        st = os.stat(file_path, follow_symlinks=False)
        size = st.st_size
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))

        cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{self.get_hash_algorithm()}"
        file_hash = self._fast_cache.get(cache_key)

        if file_hash is not None:
            return file_hash
        
        # If not in cache or file has been modified, calculate the hash
        with open(file_path, "rb") as f:
            bytes = f.read()  # Read the entire file as bytes
            file_hash = hashlib.md5(bytes).hexdigest()
        
        # Update the cache
        self._fast_cache.set(cache_key, file_hash)
        return file_hash

    def get_metadata(self, file_path, file_hash=None) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get the last modified time of the file (still used for time-based invalidation)
        last_modified_time = os.path.getmtime(file_path)

        cache_key = f"METADATA_OF_FILE::{file_path}::{last_modified_time}"
        file_metadata = self._fast_cache.get(cache_key)

        if file_metadata is not None:
            return file_metadata

        # If not in cache or file has been modified, extract metadata using _get_metadata_function
        metadata = self._get_metadata(file_path) # Method expected to return a dictionary of metadata attributes
        metadata['file_path'] = file_path

        # Update the cache 
        self._fast_cache.set(cache_key, metadata)
        return metadata

    def _process_batch_files(self, file_paths: List[str], **kwargs) -> List[torch.Tensor]:
        """
        Default batching: call _process_single_file per path.
        Engines can override to do true vectorized/batched inference.
        Must return a list aligned with file_paths; any failed item can be None.
        """
        results: List[torch.Tensor | None] = []
        for p in file_paths:
            try:
                emb = self._process_single_file(p, **kwargs)
            except Exception:
                print(f"ERROR: Failed to process file '{p}'.")
                traceback.print_exc()
                emb = None
            results.append(emb)
        return results
    
    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """
        Processes files with per-file caching and batched inference for cache misses.
        Returns a single tensor [N, D]. Failed files get zeros([1, D]).
        """
        batch_size = self._batch_processing_size

        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        N = len(file_paths)
        outputs: list[torch.Tensor | None] = [None] * N

        # 1) Build items with hashes and check cache
        items = []
        for i, path in enumerate(file_paths):
            try:
                file_hash = self.get_file_hash(path)
            except Exception:
                # If hash fails, force compute with a synthetic key
                file_hash = f"INVALID::{path}"
            cache_key = f"{file_hash}::{self.model_hash}{self._get_model_hash_postfix()}"
            cached = self._fast_cache.get(cache_key)
            if cached is not None:
                outputs[i] = cached  # expected to be CPU tensor
                if callback:
                    callback(sum(1 for o in outputs if o is not None), N)
            else:
                items.append((i, path, cache_key))

        # 2) Batch process only misses
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            idxs = [i for (i, _, _) in batch]
            paths = [p for (_, p, _) in batch]
            keys = [k for (_, _, k) in batch]

            try:
                batch_embs = self._process_batch_files(paths, media_folder=media_folder, **kwargs)
            except Exception:
                # catastrophic batch failure
                batch_embs = [None] * len(paths)

            # 3) Normalize results, cache, and place in outputs
            for j, emb in enumerate(batch_embs):
                out_idx = idxs[j]
                key = keys[j]
                if emb is None:
                    emb = torch.zeros(1, self.embedding_dim)
                # Move to CPU for caching; keep single-file [1, D] shape
                emb_cpu = emb.detach().to("cpu")
                if emb is not None: 
                    self._fast_cache.set(key, emb_cpu)
                outputs[out_idx] = emb_cpu
                if callback:
                    callback(sum(1 for o in outputs if o is not None), N)

        # 4) Safety: fill any remaining gaps with zeros
        for i in range(N):
            if outputs[i] is None:
                outputs[i] = torch.zeros(1, self.embedding_dim)

        # 5) Sanity check: move all the outputs to the same device before concatenation
        for i in range(N):
            outputs[i] = outputs[i].to(device=self.device)
        
        # 6) Concatenate and move to active device
        out = torch.cat([t for t in outputs], dim=0)
        return out.to(self.device)

    def process_text(self, text: str, **kwargs) -> torch.Tensor:
        """
        Generic method for processing a text query (e.g., for semantic search).
        This will typically just use the model's text encoder.
        """
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        
        # Ensure model is loaded to device for inference
        model_instance = self._model_manager # This triggers loading to GPU if needed
        
        # Use the specific model's text processing capabilities
        # This part still needs to be implemented in subclasses if the text processing
        # is significantly different (e.g., tokenizer usage, specific model methods)
        raise NotImplementedError("Text processing is not implemented in BaseSearchEngine directly. "
                                  "Subclasses should implement their own 'process_text' if applicable.")
    
    def compare(self, embeds_target: torch.Tensor, embeds_query: torch.Tensor) -> np.ndarray:
        """
        Generic method for comparing two sets of embeddings (e.g., image vs text).
        Performs dot product and applies sigmoid/softmax as appropriate.
        Subclasses can override if their comparison logic is more complex.
        """
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        
        # print("Warning: Converting embeds_target to torch.Tensor")
        # print(f"Type of embeds_target: {type(embeds_target)}")
        # print(f"Shape of embeds_target[0]: {np.shape(embeds_target)}")

        # # Ensure embeddings are torch tensors
        # if type(embeds_target) is not torch.Tensor:
        #     embeds_target = torch.tensor(embeds_target)
        #     print(f"New type of embeds_target: {type(embeds_target)}")
        #     print(f"New shape of embeds_target: {embeds_target.shape}")
        # if type(embeds_query) is not torch.Tensor:
        #     embeds_query = torch.tensor(embeds_query) 

        # Ensure embeddings are on the correct device
        embeds_target = embeds_target.to(self.device)
        embeds_query = embeds_query.to(self.device)

        # Normalize features
        embeds_target = embeds_target / embeds_target.norm(p=2, dim=-1, keepdim=True)
        embeds_query = embeds_query / embeds_query.norm(p=2, dim=-1, keepdim=True)

        # Dot product for similarity
        logits = torch.matmul(embeds_query, embeds_target.t())

        # # If model has logit_scale (like CLIP/CLAP), apply it
        # if hasattr(self._model_manager, 'logit_scale'):
        #     logits = logits * self._model_manager.logit_scale.exp()
        # elif hasattr(self._model_manager, 'logit_scale_t') and hasattr(self._model_manager, 'logit_scale_a'): # For CLAP
        #     # Assuming embeds_query is text and embeds_target is audio/image
        #     logits_per_text = logits * self._model_manager.logit_scale_t.exp()
        #     logits_per_audio = torch.matmul(embeds_target, embeds_query.t()) * self._model_manager.logit_scale_a.exp()
        #     logits = (logits_per_text + logits_per_audio.t()) / 2.0
            
        # # probs = torch.sigmoid(logits) # Sigmoid for similarity score (0-1)

        cosine_sim = logits # Return raw logits that represent cosine similarity

        return cosine_sim.cpu().detach().numpy()