import os
import torch
import numpy as np
# from PIL import Image # Not strictly needed for metadata stub, but often part of video processing
from src.base_search_engine import BaseSearchEngine
from src.model_manager import ModelManager
import src.scoring_models
import pages.videos.db_models as db_models
import hashlib
import xxhash

def get_video_metadata_stub(file_path):
    """
    Stub function for video metadata extraction.
    Currently returns dummy resolution and duration.
    Will be properly implemented when full video processing is added.
    """
    metadata = {}
    try:
        # Placeholder for actual video metadata extraction (e.g., using moviepy or ffprobe)
        # For now, return a dummy resolution and duration
        metadata['resolution'] = (1920, 1080) # Default to common HD resolution
        metadata['duration'] = 0 # Default to 0 seconds
    except Exception:
        metadata['resolution'] = None
        metadata['duration'] = None
    return metadata

class VideoSearch(BaseSearchEngine):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.cfg = cfg # Store cfg for reading parameters

    @property
    def model_name(self) -> str:
        # Video module does not use an embedding model for now, return a placeholder name.
        # This will prevent an error from BaseSearchEngine trying to download a non-existent model
        # if the config doesn't specify one or if it's not truly needed yet.
        return None #"video-embedding-model-stub" 
    
    @property
    def cache_prefix(self) -> str:
        return 'videos'

    def get_hash_algorithm(self) -> str:
        """
        Returns the current hashing algorithm identifier used by this engine.
        This is useful for storing in the DB alongside file hashes.
        """
        return "xxh3s:s5m1:v1" # Sampled xxh3_128 with 5 samples of 1 MiB each
    
    def get_file_hash(self, file_path: str) -> str:
        """
        Extremely fast content fingerprint for large files using sampled xxh3_128:
        - Reads fixed-size blocks from head, middle, and tail (samples=3) to minimize I/O.
        - Mixes in file size to reduce collisions between similar files.
        - For small files (<= total sampled bytes), falls back to full-file streaming xxh3_128.
        The cache key includes size and mtime_ns, so recomputation happens only on change.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        st = os.stat(file_path, follow_symlinks=False)
        size = st.st_size
        mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))

        # Sampling params (tuned for speed vs. collision resistance)
        block = 1 * 1024 * 1024  # 1 MiB per sample
        samples = 5               # head, middle, tail pattern

        cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{self.get_hash_algorithm()}"
        cached = self._fast_cache.get(cache_key)
        if cached is not None:
            return cached

        if size <= block * samples:
            # Small files: stream whole file (still very fast)
            digest = self._xxh3_hash_stream(file_path)
            result = f"{digest}"
        else:
            # Large files: sample head/middle/tail
            digest = self._xxh3_hash_sampled(file_path, size=size, block=block, samples=samples)
            result = f"{digest}"

        self._fast_cache.set(cache_key, result)
        return result
    
    def _xxh3_hash_stream(self, file_path: str, chunk_size: int = 16 * 1024 * 1024) -> str:
        h = xxhash.xxh3_128()
        with open(file_path, 'rb', buffering=0) as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                h.update(chunk)
        return h.hexdigest()

    def _xxh3_hash_sampled(self, file_path: str, size: int, block: int, samples: int) -> str:
        # Compute positions: head, evenly spaced, tail
        if samples <= 1:
            positions = [0]
        elif samples == 2:
            positions = [0, max(0, size - block)]
        else:
            step = (size - block) // (samples - 1)
            positions = [min(i * step, max(0, size - block)) for i in range(samples)]
            positions[0] = 0
            positions[-1] = max(0, size - block)

        h = xxhash.xxh3_128()
        with open(file_path, 'rb', buffering=0) as f:
            for pos in positions:
                f.seek(pos, os.SEEK_SET)
                chunk = f.read(block)
                if not chunk:
                    break
                h.update(chunk)
        # Mix file size to reduce collisions across similarly sampled files
        h.update(size.to_bytes(8, byteorder='little', signed=False))
        return h.hexdigest()
        
    def _get_metadata(self, file_path):
        # Even if not actively caching metadata right now, BaseSearchEngine requires this.
        return get_video_metadata_stub(file_path)

    def _get_db_model_class(self):
        return db_models.VideosLibrary
    
    def _get_model_hash_postfix(self):
        return ""

    def _load_model_and_processor(self, local_model_path: str):
        """
        Stub: No actual model loaded for now.
        For future: Implement loading of a video embedding model.
        """
        print(f"VideoSearch: No embedding model loaded yet. Using stub.")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Simulate an embedding dimension for consistency with BaseSearchEngine and Evaluator
        self.embedding_dim = 512 # A common default for many models, can be changed later.
        
        # Set self.model to a dummy torch.nn.Module, so ModelManager can wrap it.
        # This prevents crashes if ModelManager expects a callable model.
        self.model = torch.nn.Linear(self.embedding_dim, self.embedding_dim) # Simple dummy model
        self.model = self.model.cpu() # Ensure it starts on CPU
        print(f"VideoSearch: Stub model initialized on CPU. Embedding dim: {self.embedding_dim}")

    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Stub: Returns a zero tensor as embedding for now.
        Future: Implement actual video frame extraction and embedding.
        """
        # For now, no actual processing, just return a dummy tensor.
        # This will be stored in the DB as a pickled zero tensor if `process_files` is called.
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dimension not set. Call initiate() first.")
        return torch.zeros(1, self.embedding_dim).to(self.device)

    # Public method for consistency with other search engines.
    # It will call super().process_files, which will in turn call _process_single_file.
    def process_videos(self, video_paths: list[str], callback=None, media_folder: str = None) -> torch.Tensor:
        """
        Public method for video processing, calls the base class's logic.
        """
        return super().process_files(video_paths, callback=callback, media_folder=media_folder)

    def process_text(self, text: str) -> torch.Tensor:
        """
        Stub: No text processing for video search implemented yet.
        """
        if self.embedding_dim is None:
            raise RuntimeError("Embedding dimension not set. Call initiate() first.")
        # Return a dummy text embedding (zeros)
        return torch.zeros(1, self.embedding_dim).to(self.device) 
    
    def compare(self, embeds_target, embeds_query):
        return [0.0] 

# Create scoring model singleton class
class VideoEvaluator(src.scoring_models.Evaluator):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VideoEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=512, rate_classes=11): # Use 512 for now for consistency with VideoSearch stub
        if not hasattr(self, '_initialized'): # Prevent re-initialization on subsequent __new__ calls
            super(VideoEvaluator, self).__init__(embedding_dim, rate_classes, name="VideoEvaluator")
            self._initialized = True