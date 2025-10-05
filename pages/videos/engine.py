import os
import torch
import numpy as np
# from PIL import Image # Not strictly needed for metadata stub, but often part of video processing
from src.base_search_engine import BaseSearchEngine
from src.model_manager import ModelManager
import src.scoring_models
import pages.videos.db_models as db_models

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
        
    def _get_metadata_function(self):
        # Even if not actively caching metadata right now, BaseSearchEngine requires this.
        return get_video_metadata_stub

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
            super(VideoEvaluator, self).__init__(embedding_dim, rate_classes)
            self._initialized = True