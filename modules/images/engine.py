from PIL import Image
import requests
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import hashlib
import datetime
import io
import time

import src.scoring_models
import modules.images.db_models as db_models
import src.file_manager as file_manager
from src.base_search_engine import BaseSearchEngine
from src.image_embedder import ImageEmbedder


def get_image_metadata(file_path):
    """
    Extracts relevant metadata from an image file and returns a flat dictionary
    with string values suitable for metadata search.
    """
    metadata = {}
    max_value_length = 1000  # Skip very long values (e.g., base64 data)
    
    try:
        with Image.open(file_path) as img:
            # Basic image properties
            metadata['format'] = str(img.format) if img.format else 'Unknown'
            metadata['mode'] = str(img.mode) if img.mode else 'Unknown'
            
            if img.size:
                width, height = img.size
                metadata['width'] = str(width)
                metadata['height'] = str(height)
                metadata['resolution'] = f"{width}x{height}"
            
            # Extract EXIF data if available
            exif_data = img.getexif()
            if exif_data:
                from PIL.ExifTags import TAGS
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, f'EXIF_{tag_id}')
                    
                    # Convert bytes to string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore').strip()
                        except:
                            continue
                    
                    # Convert to string and filter by length
                    value_str = str(value)
                    if len(value_str) <= max_value_length and value_str.strip():
                        metadata[str(tag)] = value_str
            
            # Extract general info dict if available (PNG, GIF, etc.)
            if hasattr(img, 'info') and img.info:
                for key, value in img.info.items():
                    # Skip binary/bytes data
                    if isinstance(value, bytes):
                        continue
                    
                    # Convert to string and filter
                    if isinstance(value, (str, int, float, bool)):
                        value_str = str(value)
                        if len(value_str) <= max_value_length and value_str.strip():
                            # Prefix with 'INFO_' to distinguish from EXIF
                            metadata[f'INFO_{key}'] = value_str
                
    except Exception as e:
        # On error, return basic info
        metadata['error'] = f"Failed to extract metadata: {str(e)[:200]}"
        
    return metadata


class ImageSearch(BaseSearchEngine):
    def __init__(self, cfg=None):
        super().__init__(cfg) # Call base class __init__ to set up singleton and flags
        self.cfg = cfg # Store cfg for reading parameters
        self._image_embedder = None

    @property
    def model_name(self) -> str:
        # Use cfg.images.embedding_model for dynamic model selection
        if self.cfg is None or not hasattr(self.cfg, 'images') or not hasattr(self.cfg.images, 'embedding_model'):
            raise ValueError("Images embedding model not specified in config.")
        return self.cfg.images.embedding_model # e.g., "google/siglip-base-patch16-224"
  
    @property
    def cache_prefix(self) -> str:
        return 'images'
      
    def _get_metadata(self, file_path):
        return get_image_metadata(file_path)

    def _get_db_model_class(self):
        return db_models.ImagesLibrary
    
    def _get_model_hash_postfix(self):
        return "_v1.0.1"
    
    def _get_media_folder(self) -> str:
        if self.cfg is None or not hasattr(self.cfg, 'images') or not hasattr(self.cfg.images, 'media_directory'):
            raise ValueError("Media folder not specified in config.")
        return self.cfg.images.media_directory

    def initiate(self, models_folder: str, cache_folder: str = None, **kwargs):
        """
        Override base initiate: delegate model loading to the ImageEmbedder subprocess.
        The SigLIP model never lives in the main process, freeing VRAM between calls.
        """
        if self._model_manager is not None:
            return

        self._image_embedder = ImageEmbedder(cfg=self.cfg)
        res = self._image_embedder.initiate(models_folder=models_folder)

        # Mirror attributes needed by BaseSearchEngine helpers and EmbeddingProxyGenerator.
        self.embedding_dim = self._image_embedder.embedding_dim
        self.model_hash    = self._image_embedder.model_hash
        self.device        = self._image_embedder.device

        # Sentinel: satisfies BaseSearchEngine.compare()'s None guard and re-init guard.
        self._model_manager = True

        print(f"ImageSearch initiated successfully (via ImageEmbedder subprocess).")

    def _load_model_and_processor(self, local_model_path: str):
        # Never called: initiate() is fully overridden above.
        pass



    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Generates a SigLIP image embedding via the ImageEmbedder subprocess.
        Returns a [1, embedding_dim] tensor.
        """
        if self._image_embedder is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        arr = self._image_embedder.embed_image(file_path)  # np.ndarray (1, D)
        return torch.from_numpy(arr)

    def process_images(self, images: list[str], callback=None, media_folder: str = None) -> torch.Tensor:
        """
        Public method for image processing, now calls the base class's logic.
        Kept for backward compatibility with existing calls in serve.py.
        """
        return super().process_files(images, callback=callback, media_folder=media_folder)

    def process_text(self, text: str) -> torch.Tensor:
        """
        Generates a SigLIP text embedding via the ImageEmbedder subprocess.
        Returns a 1-D tensor of shape (embedding_dim,).
        """
        if self._image_embedder is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        arr = self._image_embedder.embed_text(text)  # np.ndarray (D,)
        return torch.from_numpy(arr)

# Create scoring model singleton class so it easily accessible from other modules
class ImageEvaluator(src.scoring_models.Evaluator):
    _instance = None # This is important for the singleton pattern on the evaluator as well

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ImageEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=768, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super(ImageEvaluator, self).__init__(embedding_dim, rate_classes, name="ImageEvaluator")
            self._initialized = True


# --- Testing Section ---
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import colorama  # For colored terminal output
    
    # Initialize colorama for cross-platform colored terminal output
    colorama.init()

    print("--- Running ImageSearch Engine Test ---")

    # Create a dummy config for testing
    dummy_cfg_dict = {
        'main': {
            'embedding_models_path': './models',
            'cache_path': './cache'
        },
        'images': {
            'embedding_model': "google/siglip-base-patch16-224",
            'media_formats': ['.jpg', '.png', '.jpeg']
        }
    }
    cfg = OmegaConf.create(dummy_cfg_dict)

    # Ensure directories exist
    os.makedirs(cfg.main.embedding_models_path, exist_ok=True)
    os.makedirs(cfg.main.cache_path, exist_ok=True)

    # Create dummy image files for testing
    path = os.path.dirname(os.path.abspath(__file__))
    test_image_dir = os.path.join(path, "engine_test_data")
    os.makedirs(test_image_dir, exist_ok=True)

    dummy_image_path1 = os.path.join(test_image_dir, "cat.jpeg")
    dummy_image_path2 = os.path.join(test_image_dir, "dog.jpeg")
    dummy_image_path3 = os.path.join(test_image_dir, "flower.jpeg") 
    dummy_image_path4 = os.path.join(test_image_dir, "icecream.jpeg")  

    # --- Initialize the model ---
    try:
        print("\nInitializing ImageSearch engine...")
        image_search_engine = ImageSearch(cfg=cfg)
        image_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)
        print(f"ImageSearch engine initialized. Model hash: {image_search_engine.model_hash}")
        print(f"Model on device: {image_search_engine.device}")
    except Exception as e:
        print(f"FATAL: ImageSearch engine initiation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Test file processing ---
    print("\n--- Test file processing (embedding generation and caching) ---")
    test_files = [dummy_image_path1, dummy_image_path2, dummy_image_path3, dummy_image_path4]
    
    def test_callback(num_processed, num_total):
        print(f"Processed {num_processed}/{num_total} files...")

    try:
        print(f"{colorama.Fore.CYAN}Processing files for the first time (should generate embeddings):{colorama.Style.RESET_ALL}")
        embeddings = image_search_engine.process_images(
            test_files, 
            callback=test_callback, 
            media_folder=test_image_dir
        )
        print(f"Generated embeddings shape: {embeddings.shape}")
        # Expect 4 embeddings (all dummy images should succeed)
        assert embeddings.shape[0] == 4, f"Expected 4 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == image_search_engine.embedding_dim, f"Expected embedding dim {image_search_engine.embedding_dim}, got {embeddings.shape[1]}"
        print(f"{colorama.Fore.GREEN}First processing successful.{colorama.Style.RESET_ALL}")

        print(f"\n{colorama.Fore.CYAN}Processing files again (should use cache):{colorama.Style.RESET_ALL}")

        # TODO: New caching is working with TwoLevelCache instead of in-memory dict
        # The test related to caching needs to be adapted accordingly.

        # Clear in-memory cache to force loading from disk/DB cache
        image_search_engine._fast_cache.ram._data.clear()
        embeddings_cached = image_search_engine.process_images(
            test_files, 
            callback=test_callback, 
            media_folder=test_image_dir
        )
        print(f"Generated embeddings shape (cached): {embeddings_cached.shape}")
        assert torch.allclose(embeddings, embeddings_cached), "Cached embeddings do not match original"
        print(f"{colorama.Fore.GREEN}Cached processing successful.{colorama.Style.RESET_ALL}")

    except Exception as e:
        print(f"{colorama.Fore.RED}File processing test FAILED: {e}{colorama.Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

    # --- Test text processing and comparison ---
    print("\n--- Test text processing and comparison ---")
    try:
        # Define realistic search queries
        search_queries = [
            "a photo of a cat",
            "a photo of a dog",
            "a photo of a flower",
            "picture of an ice cream",
        ]

        for query in search_queries:
            print(f"\nProcessing query: '{query}'")
            query_embedding = image_search_engine.process_text(query)
            print(f"Query embedding shape: {query_embedding.shape}")
            
            scores_data = image_search_engine.compare(embeddings, query_embedding)
            
            print(f"Scores for '{query}'")

            # Combine file paths with their scores
            results = []
            for i, file_path in enumerate(test_files):
                file_name = os.path.basename(file_path)
                score = scores_data[i] if i < len(scores_data) else 0.0
                results.append((file_name, score))

            # Sort results by score in descending order
            results.sort(key=lambda item: item[1], reverse=True)

            # Print sorted results
            for file_name, score in results:
                print(f"  {file_name}: {score:.4f}")

    except Exception as e:
        print(f"{colorama.Fore.RED}Text processing or comparison test FAILED: {e}{colorama.Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

    # NOTE: Subprocess idle/restart lifecycle is tested in src/image_embedder.py.
    print("\n--- ImageSearch Engine Test Completed ---")