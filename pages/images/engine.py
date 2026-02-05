from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel #, SiglipVisionModel, SiglipTextModel - these are no longer needed directly
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import hashlib
import datetime
import io
import time
import cv2
import imageio
import threading

import src.scoring_models
import pages.images.db_models as db_models
import pages.file_manager as file_manager
from src.base_search_engine import BaseSearchEngine # Import the base class
from src.model_manager import ModelManager # Still needed for ImageEvaluator, but ModelManager is central to BaseSearchEngine


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
        self.tokenizer = None # Will be set in _load_model_and_processor
        self._embedder_lock = threading.Lock()
        self.is_embedder_busy = False
  
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

    def _load_model_and_processor(self, local_model_path: str):
        """
        Loads the SigLIP model and processor from the local path.
        Ensures the model is loaded to CPU before being wrapped by ModelManager.
        """
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"ImageSearch: Loading model to CPU first...")
            # Load from the specific local path, ensuring local_files_only=True
            self.model = AutoModel.from_pretrained(local_model_path, local_files_only=True).cpu() # Load to CPU
            self.processor = AutoProcessor.from_pretrained(local_model_path, local_files_only=True)
            self.embedding_dim = self.model.config.text_config.hidden_size # Safely access config
            print(f"ImageSearch: Model loaded to CPU. Embedding dim: {self.embedding_dim}")

        except Exception as e:
            print(f"ERROR: Failed to load SigLIP model from '{local_model_path}'. The download might be incomplete or corrupted.")
            print(f"Error details: {e}")
            raise RuntimeError(f"Failed to load required model: {self.model_name}") from e

            

    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Reads and preprocesses a single image, then generates its embedding.
        """
        image = self._read_image(file_path)
        if image is None:
            raise Exception(f"Failed to read image: {file_path}")
        
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        image_embeds = None
        
        with self._embedder_lock:
            self.is_embedder_busy = True
            try:
                # Use the wrapped model (self.model is now ModelManager instance)
                # Calling model_instance.get_image_features() will trigger ModelManager to load to GPU
                model_instance = self.model 
                
                inputs_images = self.processor(images=[image], padding="max_length", return_tensors="pt").to(self.device) # Use managed device
                
                with torch.no_grad():
                    outputs = model_instance.get_image_features(**inputs_images)

                # Normalize the image embeddings (important for similarity)
                image_embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

                return image_embeds
            finally:
                self.is_embedder_busy = False


    def _read_image(self, image_path: str):
        """Helper method to read image files, handling various formats and conversions."""
        try:
            if image_path.lower().endswith('.gif'):
                gif = imageio.mimread(image_path)
                image = np.array(gif[0])
            else:
                image = cv2.imread(image_path)

            if image is None:
                return None

            # Convert to RGB (OpenCV reads BGR by default, grayscale might be 2D)
            if len(image.shape) == 2 or image.shape[2] == 1:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                raise ValueError("Invalid image shape")
            return image
        
        except OSError as e:
            print(f"Error reading image {image_path}: {e}")
            return None

    def process_images(self, images: list[str], callback=None, media_folder: str = None) -> torch.Tensor:
        """
        Public method for image processing, now calls the base class's logic.
        Kept for backward compatibility with existing calls in serve.py.
        """
        return super().process_files(images, callback=callback, media_folder=media_folder)

    def process_text(self, text: str) -> torch.Tensor:
        """
        Processes a text query to generate its embedding using the SigLIP text encoder.
        """
        if self.model is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        with self._embedder_lock:
            self.is_embedder_busy = True
            try:
                model_instance = self.model # Triggers loading to GPU if needed

                inputs_text = self.processor(text=text, padding="max_length", return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = model_instance.get_text_features(**inputs_text)

                text_embeds = outputs / outputs.norm(p=2, dim=-1, keepdim=True)

                # TODO: Find out why .squeeze(0) is needed here, it was working without it before refactoring
                return text_embeds.squeeze(0)  # Return as 1D tensor
            finally:
                self.is_embedder_busy = False

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
        print(f"Model on device: {image_search_engine.model.device}")
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
        # Expect 2 embeddings (dummy_image_path3 fails)
        assert embeddings.shape[0] == 3, f"Expected 3 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == image_search_engine.embedding_dim, f"Expected embedding dim {image_search_engine.embedding_dim}, got {embeddings.shape[1]}"
        print(f"{colorama.Fore.GREEN}First processing successful.{colorama.Style.RESET_ALL}")

        print(f"\n{colorama.Fore.CYAN}Processing files again (should use cache):{colorama.Style.RESET_ALL}")

        # TODO: New caching is working with TwoLevelCache instead of in-memory dict
        # The test related to caching needs to be adapted accordingly.

        # Clear in-memory cache to force loading from disk/DB cache
        image_search_engine._fast_cache = {} 
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

    # --- Test ModelManager lazy loading/unloading ---
    print("\n--- Testing ModelManager lazy loading ---")
    # At this point, the model should be loaded on GPU because we just used it.
    print(f"Model manager loaded: {image_search_engine.model._loaded}")
    print(f"Model device: {image_search_engine.model._model.device}")
    assert image_search_engine.model._loaded and image_search_engine.model._model.device.type == 'cuda', "Model not loaded to GPU after use."
    
    print("Waiting for idle timeout (120 seconds + 30s cleanup check)...")
    # Give enough time for the cleanup thread to potentially unload
    time.sleep(160) # Wait for a bit longer than cleanup check interval
    
    # Check if model is unloaded
    print(f"Model manager loaded after idle: {image_search_engine.model._loaded}")
    assert not image_search_engine.model._loaded, "Model was not unloaded after idle period."
    print(f"{colorama.Fore.GREEN}ModelManager unloading test successful.{colorama.Style.RESET_ALL}")

    # Re-use the model to ensure it reloads
    print("\nRe-using model to trigger reload...")
    dummy_text = "This is a test text to trigger model reloading."
    query_embedding = image_search_engine.process_text(dummy_text)

    print(f"Model manager loaded after re-use: {image_search_engine.model._loaded}")
    print(f"Model device after re-use: {image_search_engine.model._model.device}")
    assert image_search_engine.model._loaded and image_search_engine.model._model.device.type == 'cuda', "Model did not reload to GPU on re-use."
    print(f"{colorama.Fore.GREEN}ModelManager reloading test successful.{colorama.Style.RESET_ALL}")

    # Shut down ModelManager gracefully at the end of tests
    ModelManager.shutdown()
    print("\n--- ImageSearch Engine Test Completed ---")