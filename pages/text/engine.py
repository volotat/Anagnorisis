import os
import pickle
import hashlib
import datetime
import io
import time
import gc
import traceback
import threading

import flask
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer # Sentencetransformer model will include its own tokenizer
from sentence_transformers import SentenceTransformer 
from huggingface_hub import snapshot_download

import src.scoring_models
import pages.file_manager as file_manager
import pages.text.db_models as db_models
from src.base_search_engine import BaseSearchEngine # Import the base class
from src.model_manager import ModelManager # Still needed for TextEvaluator
from src.text_embedder import TextEmbedder # Import the new TextEmbedder

from scipy.spatial.distance import cosine


def get_text_metadata(file_path: str):
    """
    Extracts basic metadata from a text file.
    Can be extended later for more fields like creation date, modification date, etc.
    """
    metadata = {}
    try:
        # Get file size
        metadata['file_size'] = os.path.getsize(file_path)
        # Get creation time (Unix timestamp)
        metadata['creation_time'] = os.path.getctime(file_path)
        # Get modification time (Unix timestamp)
        metadata['modification_time'] = os.path.getmtime(file_path)
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        metadata['file_size'] = None
        metadata['creation_time'] = None
        metadata['modification_time'] = None

    return metadata


class TextSearch(BaseSearchEngine):
    def __init__(self, cfg=None):
        super().__init__(cfg) # Call base class __init__
        self.cfg = cfg # Store cfg for chunking parameters
        self.embedder = TextEmbedder(cfg)
        self.tokenizer = None # Will be set in _load_model_and_processor
        self._embedder_lock = threading.Lock()
        self.is_embedder_busy = False

    @property
    def model_name(self) -> str:
        # Use cfg.text.embedding_model for dynamic model selection
        if self.cfg is None or not hasattr(self.cfg, 'text') or not hasattr(self.cfg.text, 'embedding_model'):
            raise ValueError("Text embedding model not specified in config.")
        return self.cfg.text.embedding_model # e.g., "jinaai/jina-embeddings-v3"
    
    @property
    def cache_prefix(self) -> str:
        return 'text'
        
    def _get_metadata(self, file_path):
        # TextSearch does not currently use cached_metadata, but the function is needed for BaseSearchEngine
        # It's okay to return a dummy if metadata isn't actively extracted and cached in the same way
        # For now, get_text_metadata is a simple function
        return get_text_metadata(file_path)

    def _get_db_model_class(self):
        return db_models.TextLibrary
    
    def _get_model_hash_postfix(self):
        return "_v1.0.4"

    def _load_model_and_processor(self, local_model_path: str):
        """
        Loads the SentenceTransformer model and its tokenizer from the local path.
        Ensures the model is loaded to CPU before being wrapped by ModelManager.
        """
        try:
            # The TextEmbedder handles its own model loading and management.
            # We pass the base models folder to its initiate method.
            models_folder = os.path.dirname(local_model_path)
            self.embedder.initiate(models_folder=models_folder)
            
            # For BaseSearchEngine compatibility, we need to set self.model and self.embedding_dim
            self.model = self.embedder.model
            self.tokenizer = self.embedder.tokenizer
            self.embedding_dim = self.embedder.embedding_dim
            self.device = self.embedder.device
            
            print(f"TextSearch: TextEmbedder initiated. Embedding dim: {self.embedding_dim}")

        except Exception as e:
            print(f"ERROR: Failed to load text embedding model from '{local_model_path}'. The download might be incomplete or corrupted.")
            print(f"Error details: {e}")
            raise RuntimeError(f"Failed to load required model: {self.model_name}") from e

    def _process_single_file(self, file_path: str, **kwargs) -> list[np.ndarray]:
        """
        Reads a text file and generates embeddings for each chunk using TextEmbedder.
        Returns a list of numpy arrays (each representing a chunk embedding).
        """
        if self.embedder is None or not self.embedder._initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        try:
            # 1. Read File Content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 2. Generate Embeddings for Chunks via TextEmbedder
            with self._embedder_lock:
                self.is_embedder_busy = True
                try:
                    chunk_embeddings_list_np = self.embedder.embed_text(content)
                finally:
                    self.is_embedder_busy = False

            # embed_text returns a numpy array of embeddings. Convert to a list of arrays.
            if chunk_embeddings_list_np is not None and len(chunk_embeddings_list_np) > 0:
                return [emb for emb in chunk_embeddings_list_np]
            else:
                return []

        except Exception as e:
            print(f"Critical error processing file {file_path}: {e}")
            traceback.print_exc()
            return [] # Return empty list on critical error

    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None) -> list[list[np.ndarray]]:
        """
        Public method for text file processing, now calls the base class's logic.
        Note: The base class's `process_files` expects `_process_single_file` to return a `torch.Tensor`.
        However, for text, we return a `list[np.ndarray]` (chunk embeddings).
        This means `super().process_files` needs to be adapted or this method handles the loop itself.
        
        Given the current `TextSearch.compare` expects `List[List[np.ndarray]]`,
        it's better to override the *entire* `process_files` method here to handle this list of lists directly
        rather than forcing `_process_single_file` to return a concatenated tensor.
        """
        
        all_files_chunk_embeddings = [] # Accumulate list of list of numpy arrays
        
        for ind, file_path in enumerate(tqdm(file_paths, desc=f"Processing {self.cache_prefix} files")):
            current_file_embeddings = None
            try:
                file_hash = self.get_file_hash(file_path)
                cache_key = f"{file_hash}::{self.model_hash}{self._get_model_hash_postfix()}"

                cached_chunk_embeddings = self._fast_cache.get(cache_key)
                if cached_chunk_embeddings is not None:
                    current_file_embeddings = cached_chunk_embeddings
                else:
                    current_file_embeddings = self._process_single_file(file_path, media_folder=media_folder)
                    self._fast_cache.set(cache_key, current_file_embeddings)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                traceback.print_exc()
                current_file_embeddings = [] # On error, return empty list of chunks
            
            all_files_chunk_embeddings.append(current_file_embeddings)
            
            if callback:
                callback(ind + 1, len(file_paths))
        
        return all_files_chunk_embeddings # Returns List[List[np.ndarray]]

    def process_text(self, query_text: str) -> torch.Tensor:
        """
        Generates an embedding for a search query string using TextEmbedder.
        Uses 'retrieval.query' task and configured truncation dimension.
        """
        if self.embedder is None or not self.embedder._initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        
        try:
            # Use TextEmbedder to get the numpy array embedding
            query_embedding_np = self.embedder.embed_query(query_text)
            
            # Convert to a PyTorch tensor for compatibility with the rest of the system
            return torch.from_numpy(query_embedding_np).to(self.device)
        except Exception as e:
            print(f"Error processing text query '{query_text}': {e}")
            traceback.print_exc()
            dim = self.cfg.text.embedding_dimension or self.embedding_dim
            return torch.zeros((1, dim), device=self.device) if dim else None
        
    def compare(self, file_embeddings: list[list[np.ndarray]], query_embedding: torch.Tensor) -> np.ndarray:
        """
        Compares a query embedding against file embeddings (list of lists of chunk embeddings).
        Calculates relevance scores (max similarity across all chunks in each file).
        """
        # Ensure query embedding is numpy array (float32)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding_np = query_embedding.to(torch.float32).cpu().numpy()
        else:
            query_embedding_np = query_embedding
        
        scores = []
        for file_chunks in file_embeddings:
            if not file_chunks:
                scores.append(0.0)
                continue
            
            # TextEmbedder's compare method expects a numpy array of embeddings
            chunk_embeddings_np = np.array(file_chunks, dtype=np.float32)
            
            # Use TextEmbedder's compare logic
            chunk_scores = self.embedder.compare(chunk_embeddings_np, query_embedding_np)
                
            scores.append(max(chunk_scores) if chunk_scores else 0.0)
            
        return np.array(scores)
    
# Create scoring model singleton class
class TextEvaluator(src.scoring_models.Evaluator):
    _instance = None # This is important for the singleton pattern on the evaluator as well

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=1024, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super(TextEvaluator, self).__init__(embedding_dim, rate_classes)
            self._initialized = True


# --- Testing Section ---
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import tempfile
    import glob
    import re
    import colorama
    import shutil
    
    colorama.init()

    print("--- Running TextSearch Engine Test ---")

    # Create a dummy config for testing
    dummy_cfg_dict = {
        'main': {
            'embedding_models_path': './models',
            'cache_path': './cache'
        },
        'text': {
            'embedding_model': "jinaai/jina-embeddings-v3", # This model is large, test carefully
            'chunk_size': 128,  # Smaller chunk size for quick tests
            'chunk_overlap': 0,
            'embedding_dimension': 512 # Use model's native dimension or set a custom one
        }
    }
    cfg = OmegaConf.create(dummy_cfg_dict)

    os.makedirs(cfg.main.embedding_models_path, exist_ok=True)
    os.makedirs(cfg.main.cache_path, exist_ok=True)

    # Read dummy text files for testing
    path = os.path.dirname(os.path.abspath(__file__))
    test_text_dir = os.path.join(path, "engine_test_data")
    os.makedirs(test_text_dir, exist_ok=True)
    
    dummy_text_path1 = os.path.join(test_text_dir, "quantum_computing.txt")
    dummy_text_path2 = os.path.join(test_text_dir, "python_data_analysis.txt")
    dummy_text_path3 = os.path.join(test_text_dir, "italian_recipes.txt")

    # --- Initialize the model ---
    try:
        print("\nInitializing TextSearch engine...")
        text_search_engine = TextSearch(cfg=cfg) # Pass cfg to constructor
        text_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path, cfg=cfg)
        print(f"TextSearch engine initialized. Model hash: {text_search_engine.model_hash}")
        print(f"Model on device: {text_search_engine.model.device}")
    except Exception as e:
        print(f"FATAL: TextSearch engine initiation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Test file processing ---
    print("\n--- Test file processing (embedding generation and caching) ---")
    test_files = [dummy_text_path1, dummy_text_path2, dummy_text_path3]
    
    def test_callback(num_processed, num_total):
        print(f"Processed {num_processed}/{num_total} files...")

    try:
        print(f"{colorama.Fore.CYAN}Processing files for the first time (should generate embeddings):{colorama.Style.RESET_ALL}")
        embeddings = text_search_engine.process_files(
            test_files, 
            callback=test_callback, 
            media_folder=test_text_dir
        )
        for f, embs in zip(test_files, embeddings):
            print(f"File: {os.path.basename(f)}, Chunks: {len(embs)}, Shape: {embs[0].shape}")
        #print(f"Embeddings:", embeddings) # Print number of chunks per file
        # Expect 2 sets of chunk embeddings (dummy_text_path3 fails)
        assert len(embeddings) == 3, f"Expected embeddings for 3 files, got {len(embeddings)}"
        # Check first valid file has at least one chunk and correct dim
        print(f"Embedding dim: {text_search_engine.embedding_dim}")
        print(f"First file first chunk shape: {embeddings[0][0].shape}")
        assert len(embeddings[0]) > 0 and embeddings[0][0].shape[0] == text_search_engine.embedding_dim, f"Invalid chunk embedding shape. {embeddings[0][0].shape} instead of {text_search_engine.embedding_dim}"
        print(f"{colorama.Fore.GREEN}First processing successful.{colorama.Style.RESET_ALL}")

        print(f"\n{colorama.Fore.CYAN}Processing files again (should use cache):{colorama.Style.RESET_ALL}")
        
        # TODO: New caching is working with TwoLevelCache instead of in-memory dict
        # The test related to caching needs to be adapted accordingly.

        text_search_engine._fast_cache = {} # Clear in-memory cache
        embeddings_cached = text_search_engine.process_files(
            test_files, 
            callback=test_callback, 
            media_folder=test_text_dir
        )
        assert len(embeddings_cached) == 3, "Cached embeddings count mismatch"
        # Compare first valid file's first chunk embedding
        assert np.allclose(embeddings[0][0], embeddings_cached[0][0]), "Cached embeddings do not match original"
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
            "How do quantum computers use superposition?",
            "What is the difference between trapped ions and superconducting qubits?",
            "How to create visualizations with pandas and matplotlib",
            "Steps to handle missing values in data analysis",
            "Recipe for authentic Italian carbonara",
            "What temperature should I bake pizza at?"
        ]
        
        for query in search_queries:
            print(f"\nProcessing query: '{query}'")
            query_embedding = text_search_engine.process_text(query)
            
            scores_data = text_search_engine.compare(embeddings, query_embedding)
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

        print(f"{colorama.Fore.GREEN}Text processing and comparison test successful.{colorama.Style.RESET_ALL}")
    except Exception as e:
        print(f"{colorama.Fore.RED}Text processing or comparison test FAILED: {e}{colorama.Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

    # --- Test ModelManager lazy loading/unloading ---
    print("\n--- Testing ModelManager lazy loading ---")
    print(f"Model manager loaded: {text_search_engine.model._loaded}")
    print(f"Model device: {text_search_engine.model.device}")
    assert text_search_engine.model._loaded and text_search_engine.model.device.type == 'cuda', "Model not loaded to GPU after use."
    
    print("Waiting for idle timeout (120 seconds + for 30s cleanup)...")
    time.sleep(160)
    
    print(f"Model manager loaded after idle: {text_search_engine.model._loaded}")
    assert not text_search_engine.model._loaded, "Model was not unloaded after idle period."
    print(f"{colorama.Fore.GREEN}ModelManager unloading test successful.{colorama.Style.RESET_ALL}")

    print("\nRe-using model to trigger reload...")
    # Process dummy inputs by the model manager to trigger reloading
    dummy_text = "This is a test text to trigger model reloading."
    query_embedding = text_search_engine.process_text(dummy_text)

    print(f"Model manager loaded after re-use: {text_search_engine.model._loaded}")
    print(f"Model device after re-use: {text_search_engine.model.device}")
    assert text_search_engine.model._loaded and text_search_engine.model.device.type == 'cuda', "Model did not reload to GPU on re-use."
    print(f"{colorama.Fore.GREEN}ModelManager reloading test successful.{colorama.Style.RESET_ALL}")

    # Shut down ModelManager gracefully at the end of tests
    ModelManager.shutdown()
    print("\n--- TextSearch Engine Test Completed ---")