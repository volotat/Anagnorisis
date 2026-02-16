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
from omegaconf import OmegaConf

import src.scoring_models
import pages.file_manager as file_manager
import pages.text.db_models as db_models
from src.base_search_engine import BaseSearchEngine
from src.text_embedder import TextEmbedder 

from scipy.spatial.distance import cosine


def get_text_metadata(file_path: str):
    """
    Extracts basic metadata from a text file.
    """
    metadata = {}
    try:
        metadata['file_size'] = os.path.getsize(file_path)
        metadata['creation_time'] = os.path.getctime(file_path)
        metadata['modification_time'] = os.path.getmtime(file_path)
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        metadata['file_size'] = None
        metadata['creation_time'] = None
        metadata['modification_time'] = None

    return metadata

class TextSearch(BaseSearchEngine):
    def __init__(self, cfg=None):
        super().__init__(cfg) 
        self.cfg = cfg 
        self.embedder = TextEmbedder(cfg)
        self._embedder_lock = threading.Lock()
        
        # We bypass ModelManager for TextSearch as TextEmbedder handles process isolation
        # We set this to a marker string just in case base class checks for existence,
        # though we override the methods that use it.
        self._model_manager = "Bypassed" 

    @property
    def model_name(self) -> str:
        if self.cfg is None or not hasattr(self.cfg, 'text') or not hasattr(self.cfg.text, 'embedding_model'):
            raise ValueError("Text embedding model not specified in config.")
        return self.cfg.text.embedding_model 
    
    @property
    def cache_prefix(self) -> str:
        return 'text'
        
    def _get_metadata(self, file_path):
        return get_text_metadata(file_path)

    def _get_db_model_class(self):
        return db_models.TextLibrary
    
    def _get_model_hash_postfix(self):
        return "_v1.1.0"
    
    def _get_media_folder(self) -> str:
        if self.cfg is None or not hasattr(self.cfg, 'text') or not hasattr(self.cfg.text, 'media_directory'):
            raise ValueError("Media folder not specified in config.")
        return self.cfg.text.media_directory

    def _load_model_and_processor(self, local_model_path: str):
        # Not used in this subclass as TextEmbedder handles loading
        pass
        
    def _get_model_hash_from_instance(self) -> str:
        """
        Override base method to get hash from embedder.
        """
        if self.embedder.model_hash:
             return self.embedder.model_hash + self._get_model_hash_postfix()
        return "unknown_hash"

    def initiate(self, models_folder: str, cache_folder: str, **kwargs):
        """
        Initializes the TextEmbedder.
        """
        if self.embedder._initialized and self.model_hash:
             return

        print(f"TextSearch: Initiating TextEmbedder...")
        self.embedder.initiate(models_folder=models_folder)
        
        self.embedding_dim = self.embedder.embedding_dim
        self.model_hash = self._get_model_hash_from_instance()
        # Device is managed by embedder process, but we set a default here for compatibility
        self.device = torch.device('cpu') 
        
        print(f"{self.__class__.__name__} initiated successfully. Hash: {self.model_hash}")

    def _process_single_file(self, file_path: str, **kwargs) -> list[np.ndarray]:
        """
        Reads a text file and generates embeddings for each chunk using TextEmbedder.
        Returns a list of numpy arrays.
        """
        if not self.embedder._initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            with self._embedder_lock:
                # embed_text returns np.ndarray [chunks, dim] or empty list
                chunk_embeddings_np = self.embedder.embed_text(content)

            if chunk_embeddings_np is not None and len(chunk_embeddings_np) > 0:
                # Convert [chunks, dim] array to list of [dim] arrays
                return list(chunk_embeddings_np)
            else:
                return []

        except Exception as e:
            print(f"Critical error processing file {file_path}: {e}")
            traceback.print_exc()
            return []

    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None, ignore_cache=False) -> list[list[np.ndarray]]:
        """
        Override process_files to handle list-of-lists return type for text chunks.
        Bypasses BaseSearchEngine.process_files which assumes fixed-size tensor output.
        """
        all_files_chunk_embeddings = []
        
        for ind, file_path in enumerate(tqdm(file_paths, desc=f"Processing {self.cache_prefix} files")):
            current_file_embeddings = None
            try:
                file_hash = self.get_file_hash(file_path)
                cache_key = f"{file_hash}::{self.model_hash}"

                cached_chunk_embeddings = self._fast_cache.get(cache_key) if not ignore_cache else None
                if cached_chunk_embeddings is not None:
                    current_file_embeddings = cached_chunk_embeddings
                else:
                    current_file_embeddings = self._process_single_file(file_path, media_folder=media_folder)
                    self._fast_cache.set(cache_key, current_file_embeddings)

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                traceback.print_exc()
                current_file_embeddings = []
            
            all_files_chunk_embeddings.append(current_file_embeddings)
            
            if callback:
                callback(ind + 1, len(file_paths))
        
        return all_files_chunk_embeddings

    def process_text(self, query_text: str) -> torch.Tensor:
        """
        Generates an embedding for a search query string using TextEmbedder.
        """
        if not self.embedder._initialized:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        
        try:
            query_embedding_np = self.embedder.embed_query(query_text)
            return torch.from_numpy(query_embedding_np)
        except Exception as e:
            print(f"Error processing text query '{query_text}': {e}")
            traceback.print_exc()
            dim = self.cfg.text.embedding_dimension or self.embedding_dim
            return torch.zeros((dim,)) if dim else None
        
    def compare(self, file_embeddings: list[list[np.ndarray]], query_embedding: torch.Tensor) -> np.ndarray:
        """
        Compares a query embedding against file embeddings (list of lists of chunk embeddings).
        """
        # Ensure query embedding is numpy array (float32)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding_np = query_embedding.to(torch.float32).cpu().numpy()
        else:
            query_embedding_np = query_embedding
        
        beta = 16.0  # tune (4..20). Higher -> closer to max.
        
        scores = []
        for file_chunks in file_embeddings:
            if not file_chunks:
                scores.append(0.0)
                continue
            
            # TextEmbedder's compare method expects a numpy array of embeddings
            chunk_embeddings_np = np.array(file_chunks, dtype=np.float32)
            
            # Use TextEmbedder's compare logic
            chunk_scores = self.embedder.compare(chunk_embeddings_np, query_embedding_np)
            
            chunk_scores = np.array(chunk_scores)

            m = float(chunk_scores.max())
            x = beta * (chunk_scores - m)
            x = np.clip(x, -50.0, None)  # prevent underflow
            lse_centered = np.log(np.exp(x).sum())
            # Length‑invariant smooth max
            smooth = m + (lse_centered - np.log(len(chunk_scores))) / beta
            scores.append(float(smooth))
            
        return np.array(scores)
    
# Create scoring model singleton class
class TextEvaluator(src.scoring_models.TransformerEvaluator):
    _instance = None 

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(TextEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=1024, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super(TextEvaluator, self).__init__(embedding_dim, rate_classes, name="TextEvaluator")
            self._initialized = True


# --- Testing Section ---
if __name__ == "__main__":
    import shutil
    
    print("\n" + "="*50)
    print("Running TextSearch Engine Test")
    print("="*50 + "\n")

    # Read dummy text files for testing
    path = os.path.dirname(os.path.abspath(__file__))
    test_text_dir = os.path.join(path, "engine_test_data")

    # Create a dummy config for testing
    dummy_cfg_dict = {
        'main': {
            'embedding_models_path': './models',
            'cache_path': './cache'
        },
        'text': {
            'embedding_model': "jinaai/jina-embeddings-v3", 
            'chunk_size': 128,  
            'chunk_overlap': 0,
            'embedding_dimension': 512,
            'media_directory': test_text_dir
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
        print("Initializing TextSearch engine...")
        text_search_engine = TextSearch(cfg=cfg) 
        text_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path, cfg=cfg)
        print(f"✅ TextSearch engine initialized. Model hash: {text_search_engine.model_hash}")
    except Exception as e:
        print(f"❌ FATAL: TextSearch engine initiation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Test file processing ---
    print("\n" + "="*50)
    print("Test file processing (embedding generation and caching)")
    print("="*50 + "\n")
    
    test_files = [dummy_text_path1, dummy_text_path2, dummy_text_path3]
    
    def test_callback(num_processed, num_total):
        print(f"Processed {num_processed}/{num_total} files...")

    try:
        print("Processing files for the first time (should generate embeddings)...")
        embeddings = text_search_engine.process_files(
            test_files, 
            callback=test_callback, 
            media_folder=test_text_dir,
            ignore_cache=True
        )
        for f, embs in zip(test_files, embeddings):
            print(f"File: {os.path.basename(f)}, Chunks: {len(embs)}")
            if len(embs) > 0:
                 print(f"  First chunk shape: {embs[0].shape}")

        assert len(embeddings) == 3, f"Expected embeddings for 3 files, got {len(embeddings)}"
        # Check first valid file has at least one chunk and correct dim
        print(f"Embedding dim: {text_search_engine.embedding_dim}")
        if len(embeddings[0]) > 0:
            assert embeddings[0][0].shape[0] == text_search_engine.embedding_dim, f"Invalid chunk embedding shape."
        print("✅ First processing successful.")

        print("\nProcessing files again (should use cache)...")
        
        # Obtain presumably cached embeddings
        embeddings_cached = text_search_engine.process_files(
            test_files, 
            callback=test_callback, 
            media_folder=test_text_dir
        )
        assert len(embeddings_cached) == 3, "Cached embeddings count mismatch"
        if len(embeddings[0]) > 0:
            assert np.allclose(embeddings[0][0], embeddings_cached[0][0]), "Cached embeddings do not match original"
        print("✅ Cached processing successful.")

    except Exception as e:
        print(f"❌ File processing test FAILED: {e}")
        import traceback
        traceback.print_exc()

    # --- Test text processing and comparison ---
    print("\n" + "="*50)
    print("Test text processing and comparison")
    print("="*50 + "\n")
    try:
        search_queries = [
            "How do quantum computers use superposition?",
            "Recipe for authentic Italian carbonara",
        ]
        
        for query in search_queries:
            print(f"\nProcessing query: '{query}'")
            query_embedding = text_search_engine.process_text(query)
            
            scores_data = text_search_engine.compare(embeddings, query_embedding)
            
            results = []
            for i, file_path in enumerate(test_files):
                file_name = os.path.basename(file_path)
                score = scores_data[i] if i < len(scores_data) else 0.0
                results.append((file_name, score))
            
            results.sort(key=lambda item: item[1], reverse=True)

            for file_name, score in results:
                print(f"  {file_name}: {score:.4f}")

        print("✅ Text processing and comparison test successful.")
    except Exception as e:
        print(f"❌ Text processing or comparison test FAILED: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*50)
    print("TextSearch Engine Test Completed")
    print("="*50 + "\n")