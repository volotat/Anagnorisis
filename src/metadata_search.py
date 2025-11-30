import os
import torch
import numpy as np
import traceback
import threading

from src.text_embedder import TextEmbedder
from src.caching import TwoLevelCache

class MetadataSearch:
    """
    Handles metadata-based search.
    Multiple instances allowed; all share one TwoLevelCache.
    """

    _cache_lock = threading.Lock()
    _shared_cache = None  # class-level shared cache instance

    _MAX_META_LINES = 300
    _MAX_META_CHARS = 30_000

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(MetadataSearch, cls).__new__(cls)
    #     return cls._instance

    def __init__(self, engine=None):
        # Prevent re-initialization on subsequent calls
        if hasattr(self, '_initialized') and self._initialized:
            return

        self.engine = engine # Current module's search engine instance
        cfg = engine.cfg if engine else None

        if cfg is None:
            raise ValueError("MetadataSearch requires a configuration object (cfg) on first initialization.")
            
        self.cfg = cfg
        self.text_embedder = TextEmbedder(self.cfg) 

        # Define cache location and TTLs
        cache_folder = os.path.join(self.cfg.main.cache_path, 'metadata_cache')

        if MetadataSearch._shared_cache is None:
            with MetadataSearch._cache_lock:
                if MetadataSearch._shared_cache is None:
                    MetadataSearch._shared_cache = TwoLevelCache(cache_dir=cache_folder, name="metadata_search")

        self._fast_cache = MetadataSearch._shared_cache

        self._initialized = True
    
    def get_algorithm_version(self) -> str:
        """
        Returns the current hashing algorithm identifier used by this engine.
        Used for cache invalidation when the algorithm changes.
        """
        return "meta-search-v1.3" 

    def _read_meta_snippet(self, meta_path: str) -> tuple[str, bool]:
        """
        Read only the first N lines (and cap total chars) to avoid huge I/O and
        long embeddings. Returns (text, truncated_flag).
        """
        lines = []
        total = 0
        truncated = False
        try:
            print(f"Reading metadata file: {meta_path}")
            with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
                for i, line in enumerate(f):
                    if i >= self._MAX_META_LINES or total + len(line) > self._MAX_META_CHARS:
                        truncated = True
                        break
                    lines.append(line)
                    total += len(line)
        except Exception as e:
            print(f"Error reading metadata file {meta_path}: {e}")
        return ''.join(lines), truncated

    def process_query(self, query_text: str) -> np.ndarray:
        return self.text_embedder.embed_query(query_text)
    
    def generate_full_description(self, file_path: str, media_folder: str) -> str:
        """Generates a full metadata description for a single file."""
        media_folder = media_folder or ''
        relative_path = os.path.relpath(file_path, media_folder) if media_folder else os.path.basename(file_path)
        file_name = os.path.basename(file_path)
        
        full_description = f"File Name: {file_name}\nFile Path: {relative_path}\n"
        full_description += "\n"

        # Include basic internal metadata, only textual fields
        internal_meta = self.engine.get_metadata(file_path)
        # Filter out  string that are bigger then ? to avoid base64 blobs and similar big stuff
        max_length = 1000  # Temporary length limit
        if internal_meta:
            full_description += "# Internal metadata:\n"
            for key, value in internal_meta.items():
                if isinstance(value, str) and len(value) <= max_length:
                    full_description += f"{key}: {value}\n"
            full_description += "\n"
        
        # Include file's special {file_name}.meta file content if it exists
        if os.path.exists(file_path + '.meta'):
            full_description += f"# External metadata from '{file_name}.meta' file:\n"
            meta_content, _ = self._read_meta_snippet(file_path + '.meta')
            full_description += meta_content
            full_description += "\n"

        # TODO: Include automatically generated descriptions from VLMs and other ML models. 
        return full_description
    
    def _generate_embedding(self, file_path: str, media_folder: str) -> np.ndarray:
        """Generates a metadata embedding for a single file."""
        meta_text = self.generate_full_description(file_path, media_folder)
        
        meta_embeddings = self.text_embedder.embed_text(meta_text)
        
        if meta_embeddings is not None and len(meta_embeddings) > 0:
            return meta_embeddings[0]
        
        # Fallback to a zero embedding on failure
        dim = self.text_embedder.embedding_dim
        return np.zeros((dim,), dtype=np.float32) if dim else np.array([], dtype=np.float32)

    def _process_single_file_meta(self, file_path: str, media_folder: str = None) -> list[np.ndarray]:
        """
        Processes a single file's metadata, utilizing the cache.
        Returns a list containing one embedding.
        """
        try:
            # 1. Generate a robust cache key from file stats
            file_stat = os.stat(file_path)
            meta_path = file_path + '.meta'
            if os.path.exists(meta_path):
                try:
                    meta_stat = os.stat(meta_path)
                    meta_sig = f"{meta_stat.st_mtime}::{meta_stat.st_size}"
                except Exception:
                    meta_sig = "meta_stat_error"
            else:
                meta_sig = "no_meta"

            cache_key = (
                f"meta::{file_path}::"
                f"{file_stat.st_mtime}::{file_stat.st_size}::"
                f"meta::{meta_sig}::"
                f"alg::{self.get_algorithm_version()}"
            )

            # 2. Check cache
            cached_embedding = self._fast_cache.get(cache_key)
            if cached_embedding is not None:
                return [cached_embedding]

            # 3. Cache Miss: Generate, cache, and return the new embedding
            new_embedding = self._generate_embedding(file_path, media_folder)
            self._fast_cache.set(cache_key, new_embedding)
            return [new_embedding]

        except Exception as e:
            print(f"Error processing metadata for {file_path}: {e}")
            traceback.print_exc()
            # Return a zero embedding on any critical error
            dim = self.text_embedder.embedding_dim
            zero_embedding = np.zeros((dim,), dtype=np.float32) if dim else np.array([], dtype=np.float32)
            return [zero_embedding]
        
    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None, **kwargs) -> list[list[np.ndarray]]:
        """
        Processes metadata for a list of files by calling the single-file processor in a loop.
        Generates embeddings for this metadata text.
        Returns a list of lists of numpy arrays (one list per file, each containing one metadata embedding).
        """
        all_files_meta_embeddings = []
        for ind, file_path in enumerate(file_paths):
            embedding_list = self._process_single_file_meta(file_path, media_folder)
            all_files_meta_embeddings.append(embedding_list)
            
            if callback:
                callback(ind + 1, len(file_paths))
        
        return all_files_meta_embeddings
    
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

        beta = 16.0  # tune (4..20). Higher -> closer to max.
        
        scores = []
        for file_chunks in file_embeddings:
            if not file_chunks:
                scores.append(0.0)
                continue
            
            # TextEmbedder's compare method expects a numpy array of embeddings
            chunk_embeddings_np = np.array(file_chunks, dtype=np.float32)
            
            # Use TextEmbedder's compare logic
            chunk_scores = self.text_embedder.compare(chunk_embeddings_np, query_embedding_np)
                
            # scores.append(max(chunk_scores) if chunk_scores else 0.0)

            # Calculate smooth-max-based score for better differentiation (f(a,b) = ln(e^a + e^b) with normalization to avoid large values for files with many chunks)
            chunk_scores = np.array(chunk_scores)

            m = float(chunk_scores.max())
            x = beta * (chunk_scores - m)
            x = np.clip(x, -50.0, None)  # prevent underflow
            lse_centered = np.log(np.exp(x).sum())
            # Length‑invariant smooth max
            smooth = m + (lse_centered - np.log(len(chunk_scores))) / beta
            # scores.append(float(np.clip(smooth, -1.0, 1.0)))
            scores.append(float(smooth))
            
        return np.array(scores)
    