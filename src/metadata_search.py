import os
import torch
import numpy as np
import traceback
import threading
import time
import pickle
import hashlib
from src.text_embedder import TextEmbedder


class RAMCache:
    """
    A simple thread-safe in-memory cache with a Time-To-Live (TTL) for each item.
    """
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self._data = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        """
        Retrieves an item from the cache. Returns the item if it exists and has not
        expired, otherwise returns None.
        """
        with self._lock:
            if key not in self._data:
                return None
            
            value, timestamp = self._data[key]
            
            # Check if the item has expired
            if time.time() - timestamp > self.ttl:
                del self._data[key] # Remove expired item
                return None
            
            return value

    def set(self, key: str, value):
        """
        Adds an item to the cache with the current timestamp.
        """
        with self._lock:
            self._data[key] = (value, time.time())

class DiskCache:
    """
    A two-level cache: a fast in-memory RAM cache backed by a persistent,
    sharded disk cache using pickle files.
    """
    def __init__(self, cache_dir: str, disk_ttl_seconds: int, ram_ttl_seconds: int):
        self.cache_dir = cache_dir
        self.disk_ttl = disk_ttl_seconds
        self.ram_cache = RAMCache(ttl_seconds=ram_ttl_seconds)
        self._disk_locks = {} # One lock per shard file
        self._master_lock = threading.Lock() # To manage creation of shard locks
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_shard_path_and_lock(self, key: str):
        """Hashes the key to find the shard path and its corresponding lock."""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        shard_filename = key_hash[:2] + '.pkl'
        shard_path = os.path.join(self.cache_dir, shard_filename)

        with self._master_lock:
            if shard_path not in self._disk_locks:
                self._disk_locks[shard_path] = threading.Lock()
        
        return shard_path, self._disk_locks[shard_path]

    def _load_and_clean_shard(self, shard_path: str) -> dict:
        """Loads a shard from disk and removes expired entries."""
        try:
            with open(shard_path, 'rb') as f:
                shard_data = pickle.load(f)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            return {}

        current_time = time.time()
        # Filter out entries older than the disk TTL (e.g., 3 months)
        cleaned_data = {
            k: (v, ts) for k, (v, ts) in shard_data.items()
            if current_time - ts <= self.disk_ttl
        }

        # If any entries were removed, rewrite the cleaned shard to disk
        if len(cleaned_data) < len(shard_data):
            try:
                with open(shard_path, 'wb') as f:
                    pickle.dump(cleaned_data, f)
            except Exception as e:
                print(f"Warning: Could not rewrite cleaned cache shard {shard_path}: {e}")

        return cleaned_data

    def get(self, key: str):
        """
        Tries to get a value from RAM cache first, then falls back to disk cache.
        """
        # 1. Check RAM cache (fastest)
        value = self.ram_cache.get(key)
        if value is not None:
            return value

        # 2. Check Disk cache
        shard_path, lock = self._get_shard_path_and_lock(key)
        with lock:
            shard_data = self._load_and_clean_shard(shard_path)
            if key in shard_data:
                value, _ = shard_data[key]
                # Promote the found value to the faster RAM cache for next time
                self.ram_cache.set(key, value)
                return value
        
        return None # Cache miss

    def set(self, key: str, value):
        """Sets a value in both RAM and disk caches."""
        # 1. Set in RAM cache
        self.ram_cache.set(key, value)

        # 2. Set in Disk cache
        shard_path, lock = self._get_shard_path_and_lock(key)
        with lock:
            # Load existing data to avoid overwriting other keys in the shard
            shard_data = self._load_and_clean_shard(shard_path)
            shard_data[key] = (value, time.time())
            try:
                with open(shard_path, 'wb') as f:
                    pickle.dump(shard_data, f)
            except Exception as e:
                print(f"Warning: Could not write to cache shard {shard_path}: {e}")


class MetadataSearch:
    """
    A singleton class to handle metadata-based search.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MetadataSearch, cls).__new__(cls)
        return cls._instance

    def __init__(self, cfg=None):
        # Prevent re-initialization on subsequent calls
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        if cfg is None:
            raise ValueError("MetadataSearch requires a configuration object (cfg) on first initialization.")
            
        self.cfg = cfg
        self.text_embedder = TextEmbedder(self.cfg) 

        # Define cache location and TTLs
        cache_folder = os.path.join(self.cfg.main.cache_path, 'metadata_cache')
        three_months_in_seconds = 90 * 24 * 60 * 60
        one_hour_in_seconds = 3600

        # Use the new DiskCache
        self._fast_cache = DiskCache(
            cache_dir=cache_folder,
            disk_ttl_seconds=three_months_in_seconds,
            ram_ttl_seconds=one_hour_in_seconds
        )

        self._initialized = True


    def process_query(self, query_text: str) -> np.ndarray:
        return self.text_embedder.embed_query(query_text)
    
    def _generate_embedding(self, file_path: str, media_folder: str) -> np.ndarray:
        """Generates a metadata embedding for a single file."""
        media_folder = media_folder or ''
        relative_path = os.path.relpath(file_path, media_folder) if media_folder else os.path.basename(file_path)
        file_name = os.path.basename(file_path)
        
        meta_text = f"{file_name}\n{relative_path}"
        
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
            cache_key = f"meta::{file_path}::{file_stat.st_mtime}::{file_stat.st_size}"

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
        
        scores = []
        for file_chunks in file_embeddings:
            if not file_chunks:
                scores.append(0.0)
                continue
            
            # TextEmbedder's compare method expects a numpy array of embeddings
            chunk_embeddings_np = np.array(file_chunks, dtype=np.float32)
            
            # Use TextEmbedder's compare logic
            chunk_scores = self.text_embedder.compare(chunk_embeddings_np, query_embedding_np)
                
            scores.append(max(chunk_scores) if chunk_scores else 0.0)
            
        return np.array(scores)
    