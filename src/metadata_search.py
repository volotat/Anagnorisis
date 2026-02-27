import os
import torch
import numpy as np
import traceback
import threading
import time
from typing import Optional

from src.text_embedder import TextEmbedder
from src.caching import TwoLevelCache
from src.omni_descriptor import OmniDescriptor

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
        self.omni_descriptor = OmniDescriptor(self.cfg)

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
        return "meta-search-v1.5" 

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Return a human-readable H:M:S string for a given number of seconds."""
        # time.gmtime produces a struct_time in UTC; strftime on that gives H:M:S
        return time.strftime("%Hh %Mm", time.gmtime(seconds))
    
            
    @staticmethod
    def _calculate_progress(processed, total, start_time, max_elapsed: float | None = None):
        """Return (percent, remaining) for progress.
        If ``max_elapsed`` is provided and positive, use it as the per-item
        duration for a pessimistic (longer) estimate. Otherwise fall back to
        the simple average since start_time.
        """
        percent = (processed / total) * 100
        elapsed = time.time() - start_time
        avg = elapsed / processed if processed > 0 else 0

        if max_elapsed is not None and max_elapsed > 0 and processed > 0:
            remaining = (avg*0.2 + max_elapsed*0.8) * (total - processed)
        else:
            remaining = avg * (total - processed)
        return percent, remaining

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
        # Unload to free VRAM, if it was loaded for description generation
        self.omni_descriptor.unload() 
        
        return self.text_embedder.embed_query(query_text)

    def _extension_to_describe_method(self, ext: str) -> Optional[str]:
        """
        Returns the OmniDescriptor method name to use for a given file extension,
        or None if the extension is not handled by any configured module.
        """
        cfg = self.cfg
        try:
            image_fmts = set(cfg.images.media_formats)
        except Exception:
            image_fmts = set()
        try:
            music_fmts = set(cfg.music.media_formats)
        except Exception:
            music_fmts = set()
        try:
            video_fmts = set(cfg.videos.media_formats)
        except Exception:
            video_fmts = set()
        try:
            text_fmts = set(cfg.text.media_formats)
        except Exception:
            text_fmts = set()

        if ext in image_fmts:
            return 'describe_image'
        if ext in music_fmts:
            return 'describe_audio_sampled'
        if ext in video_fmts:
            return 'describe_video_sampled'
        if ext in text_fmts:
            return 'describe_text'
        return None

    def _get_auto_description(self, file_path: str, generate_desc_if_not_in_cache: bool = True) -> str:
        """
        Uses OmniDescriptor to generate an automatic description of the file.
        Cached by (file_hash, omni_model_hash) so recomputation only happens
        when the file content or the descriptor model changes.
        Returns an empty string if OmniDescriptor is unavailable or not yet initiated.
        """

        print(f"Generating auto-description for {file_path} using OmniDescriptor...")

        # Try to get cached description first, using file hash and model hash for invalidation.
        descriptor = self.omni_descriptor

        try:
            file_hash = self.engine.get_file_hash(file_path)
        except Exception:
            raise ValueError(f"Failed to get file hash for {file_path}, cannot generate auto-description.")
        
        ext = os.path.splitext(file_path)[1].lower()
        method_name = self._extension_to_describe_method(ext)
        if method_name is None:
            raise ValueError(f"Unsupported file extension for auto-description: {ext}")

        cache_key = f"auto_desc::{file_hash}::{descriptor.model_hash}::{method_name}"
        cached = self._fast_cache.get(cache_key)
        if cached is not None:
            return cached

        if not generate_desc_if_not_in_cache:
            return None

        # Unload TextEmbedder before starting OmniDescriptor — both are large GPU models
        # and cannot comfortably coexist in VRAM. TextEmbedder auto-restarts on next embed call.
        self.text_embedder.unload()

        # Initiate the descriptor if not already done (lazy loading). This will load the model into VRAM.
        if not getattr(descriptor, 'model_hash', None):
            models_folder = self.cfg.main.embedding_models_path
            descriptor.initiate(models_folder)

        try:
            method = getattr(descriptor, method_name)
            if method_name == 'describe_text':
                # Read text content first; cap at 30 000 chars to stay within model context.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as fh:
                    text_content = fh.read(30_000)
                description = method(text_content)
            else:
                description = method(file_path)
        except Exception as e:
            print(f"[MetadataSearch] Auto-description failed for {file_path}: {e}")
            description = "[Error] Failed to generate auto-description."
            self._fast_cache.set(cache_key, description, save_to_disk=False)  # Cache the failure result to avoid repeated attempts
            return description

        self._fast_cache.set(cache_key, description)

        print(f"Generated auto-description for {file_path} (method: {method_name}): {description[:100]}{'...' if len(description) > 100 else ''}")
        return description

    def generate_full_description(self, file_path: str, media_folder: str) -> str:
        """Generates a full metadata description for a single file."""
        media_folder = media_folder or ''
        relative_path = os.path.relpath(file_path, media_folder) if media_folder else os.path.basename(file_path)
        file_name = os.path.basename(file_path)
        
        full_description = f"File Name: {file_name}\nFile Path: {relative_path}\n"
        full_description += "\n"

        # Automatic description generated by OmniDescriptor (image captioning / audio summary /
        # video description / text summary). Cached by (file_hash, model_hash) — free when
        # the model is not loaded.
        auto_desc = self._get_auto_description(file_path, generate_desc_if_not_in_cache=True)
        if auto_desc:
            full_description += "# Automatic description:\n"
            full_description += auto_desc
            full_description += "\n\n"

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

        return full_description
    
    def _generate_embedding(self, file_path: str, media_folder: str) -> np.ndarray:
        """Generates a metadata embedding for a single file."""
        meta_text = self.generate_full_description(file_path, media_folder)

        # OmniDescriptor (if used) has finished. Unload it before TextEmbedder
        # reclaims VRAM to generate the embedding. OmniDescriptor auto-restarts
        # on the next describe call.
        self.omni_descriptor.unload()

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
                # f"auto_desc_hash::{self.omni_descriptor.model_hash if getattr(self.omni_descriptor, 'model_hash', None) else 'no_desc'}::"
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
        # To avoid getting omni and text embedding models constantly loading and unloading from VRAM, 
        # we want to call '_get_auto_description' on all files in list first to populate the description cache
        # while only omni descriptor is loaded, then when we will call '_process_single_file_meta' it will gather
        # descriptions from cache and only text embedding model will be loaded. This is a bit of a hack but it 
        # allows us to use the cache effectively and avoid VRAM issues.

        total_files = len(file_paths)
        if total_files == 0:
            return []
        
    
        # first pass: automatic descriptions
        start_time = time.time()
        max_elapsed = 0.0
        for ind, file_path in enumerate(file_paths):
            # determine progress using previously observed durations
            if callback:
                percent, remaining = self._calculate_progress(ind, total_files, start_time, max_elapsed)
                callback(
                    f"Extracting automatic descriptions for {ind}/{total_files} ({percent:.2f}%) files... "
                    f"ETA: {self._format_duration(remaining)}"
                )

            file_start = time.time()
            self._get_auto_description(file_path)

            file_elapsed = time.time() - file_start
            if file_elapsed > max_elapsed:
                max_elapsed = file_elapsed

        # second pass: embeddings
        start_time = time.time()
        all_files_meta_embeddings = []
        max_elapsed = 0.0
        for ind, file_path in enumerate(file_paths):
            if callback:
                percent, remaining = self._calculate_progress(ind, total_files, start_time, max_elapsed)
                callback(
                    f"Extracting full metadata embeddings for {ind}/{total_files} ({percent:.2f}%) files... "
                    f"ETA: {self._format_duration(remaining)}"
                )

            file_start = time.time()
            embedding_list = self._process_single_file_meta(file_path, media_folder)

            file_elapsed = time.time() - file_start
            if file_elapsed > max_elapsed:
                max_elapsed = file_elapsed

            all_files_meta_embeddings.append(embedding_list)

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

        # Unload to free VRAM, if it was loaded for description generation
        self.omni_descriptor.unload()  

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
    