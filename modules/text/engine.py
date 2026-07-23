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

import src.file_manager as file_manager
import modules.text.db_models as db_models
import src.db_models as main_db_models
from src.base_search_engine import BaseSearchEngine
from src.text_embedder import TextEmbedder 

from scipy.spatial.distance import cosine
import src.virtual_file_system as vfs
import fs



def get_text_metadata(file_path: str) -> dict:
    """
    Extracts basic metadata from a text file via VFS-aware stat calls.

    Text files have no embedded metadata block (unlike images' EXIF or
    audio's ID3 tags), so we expose only filesystem-level information
    that helps the embedding understand the file:

      - file_size        : integer byte count
      - modification_time: ISO 8601 datetime of last modification
      - creation_time    : ISO 8601 datetime of creation (if FS supports it)
      - access_time      : ISO 8601 datetime of last access (if FS supports it)
      - extension        : lowercase file extension (txt, md, html, …)
      - content_type     : human-readable hint derived from extension

    File content is NEVER read, so this stays O(1) regardless of file
    size — important since text libraries often contain multi-GB logs.

    All values are stringified so that ``MetadataSearch.generate_full_description``
    can drop them straight into the embedding payload.
    """
    metadata = {}
    try:
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)

        with fs.open_fs(base_url) as my_fs:
            info = my_fs.getinfo(path_in_fs, namespaces=['details'])

            # File size — cheap, useful as a "document size" hint.
            if getattr(info, 'size', None) is not None:
                metadata['file_size'] = f"{info.size} bytes"

            # Timestamps — stringified via isoformat() so they fit
            # directly into the embedding text payload.
            for attr, key in (
                ('modified', 'modification_time'),
                ('created',  'creation_time'),
                ('accessed', 'access_time'),
            ):
                dt = getattr(info, attr, None)
                if dt is None:
                    continue
                metadata[key] = (
                    dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                )

            # Extension — the single most useful field for text files.
            # Cheapest possible computation, and it tells the embedding
            # "this is markdown" / "this is JSON" before it has seen
            # a single line of content.
            ext = os.path.splitext(path_in_fs)[1].lstrip('.').lower()
            if ext:
                metadata['extension'] = ext

    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")

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
        self._query_embedder = None  # lazy CPU query embedder

    @property
    def model_name(self) -> str:
        if self.cfg is None or not hasattr(self.cfg, 'text_embedder') or not hasattr(self.cfg.text_embedder, 'model_name'):
            raise ValueError("Text embedding model not specified in config.")
        return self.cfg.text_embedder.model_name 
    
    @property
    def cache_prefix(self) -> str:
        return 'text'
    
    @property
    def query_embedder(self):
        """Lazy CPU query embedder — used by CommonFilters for search-time only.

        Loads the same embedding model as the subprocess, but
        directly in the main process on CPU, so a search never touches the
        GPU and cannot be starved by background rating tasks.
        """
        if self._query_embedder is None:
            from src.query_embedder import QueryEmbedder
            self._query_embedder = QueryEmbedder.get_instance(
                model_name=self.model_name,
                models_folder=self.cfg.main.embedding_models_path,
                model_type='text',
            )
        return self._query_embedder
        
    def _get_metadata(self, file_path):
        return get_text_metadata(file_path)

    def _get_db_model_class(self):
        return main_db_models.FilesLibrary
    
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
        
        dim = self.cfg.text_embedder.embedding_dimension or self.embedding_dim
        try:
            # with open(file_path, 'r', encoding='utf-8') as f:
            #     content = f.read()

            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
            with fs.open_fs(base_url) as my_fs:
                # Open as binary 'rb' (fully compatible with all remote/WebDAV providers)
                with my_fs.open(path_in_fs, 'rb') as f:
                    content_bytes = f.read()
                    # Decode manually to string, ignoring any invalid bytes safely
                    content = content_bytes.decode('utf-8', errors='ignore')

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

    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None, ignore_cache=False, generate_embs_if_not_in_cache=True) -> list[list[np.ndarray]]:
        """
        Override process_files to handle list-of-lists return type for text chunks.
        Bypasses BaseSearchEngine.process_files which assumes fixed-size tensor output.
        """
        all_files_chunk_embeddings = []
        dim = self.cfg.text_embedder.embedding_dimension or self.embedding_dim

        for ind, file_path in enumerate(tqdm(file_paths, desc=f"Processing {self.cache_prefix} files")):
            current_file_embeddings = None
            try:
                cache_key = self.make_embedding_cache_key(file_path)

                cached_chunk_embeddings = self._fast_cache.get(cache_key) if not ignore_cache else None
                if cached_chunk_embeddings is not None:
                    current_file_embeddings = cached_chunk_embeddings
                else:
                    if generate_embs_if_not_in_cache:
                        current_file_embeddings = self._process_single_file(file_path, media_folder=media_folder)
                        self._fast_cache.set(cache_key, current_file_embeddings)
                    else:
                        current_file_embeddings = []


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
            dim = self.cfg.text_embedder.embedding_dimension or self.embedding_dim
            return torch.zeros((dim,))
        
    def compare(self, file_embeddings, query_embedding) -> np.ndarray:
        """
        Compute per-file similarity scores between a query embedding and all
        chunk embeddings, using ONE subprocess call regardless of file/chunk count.

        Returns
        -------
        np.ndarray of shape [N_files] with per-file smooth-max similarity.
        NaN marks files whose embedding is missing or all-zero (unindexed),
        so the downstream filter can drop them exactly like image/music.

        Multi-chunk semantics are preserved: each file's score is the
        smooth-max over the similarities of *its* individual chunks to the
        query — the "best matching chunk wins" behaviour.
        """
        # 1. Normalize the query embedding once on the calling side.
        if isinstance(query_embedding, torch.Tensor):
            query_np = query_embedding.detach().to(torch.float32).cpu().numpy().ravel()
        else:
            query_np = np.asarray(query_embedding, dtype=np.float32).ravel()

        n_files = len(file_embeddings)
        # NaN by default → unindexed files are dropped by FileManager.is_valid_pair().
        scores = np.full(n_files, np.nan, dtype=np.float32)

        # 2. Collect ALL valid chunks from ALL files into one flat list,
        #    remembering which file each chunk came from. Skip all-zero
        #    chunks (failed embeddings) so they don't poison the smooth-max.
        all_chunks = []
        chunk_file_indices = []
        for file_idx, file_chunks in enumerate(file_embeddings):
            if not file_chunks:
                continue  # stays NaN — unindexed (empty file / cache miss with no generation)
            for chunk in file_chunks:
                chunk_np = np.asarray(chunk, dtype=np.float32)
                if not np.any(np.abs(chunk_np) > 1e-5):
                    # All-zero chunk → embedding failed for this chunk.
                    # We skip it instead of letting it pollute the smooth-max.
                    continue
                all_chunks.append(chunk_np)
                chunk_file_indices.append(file_idx)

        if not all_chunks:
            return scores  # all files are unindexed

        # 3. ONE subprocess call for the whole batch.
        big_array = np.stack(all_chunks)                              # [total_chunks, D]
        flat_sims = self.embedder.compare(big_array, query_np)  # list of total_chunks floats
        flat_sims = np.asarray(flat_sims, dtype=np.float32)

        # 4. Smooth-max per file. Indexing by mask is O(N_files × avg_chunks),
        #    not O(N_files × N_total_chunks) like a Python per-file loop.
        beta = 16.0
        chunk_file_indices = np.asarray(chunk_file_indices, dtype=np.int64)
        for file_idx in range(n_files):
            chunk_sims = flat_sims[chunk_file_indices == file_idx]
            if chunk_sims.size == 0:
                continue  # all of this file's chunks were zero → stays NaN

            m = float(chunk_sims.max())
            x = beta * (chunk_sims - m)
            x = np.clip(x, -50.0, None)
            lse_centered = np.log(np.exp(x).sum())
            smooth = m + (lse_centered - np.math.log(len(chunk_sims))) / beta
            scores[file_idx] = float(smooth)

        return scores

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
            'media_directory': test_text_dir
        },
        'text_embedder': {
            'model_name': "Qwen/Qwen3-Embedding-0.6B",
            'chunk_size': 128,
            'chunk_overlap': 0,
            'embedding_dimension': 512
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