"""
engine.py — Search / embedding engine (REQUIRED for media modules)

This file contains the search engine class that inherits from
``BaseSearchEngine``.  The base class handles:
  • Model downloading from Hugging Face Hub
  • Two-level caching (in-memory + on-disk)
  • File hashing (default MD5; overridable for custom schemes like xxh3)
  • Batch processing orchestration with progress callbacks
  • Singleton pattern (one instance per subclass)

Your subclass must implement the abstract methods listed below.

If your module doesn't need ML-based search (e.g. a settings page), you can
skip this file — but then you also won't need ``serve.py`` to create a
search engine.
"""

import os
import threading
import torch
import numpy as np

from src.base_search_engine import BaseSearchEngine
from src.model_manager import ModelManager

import modules._module_template.db_models as db_models


def get_example_metadata(file_path: str) -> dict:
    """
    Extract metadata from a file.  Return a flat dict of strings.

    This is called by the framework for metadata search and for displaying
    file information on the frontend. Adapt it for your media type —
    for example:
      • Images → EXIF data, resolution, colour mode
      • Audio  → TinyTag (title, artist, duration, bitrate)
      • Text   → file size, creation time
      • Video  → resolution, duration, codec

    This function is also used by MetadataSearch.generate_full_description()
    to build the text representation for universal evaluator training.
    Keep it fast — it is called for every file on every page load.
    """
    metadata = {}
    try:
        stat = os.stat(file_path)
        metadata['file_size'] = str(stat.st_size)
    except Exception as e:
        metadata['error'] = str(e)
    return metadata


class ExampleSearch(BaseSearchEngine):
    """
    Minimal search-engine skeleton.

    To make this work you need to:
      1. Set ``model_name`` to a valid Hugging Face model ID (or ``None``
         if you handle model loading yourself).
      2. Set ``cache_prefix`` to a unique string for your module.
      3. Implement ``_load_model_and_processor`` to load your model.
      4. Implement ``_process_single_file`` to produce an embedding tensor.
      5. Implement ``_get_metadata`` to return file metadata.
      6. Implement ``_get_db_model_class`` to return your SQLAlchemy model.
      7. Implement ``_get_model_hash_postfix`` for embedding versioning.

    Optional overrides:
      • ``_get_media_folder()`` — return the media directory path (has a
        default empty-string implementation in the base class; override if
        you need path validation in the engine layer).
      • ``get_file_hash(file_path)`` — override the default MD5 hash with
        a custom algorithm (e.g. videos use xxh3 sampled hashing for speed).
      • ``compare(embeds_target, embeds_query)`` — override for custom
        similarity logic (e.g. text module uses smooth-max over chunks).
    """

    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.cfg = cfg
        self._embedder_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Abstract property implementations
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        """
        Return the Hugging Face model name used for embeddings.
        Read it from your config section so users can swap models easily.
        Return ``None`` if the module doesn't use an embedding model.
        """
        if self.cfg is None:
            return None
        return self.cfg.get("example", {}).get("embedding_model", None)

    @property
    def cache_prefix(self) -> str:
        """Unique prefix for cache files — must match the module name."""
        return 'example'

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _load_model_and_processor(self, local_model_path: str):
        """
        Load the ML model and processor/tokenizer from a local directory.

        Example (for an image model):
            from transformers import AutoProcessor, AutoModel
            self.processor = AutoProcessor.from_pretrained(local_model_path)
            model = AutoModel.from_pretrained(local_model_path)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self._model_manager = ModelManager(model, self.device)
        """
        pass  # TODO: implement for your media type

    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Read one file and return its embedding as a ``torch.Tensor``
        of shape ``(1, embedding_dim)``.

        This is called by the base class during batch processing.
        The result is cached using ``(file_hash, model_hash)`` as key.
        Must be deterministic — same file always produces the same embedding.
        """
        raise NotImplementedError("Implement _process_single_file for your media type")

    def _get_metadata(self, file_path: str) -> dict:
        """Return metadata dict for the given file."""
        return get_example_metadata(file_path)

    def _get_db_model_class(self):
        """Return the SQLAlchemy model class for this module."""
        return db_models.ExampleLibrary

    def _get_model_hash_postfix(self) -> str:
        """
        Return a short postfix appended to the model hash for cache keying.
        Bump this when you change how embeddings are computed (different
        preprocessing, pooling, etc.) to invalidate stale cached embeddings.
        """
        return "_v1.0.0"

    def _get_media_folder(self) -> str:
        """
        Return the configured media directory path.
        Used by the base class for file path validation.
        Override this if you need the engine to know the media folder
        (not required if only serve.py uses it).
        """
        if self.cfg is None:
            return ""
        return self.cfg.get("example", {}).get("media_directory", "") or ""
