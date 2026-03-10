"""
engine.py — Search / embedding engine (REQUIRED for media modules)

This file contains the search engine class that inherits from
``BaseSearchEngine``.  The base class handles:
  • Model downloading from Hugging Face Hub
  • Two-level caching (in-memory + on-disk)
  • File hashing
  • Batch processing orchestration

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

import pages._module_template.db_models as db_models


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
        """
        raise NotImplementedError("Implement _process_single_file for your media type")

    def _get_metadata(self, file_path: str) -> dict:
        """Return metadata dict for the given file."""
        return get_example_metadata(file_path)

    def _get_db_model_class(self):
        """Return the SQLAlchemy model class for this module."""
        return db_models.ExampleLibrary

    def _get_media_folder(self) -> str:
        """Return the configured media directory path."""
        if self.cfg is None:
            raise ValueError("Config not available")
        return self.cfg.get("example", {}).get("media_directory", "")
