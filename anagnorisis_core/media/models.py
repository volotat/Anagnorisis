"""models.py — the one seam between a media type and an embedding model.

Media types are content kinds (see ``media_types.py``); nothing in the taxonomy
or in ``media_types/*.yaml`` names a model. The mapping from a content kind to
the model that handles it lives here and nowhere else.

There is now a single multimodal embedder behind every type, so the only
per-type question left is whether a type *has* content to embed at all — a PDF
has no content embedding yet, so it gets no proxy section. When that changes,
add the type to CONTENT_EMBEDDABLE and its tag vocabulary starts being used.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np
import torch

from anagnorisis_core.storage.caching import get_two_level_cache
from anagnorisis_core.search.content_cache import (CACHE_PREFIX, CONTENT_ALGORITHM_VERSION,
                                            embedding_cache_key)
from anagnorisis_core.media.media_types import MediaType
from anagnorisis_core.search.proxy import EmbeddingProxyGenerator
from anagnorisis_core.models.embedder import get_omni_embedder, get_query_embedder


# Media types whose files can be embedded from their content. Types absent here
# are described by filename, path and metadata alone, and get no proxy section.
CONTENT_EMBEDDABLE: frozenset[str] = frozenset({'audio', 'images', 'videos', 'text'})

# Which OmniDescriptor method produces a natural-language description of each
# type. Types absent here get no automatic description.
_DESCRIBE_METHODS: dict[str, str] = {
    'audio': 'describe_audio_sampled',
    'images': 'describe_image',
    'videos': 'describe_video_sampled',
    'text': 'describe_text',
}


# ---- lookups ----------------------------------------------------------

def has_content_embedding(type_name: Optional[str]) -> bool:
    """True if files of this type can be embedded from their content."""
    return bool(type_name) and type_name in CONTENT_EMBEDDABLE


def describe_method_for(type_name: Optional[str]) -> Optional[str]:
    """OmniDescriptor method name for *type_name*, or None if undescribable."""
    return _DESCRIBE_METHODS.get(type_name) if type_name else None


# ---- proxy construction -----------------------------------------------

class _ProxySource:
    """Gives the embedding proxy a model without giving it a search engine.

    Reads embeddings straight from the cache ContentSearch writes to, so a proxy
    section costs no model load and no file read. The model is loaded only to
    embed the tag vocabulary, which happens once per vocabulary.
    """

    def __init__(self, cfg, media_type: MediaType):
        self._cfg = cfg
        self.name = media_type.name
        # Tag vocabularies are text and are embedded on the CPU, so building a
        # proxy section never pulls the GPU away from a background task.
        # Only its mirrored model_hash is read here — never a command, so this
        # cannot wake the worker. The hash must be the GPU worker's, because it
        # is what ContentSearch keyed the cached embeddings with.
        self._embedder = get_omni_embedder(cfg)
        self._query_embedder = get_query_embedder(cfg)
        self._cache = get_two_level_cache(
            cache_dir=f"{cfg.main.cache_path}/{CACHE_PREFIX}",
            name=CACHE_PREFIX,
        )

    @property
    def model_hash(self) -> Optional[str]:
        """Hash of the loaded model, or None while no model has been loaded."""
        return self._embedder.model_hash

    def embedding_key(self, file_path: str) -> Optional[str]:
        """Cache key of *file_path*'s content embedding, or None if unknowable."""
        model_hash = self.model_hash
        if not model_hash:
            return None
        return embedding_cache_key(file_path, model_hash, CONTENT_ALGORITHM_VERSION)

    def cached_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """The file's content embedding if already computed, else None.

        Never loads a model and never reads the file. Content embeddings are
        stored as a list of chunk vectors; the proxy describes the file as a
        whole, so the first chunk is the one that represents it.
        """
        key = self.embedding_key(file_path)
        if key is None:
            return None
        value = self._cache.get(key)
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            if not value:
                return None
            value = value[0]
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().ravel().astype(np.float32)
        return np.asarray(value, dtype=np.float32).ravel()

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a tag string on the CPU."""
        vector = self._query_embedder.embed_document(text)
        if vector is None:
            raise RuntimeError("Query embedder unavailable; cannot embed tag vocabulary.")
        return np.asarray(vector, dtype=np.float32).ravel()


_PROXIES: dict[str, Optional[EmbeddingProxyGenerator]] = {}
_PROXIES_LOCK = threading.Lock()


def get_proxy(cfg, media_type: Optional[MediaType]) -> Optional[EmbeddingProxyGenerator]:
    """Process-wide proxy generator for *media_type*, or None if it has no model."""
    if media_type is None:
        return None
    if media_type.name not in _PROXIES:
        with _PROXIES_LOCK:
            if media_type.name not in _PROXIES:
                _PROXIES[media_type.name] = _build_proxy(cfg, media_type)
    return _PROXIES[media_type.name]


def _build_proxy(cfg, media_type: MediaType) -> Optional[EmbeddingProxyGenerator]:
    if not has_content_embedding(media_type.name):
        return None
    return EmbeddingProxyGenerator(
        source=_ProxySource(cfg, media_type),
        tags=media_type.tags,
        threshold=media_type.tags_threshold,
        cache_path=cfg.main.cache_path,
    )
