"""search.py — describe any file in text, then search those descriptions.

MetadataSearch is one process-wide instance that knows nothing about modules.
Given a path it looks up the file's media type, and from that alone it knows how
to read its internal metadata, which tag vocabulary applies, and how to describe
it — so it can index every file on every configured server, whether or not a
module happens to own that kind of content.

Everything expensive is cached and the search path never loads a model it does
not already need: descriptions come from the description cache, proxy sections
from the content-embedding cache, and only the text embedder runs to turn an
assembled description into a vector.

Get the instance with ``get_metadata_search(cfg)``.
"""

import hashlib
import os
import threading
import time
import traceback
from typing import Optional

import numpy as np

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.media import description as description_mod
from anagnorisis_core.storage.caching import get_two_level_cache
from anagnorisis_core.media import models
from anagnorisis_core.media.media_types import MediaType, get_registry
from anagnorisis_core.models.descriptor import OmniDescriptor
from anagnorisis_core.search.content_search import _as_query_vector, smooth_max_by_owner
from anagnorisis_core.models.embedder import (cosine_similarity, get_omni_embedder,
                                              get_query_embedder, is_tensor)


class MetadataSearch:
    """Builds, embeds and compares the text description of a file."""

    def __init__(self, cfg):
        self.cfg = cfg
        # Memoised answer for "what hash did a previous session leave behind",
        # which is constant until the descriptor is actually loaded. '' means
        # "asked, and there is none"; None means "not asked yet".
        self._cold_omni_hash: Optional[str] = None
        self.types = get_registry(cfg)
        # The GPU worker builds the index; the CPU tower answers queries.
        self.embedder = get_omni_embedder(cfg)
        self.query_embedder = get_query_embedder(cfg)
        self.omni_descriptor = OmniDescriptor(cfg)

        self._fast_cache = get_two_level_cache(
            cache_dir=os.path.join(cfg.main.cache_path, 'metadata_cache'),
            name="metadata_search",
        )

    def get_algorithm_version(self) -> str:
        """Identifier of the description/embedding scheme, for cache invalidation.

        v2.0: descriptions are assembled from the media-type registry (real image
        EXIF instead of a PIL error, media-type-named proxy header), and the
        embedding key now tracks the description and proxy behind it.
        v2.1: every chunk of the description is stored, not just the first — the
        cached value is now a list of vectors rather than one, so the key must
        change for the shape as well as for the coverage.
        v3.0: descriptions are embedded by the shared multimodal embedder, which
        puts them in the same vector space as file content.
        """
        return "meta-search-v3.0"

    # ------------------------------------------------------------------
    # Media type helpers
    # ------------------------------------------------------------------

    def media_type_for(self, file_path: str) -> Optional[MediaType]:
        """The media type owning this file, or None if the extension is unknown."""
        return self.types.for_file(file_path)

    def _proxy_for(self, media_type: Optional[MediaType]):
        return models.get_proxy(self.cfg, media_type)

    # ------------------------------------------------------------------
    # Automatic descriptions (OmniDescriptor)
    # ------------------------------------------------------------------

    def _get_omni_model_hash(self) -> Optional[str]:
        """Return the OmniDescriptor model hash from memory, or from the shared cache
        if the model hasn't been loaded yet this session (e.g. after an app restart).
        Returns None if the hash has never been persisted anywhere.
        """
        mh = self.omni_descriptor.model_hash
        if mh:
            return mh
        # Falling through here happens once per file on every search and index
        # pass, for a value that cannot change while the process runs and the
        # descriptor stays unloaded. Remember it rather than paying a cache
        # lookup per file to be told the same thing again.
        if self._cold_omni_hash is not None:
            return self._cold_omni_hash or None
        model_name = getattr(getattr(self.cfg, 'omni', None), 'model_name', None)
        if not model_name:
            return None
        cached = self._fast_cache.get(f"omni_model_hash::{model_name}")
        self._cold_omni_hash = cached or ''
        return cached

    def load_descriptor(self) -> Optional[str]:
        """Load OmniDescriptor if needed and return its model hash.

        Must be called before ``get_undescribed_files`` on a cold cache: that
        probe needs the model hash to build cache keys, and the hash only exists
        once the model has been loaded at least once. Without this the two would
        deadlock — nothing could be described because nothing had been loaded,
        and nothing would load because there was nothing to describe.
        """
        descriptor = self.omni_descriptor
        if not getattr(descriptor, 'model_hash', None):
            descriptor.initiate(self.cfg.main.embedding_models_path)
            # Persist for cold-start lookups by _get_omni_model_hash() after a restart.
            self._fast_cache.set(
                f"omni_model_hash::{self.cfg.omni.model_name}", descriptor.model_hash
            )
            # The live hash now wins in _get_omni_model_hash, but drop the
            # memoised cold-start answer so it can never shadow a newer one.
            self._cold_omni_hash = None
        return descriptor.model_hash

    def make_description_cache_key(self, file_path: str) -> str:
        media_type = self.media_type_for(file_path)
        method_name = models.describe_method_for(media_type.name if media_type else None)
        return (
            f"auto_desc::{file_path}::{self._get_omni_model_hash()}::{method_name}"
        )

    def get_undescribed_files(self, file_paths: list[str]) -> list[str] | None:
        """Return paths that have no cached auto-description.

        Skips files whose media type cannot be described at all. Cache lookups
        are stat-free, so this is cheap enough to run over a whole library.
        Returns None if the OmniDescriptor model hash is not yet known — call
        ``load_descriptor()`` first.
        """
        if self._get_omni_model_hash() is None:
            return None

        undescribed = []
        for fp in file_paths:
            media_type = self.media_type_for(fp)
            if models.describe_method_for(media_type.name if media_type else None) is None:
                continue
            if self._fast_cache.get(self.make_description_cache_key(fp)) is None:
                undescribed.append(fp)
        return undescribed

    def _get_auto_description(self, file_path: str, generate_desc_if_not_in_cache: bool = True) -> str:
        """OmniDescriptor's natural-language description of the file.

        Cached by (path, omni model hash, method) so recomputation only happens
        when the descriptor model changes. Returns '' for media types that have
        no description method, and None on a cache miss when generation is off.
        """
        media_type = self.media_type_for(file_path)
        method_name = models.describe_method_for(media_type.name if media_type else None)
        if method_name is None:
            return ''

        cache_key = self.make_description_cache_key(file_path)
        cached = self._fast_cache.get(cache_key)
        if cached is not None:
            return cached

        if not generate_desc_if_not_in_cache:
            return None

        # Short text is its own description, so decide that *before* loading a
        # descriptor. api._describe_one already works this way; this path did
        # not, so a folder of notes paid a full model run each, and
        # build_description then ignored the result and used the words anyway.
        text_content = None
        if method_name == 'describe_text':
            text_content = description_mod.read_text_content(file_path).strip()
            if len(text_content) <= description_mod.text_verbatim_limit(self.cfg):
                return text_content

        # Loading the descriptor changes the model hash, and with it the cache
        # key — so recompute the key afterwards rather than caching under a key
        # built from a 'None' hash.
        self.load_descriptor()
        cache_key = self.make_description_cache_key(file_path)

        try:
            method = getattr(self.omni_descriptor, method_name)
            if method_name == 'describe_text':
                description = method(text_content)
            else:
                description = method(file_path)
        except Exception as e:
            print(f"[MetadataSearch] Auto-description failed for {file_path}: {e}")
            description = "[Error] Failed to generate auto-description."
            # Cache the failure in RAM only, so a retry happens after a restart.
            self._fast_cache.set(cache_key, description, save_to_disk=False)
            return description

        self._fast_cache.set(cache_key, description)
        print(f"Generated auto-description for {file_path} (method: {method_name}): "
              f"{description[:100]}{'...' if len(description) > 100 else ''}")
        return description

    # ------------------------------------------------------------------
    # Full description
    # ------------------------------------------------------------------

    def generate_full_description(self, file_path: str, generate_desc_if_not_in_cache: bool = True) -> str:
        """Build the full metadata description for a file.

        Local files: filename + path + auto-desc + embedding proxy + internal
        metadata + {filename}.meta.
        Remote files: filename + path + {filename}.meta only — never touch the
        original file.

        MemorySystem (user rating path) also uses this; for remote it produces a
        thin description that the universal evaluator can score, but auto-desc /
        internal metadata / proxy are intentionally skipped to honour the
        "no automatic downloads of remote files" rule.

        Args:
            file_path: Full VFS URL of the file.
            generate_desc_if_not_in_cache: If False, skips OmniDescriptor
                auto-description generation for files not already in cache.
        """
        return description_mod.build_description(
            file_path,
            cfg=self.cfg,
            media_type=self.media_type_for(file_path),
            describe=lambda fp: self._get_auto_description(
                fp, generate_desc_if_not_in_cache=generate_desc_if_not_in_cache
            ),
        )

    # ------------------------------------------------------------------
    # Metadata embeddings
    # ------------------------------------------------------------------

    def make_embedding_cache_key(self, file_path: str) -> str:
        """Cache key for a file's metadata embedding.

        Carries the identity of everything that can change the description
        without changing the path: the cached auto-description and the content
        embedding the proxy section is derived from. Without those, a file
        indexed from its filename alone would keep that thin embedding forever,
        even after it was described — the background passes would have no way to
        tell the two apart.

        Both signatures are cache lookups: no stat call, no file read, no model.
        """
        media_type = self.media_type_for(file_path)

        description = self._fast_cache.get(self.make_description_cache_key(file_path))
        desc_sig = _digest(description) if description else 'none'

        proxy = self._proxy_for(media_type)
        proxy_sig = (proxy.source.embedding_key(file_path) if proxy else None) or 'none'

        return (
            f"meta::{file_path}::"
            f"desc::{desc_sig}::"
            f"proxy::{proxy_sig}::"
            f"alg::{self.get_algorithm_version()}"
        )

    def get_unembedded_files(self, file_paths: list[str]) -> list[str]:
        """Return paths with no cached metadata embedding, unknown types skipped."""
        return [
            fp for fp in file_paths
            if self.media_type_for(fp) is not None
            and self._fast_cache.get(self.make_embedding_cache_key(fp)) is None
        ]

    def invalidate(self, file_path: str) -> None:
        """Drop the cached metadata embedding for a file.

        Called when something outside the cache-key inputs changes the
        description — in practice, the user editing a .meta sidecar.
        """
        self._fast_cache.set(self.make_embedding_cache_key(file_path), None)

    def process_query(self, query_text: str):
        """Embed a search query on the CPU, for matching against descriptions."""
        return self.query_embedder.embed_query(query_text)

    def _generate_embedding(self, file_path: str) -> list[np.ndarray]:
        """Embed a file's assembled description — one vector per chunk.

        Every chunk is kept. A description runs well past the embedder's chunk
        size, so keeping only the first would index the opening of the text and
        silently drop the rest: internal metadata, the .meta sidecar, and the
        tail of the embedding proxy. What ``compare`` then sees would no longer
        be the description the UI shows for the file.

        Cache-only for descriptions and proxies — the only model this loads is
        the text embedder.
        """
        meta_text = self.generate_full_description(
            file_path, generate_desc_if_not_in_cache=False
        )
        meta_embeddings = self.embedder.embed_long_text(meta_text)

        if meta_embeddings is None or len(meta_embeddings) == 0:
            return [self._zero_embedding()]
        # A list of 1-D vectors, not a 2-D array: compare() tests each file's
        # chunks with `if not file_chunks`, which is ambiguous for an array.
        return [np.asarray(chunk, dtype=np.float32).ravel() for chunk in meta_embeddings]

    def _zero_embedding(self) -> np.ndarray:
        dim = self.embedder.embedding_dim
        return np.zeros((dim,), dtype=np.float32) if dim else np.array([], dtype=np.float32)

    def _process_single_file_meta(self, file_path: str, generate_embs_if_not_in_cache: bool = True) -> list[np.ndarray]:
        """
        Processes a single file's metadata, utilizing the cache.
        Returns the file's chunk embeddings, which ``compare`` reduces to one
        score per file by smooth-max.
        """
        try:
            cache_key = self.make_embedding_cache_key(file_path)

            cached_chunks = self._fast_cache.get(cache_key)
            if cached_chunks is not None:
                return cached_chunks

            if not generate_embs_if_not_in_cache:
                return [self._zero_embedding()]

            chunks = self._generate_embedding(file_path)
            self._fast_cache.set(cache_key, chunks)
            return chunks

        except Exception as e:
            print(f"Error processing metadata for {file_path}: {e}")
            traceback.print_exc()
            return [self._zero_embedding()]

    def process_files(self, file_paths: list[str], callback=None,
                      generate_embs_if_not_in_cache: bool = True, **kwargs) -> list[list[np.ndarray]]:
        """
        Processes metadata for a list of files by calling the single-file processor in a loop.
        Returns a list of lists of numpy arrays (one list per file, each containing one
        metadata embedding).
        """
        total_files = len(file_paths)
        if total_files == 0:
            return []

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
            embedding_list = self._process_single_file_meta(
                file_path, generate_embs_if_not_in_cache=generate_embs_if_not_in_cache
            )
            max_elapsed = max(max_elapsed, time.time() - file_start)

            all_files_meta_embeddings.append(embedding_list)

        return all_files_meta_embeddings

    def compare(self, file_embeddings, query_embedding):
        """
        Compare query against metadata chunk embeddings of all files in a
        single subprocess call.  Returns np.ndarray of per-file smooth-max
        similarity scores with NaN for unindexed files — identical contract
        to BaseSearchEngine.compare() and TextSearch.compare().
        """
        n_files = len(file_embeddings)
        # NaN by default → unindexed files are dropped by FileManager.is_valid_pair().
        scores = np.full(n_files, np.nan, dtype=np.float32)

        # 1. Normalize the query embedding once on the calling side. It may be
        #    None if the CPU query tower could not encode the input, in which
        #    case there are simply no results rather than an exception.
        if is_tensor(query_embedding):
            query_embedding = query_embedding.detach().float().cpu().numpy()
        query_np = _as_query_vector(query_embedding)
        if query_np is None:
            return scores

        # 2. Collect ALL valid chunks from ALL files into one flat list.
        #    Skip empty file-chunk lists and all-zero chunks (failed embeddings).
        all_chunks = []
        chunk_file_indices = []
        for file_idx, file_chunks in enumerate(file_embeddings):
            if not file_chunks:
                continue  # stays NaN — unindexed
            for chunk in file_chunks:
                chunk_np = np.asarray(chunk, dtype=np.float32)
                if not np.any(np.abs(chunk_np) > 1e-5):
                    continue  # all-zero chunk — embedding failed for this chunk
                all_chunks.append(chunk_np)
                chunk_file_indices.append(file_idx)

        if not all_chunks:
            return scores  # all NaN

        # 3. ONE subprocess call for the whole batch.
        big_array = np.stack(all_chunks)
        flat_sims = cosine_similarity(big_array, query_np)
        flat_sims = np.asarray(flat_sims, dtype=np.float32)

        # 4. Smooth-max per file. Shared with the content engine, which scored
        #    the same way: one grouped pass rather than a scan per file.
        smooth_max_by_owner(flat_sims, chunk_file_indices, scores)
        return scores

    # ------------------------------------------------------------------
    # Progress helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Return a human-readable H:M string for a given number of seconds."""
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
            remaining = (avg * 0.1 + max_elapsed * 0.9) * (total - processed)
        else:
            remaining = avg * (total - processed)
        return percent, remaining


def _digest(text: str) -> str:
    return hashlib.md5(text.encode('utf-8', errors='ignore')).hexdigest()[:8]


# ---- module-level accessor --------------------------------------------

_INSTANCE: Optional[MetadataSearch] = None
_INSTANCE_LOCK = threading.Lock()


def get_metadata_search(cfg) -> MetadataSearch:
    """Return the process-wide MetadataSearch, building it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = MetadataSearch(cfg)
    return _INSTANCE
