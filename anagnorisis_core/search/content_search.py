"""content_search.py — semantic search over what a file actually *is*.

One engine for every kind of content. It used to be four near-identical classes
(`MusicSearch`, `ImageSearch`, `VideoSearch`, `TextSearch`) plus an abstract base,
each wrapping its own embedding model in its own vector space. A single
multimodal embedder makes all of that one code path: the media type decides
which files belong to this engine, and the embedder handles the rest.

This is deliberately kept separate from metadata search (`src/metadata/`). Both
now use the same model, but they answer different questions — this one embeds
the *file*, metadata search embeds the *description of the file* — and they keep
their own caches and their own search mode.

Embeddings are per-chunk lists so that one ``compare`` serves everything: a
photo or a song is a list of one vector, a long text file is a list of several,
and scoring is a smooth maximum over whichever chunk matches best.
"""

import os
import traceback
from typing import List, Optional

import numpy as np

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.storage.caching import get_two_level_cache
from anagnorisis_core.media import extractors
from anagnorisis_core.media.media_types import MediaType, get_registry
from anagnorisis_core.models.embedder import cosine_similarity, get_omni_embedder, get_query_embedder

# Bump to invalidate every content embedding.
# The cache contract lives in the core, because the embedding proxy reads this
# cache too and the key formula must have exactly one definition.
from anagnorisis_core.search.content_cache import (  # noqa: F401  (re-exported)
    CACHE_PREFIX, CONTENT_ALGORITHM_VERSION, embedding_cache_key,
)

# Hashing a whole film to tell it apart from other films is not worth the I/O,
# so video is fingerprinted by sampling. Everything else is hashed in full.
_DEFAULT_HASH_ALGORITHM = "md5:v2"
_HASH_ALGORITHMS = {'videos': "xxh3s:s5m1:v1"}  # sampled xxh3_128, 5 × 1 MiB
_SAMPLE_BLOCK = 1024 * 1024
_SAMPLE_COUNT = 5


def _stat(file_path: str) -> tuple[int, int]:
    """(size, mtime_ns) for a local path or a VFS URL."""
    import fs
    base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
    with fs.open_fs(base_url) as my_fs:
        info = my_fs.getinfo(path_in_fs, namespaces=['details'])
        modified = info.get('details', 'modified')
        return info.size, int(modified * 1e9) if modified is not None else 0


def _sampled_xxh3(file_path: str, size: int) -> str:
    """Fingerprint from evenly spaced blocks plus the file size.

    Seeks through the VFS handle rather than materialising the file: the whole
    point is to identify an 8 GB film by reading 5 MiB of it, which downloading
    it first would defeat.
    """
    import fs
    import xxhash

    h = xxhash.xxh3_128()
    base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
    with fs.open_fs(base_url) as my_fs:
        with my_fs.open(path_in_fs, 'rb') as f:
            if size <= _SAMPLE_BLOCK * _SAMPLE_COUNT:
                for chunk in iter(lambda: f.read(16 * 1024 * 1024), b''):
                    h.update(chunk)
            else:
                step = (size - _SAMPLE_BLOCK) // (_SAMPLE_COUNT - 1)
                for i in range(_SAMPLE_COUNT):
                    f.seek(min(i * step, size - _SAMPLE_BLOCK), os.SEEK_SET)
                    chunk = f.read(_SAMPLE_BLOCK)
                    if not chunk:
                        break
                    h.update(chunk)
                # Mix in the size so similarly-sampled files still differ.
                h.update(size.to_bytes(8, byteorder='little', signed=False))
    return h.hexdigest()


def _as_query_vector(query_embedding) -> Optional[np.ndarray]:
    """Coerce whatever a caller passed into one 1-D query vector, or None.

    Callers hand in a bare vector, a list of chunk vectors (find-similar passes
    a whole file's chunks), or None when the query could not be embedded at all.
    A chunked target is reduced to its first chunk — the file as a whole — so the
    query is always a single vector.
    """
    if query_embedding is None:
        return None
    if isinstance(query_embedding, (list, tuple)):
        if not query_embedding:
            return None
        query_embedding = query_embedding[0]
    query = np.asarray(query_embedding, dtype=np.float32).ravel()
    if query.size == 0 or not np.all(np.isfinite(query)) or not np.any(np.abs(query) > 1e-5):
        return None
    return query


class ContentSearch:
    """Embeds and compares file *content* for one media type.

    One instance per media type (they share the model and the cache); construct
    through :func:`get_content_search` so that stays true.
    """

    def __init__(self, cfg, media_type: MediaType):
        self.cfg = cfg
        self.media_type = media_type
        # Two views of one model: the GPU worker embeds files in background
        # tasks, the CPU tower embeds queries on the search path.
        self.embedder = get_omni_embedder(cfg)
        self.query_embedder = get_query_embedder(cfg)
        self._fast_cache = get_two_level_cache(
            cache_dir=os.path.join(cfg.main.cache_path, CACHE_PREFIX),
            name=CACHE_PREFIX,
        )

    # ---- identity -----------------------------------------------------

    @property
    def name(self) -> str:
        return self.media_type.name

    @property
    def cache_prefix(self) -> str:
        return CACHE_PREFIX

    @property
    def model_hash(self) -> Optional[str]:
        """Hash of the loaded model, or None while nothing has been loaded.

        Cache keys depend on this, so callers use it as "is the model known?"
        rather than "is the model in VRAM?" — it survives an idle unload.
        """
        return self.embedder.model_hash

    @property
    def embedding_dim(self) -> Optional[int]:
        return self.embedder.embedding_dim

    def initiate(self, models_folder: str, cache_folder: str = None, **kwargs):
        """Load the shared model (downloading it on first run)."""
        self.embedder.initiate(models_folder)

    def make_embedding_cache_key(self, file_path: str) -> str:
        return embedding_cache_key(file_path, self.model_hash, CONTENT_ALGORITHM_VERSION)

    # ---- file facts ---------------------------------------------------

    def get_metadata(self, file_path: str) -> dict:
        """Internal metadata (ID3 / EXIF / stat), cached on path + mtime."""
        try:
            size, mtime_ns = _stat(file_path)
        except Exception:
            return extractors.read(self.media_type.metadata_extractor, file_path)

        cache_key = f"METADATA_OF_FILE::v3::{file_path}::{mtime_ns}"
        cached = self._fast_cache.get(cache_key)
        if cached is not None:
            return cached

        metadata = extractors.read(self.media_type.metadata_extractor, file_path)
        metadata['file_path'] = file_path
        self._fast_cache.set(cache_key, metadata)
        return metadata

    def get_hash_algorithm(self) -> str:
        """Identifier of this type's content-hash scheme, stored beside hashes."""
        return _HASH_ALGORITHMS.get(self.media_type.name, _DEFAULT_HASH_ALGORITHM)

    def get_file_hash(self, file_path: str) -> str:
        """Content fingerprint, cached on (path, size, mtime).

        Video uses a sampled hash: reading a few megabytes from head, middle and
        tail plus the file size distinguishes films without streaming gigabytes
        through the CPU. Everything else is hashed in full.
        """
        size, mtime_ns = _stat(file_path)
        algorithm = self.get_hash_algorithm()
        cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{algorithm}"
        cached = self._fast_cache.get(cache_key)
        if cached is not None:
            return cached

        if algorithm.startswith('xxh3s'):
            file_hash = _sampled_xxh3(file_path, size)
        else:
            import fs
            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
            with fs.open_fs(base_url) as my_fs:
                file_hash = vfs.calculate_file_hash(my_fs, path_in_fs)

        self._fast_cache.set(cache_key, file_hash)
        return file_hash

    # ---- embedding ----------------------------------------------------

    def _embed_one(self, file_path: str) -> List[np.ndarray]:
        """Embed a single file into a list of chunk vectors.

        Text is read and embedded as text — which keeps whole documents in one
        contextual vector and only splits what exceeds the context window.
        Everything else is handed to the model as a file.
        """
        if self.media_type.metadata_extractor == 'stat' or self.media_type.name == 'text':
            content = self._read_text(file_path)
            if not content.strip():
                return []
            chunks = self.embedder.embed_long_text(content)
            return [np.asarray(c, dtype=np.float32).ravel() for c in chunks]

        return [self.embedder.embed_file(file_path)]

    def _read_text(self, file_path: str, cap_bytes: int = 1_000_000) -> str:
        """Read a text file through the VFS, capped so a huge log cannot stall."""
        import fs
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
        with fs.open_fs(base_url) as my_fs:
            with my_fs.open(path_in_fs, 'rb') as fh:
                return fh.read(cap_bytes).decode('utf-8', errors='ignore')

    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None,
                      generate_embs_if_not_in_cache: bool = True, **kwargs) -> list[list[np.ndarray]]:
        """Embeddings for each file, as a list of chunk vectors per file.

        A file with no cached embedding returns an empty list when
        *generate_embs_if_not_in_cache* is False, which ``compare`` scores as
        NaN so the caller can filter it out as "not indexed yet".

        *callback* is called as ``callback(done, total)``.
        """
        total = len(file_paths)
        if total == 0:
            return []

        results: list[list[np.ndarray]] = []
        for index, file_path in enumerate(file_paths):
            if callback:
                callback(index, total)

            cache_key = None
            try:
                cache_key = self.make_embedding_cache_key(file_path)
                cached = self._fast_cache.get(cache_key)
                if cached is not None:
                    results.append(cached)
                    continue

                if not generate_embs_if_not_in_cache:
                    results.append([])
                    continue

                # Embedding needs the bytes; never pull a remote file in the
                # background just to index it.
                if not vfs.is_local_url(file_path):
                    results.append([])
                    continue

                chunks = self._embed_one(file_path)
                self._fast_cache.set(cache_key, chunks)
                results.append(chunks)
            except Exception as exc:
                print(f"[ContentSearch:{self.name}] Failed to embed {file_path}: {exc}")
                traceback.print_exc()
                # Remember the failure in RAM only. Without this a corrupt video
                # costs its full decode on every pass forever; with it, the retry
                # happens after a restart, which is when something might have
                # changed. MetadataSearch already does exactly this.
                if cache_key is not None:
                    self._fast_cache.set(cache_key, [], save_to_disk=False)
                results.append([])

        if callback:
            callback(total, total)
        return results

    def process_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a search query on the CPU. None if the model is unavailable."""
        return self.query_embedder.embed_query(text)

    def embed_query_file(self, file_path: str) -> Optional[np.ndarray]:
        """Embed a file the user is searching *with* — cached first, else on the CPU.

        A target already in the index costs nothing. One that is not — a file
        the user just dropped in, or one the background pass has not reached —
        is embedded on demand on the CPU. It is a single file and the user asked
        for it, so it does not belong in the background queue; and it must not
        go to the GPU, which belongs to the background passes.
        """
        cached = self._fast_cache.get(self.make_embedding_cache_key(file_path))
        if cached:
            return np.asarray(cached[0], dtype=np.float32).ravel()
        return self.query_embedder.embed_query_file(file_path)

    def compare(self, file_embeddings, query_embedding) -> np.ndarray:
        """Score each file against the query. NaN where a file is not indexed.

        Files hold one or more chunks; a file's score is a smooth maximum over
        its chunks, so a long document matches on its most relevant passage
        rather than being diluted by the rest of it.
        """
        n_files = len(file_embeddings)
        scores = np.full(n_files, np.nan, dtype=np.float32)

        query = _as_query_vector(query_embedding)
        if query is None:
            # No usable query (the CPU tower is unavailable, or the input is
            # something it cannot encode). Everything scores NaN and is filtered
            # out, which is the same as "no results" — never a crash.
            return scores

        flat_chunks: list[np.ndarray] = []
        owners: list[int] = []
        for file_index, chunks in enumerate(file_embeddings):
            if chunks is None or len(chunks) == 0:
                continue
            for chunk in chunks:
                chunk = np.asarray(chunk, dtype=np.float32).ravel()
                if chunk.size == 0 or not np.any(np.abs(chunk) > 1e-5):
                    continue  # failed embedding — leave the file unscored
                flat_chunks.append(chunk)
                owners.append(file_index)

        if not flat_chunks:
            return scores

        sims = np.asarray(
            cosine_similarity(np.stack(flat_chunks), query), dtype=np.float32
        )
        owners = np.asarray(owners, dtype=np.int64)

        smooth_max_by_owner(sims, owners, scores)
        return scores


def smooth_max_by_owner(sims: np.ndarray, owners: np.ndarray,
                        scores: np.ndarray, beta: float = 16.0) -> None:
    """Write each owner's smooth maximum over its chunks into *scores*.

    Owners with no surviving chunk are left untouched, which is how an
    unindexed file keeps its NaN and gets dropped rather than ranked last.

    Grouped in one pass. Both callers used to loop over files evaluating
    ``owners == file_index``, and each of those scans the whole array, so
    scoring was O(files × chunks): quadratic in the library, and on a few
    hundred files it already cost more than the similarity matmul it followed.
    Measured on synthetic data, with identical output: 54x faster at 439 files
    and 125x at 5,000, the gap widening as the library grows.
    """
    if sims.size == 0:
        return
    n = scores.shape[0]
    owners = np.asarray(owners, dtype=np.int64)

    group_max = np.full(n, -np.inf, dtype=np.float32)
    np.maximum.at(group_max, owners, sims)

    centred = np.clip(beta * (sims - group_max[owners]), -50.0, None)
    sum_exp = np.bincount(owners, weights=np.exp(centred), minlength=n)
    counts = np.bincount(owners, minlength=n)

    present = counts > 0
    scores[present] = (
        group_max[present]
        + (np.log(sum_exp[present]) - np.log(counts[present])) / beta
    ).astype(np.float32)


# ---- per-media-type instances -----------------------------------------

_ENGINES: dict[str, ContentSearch] = {}


def get_content_search(cfg, media_type) -> ContentSearch:
    """Return the engine for a media type, given its name or a MediaType."""
    if isinstance(media_type, str):
        media_type = get_registry(cfg).get(media_type)
    if media_type is None:
        raise ValueError("get_content_search needs a known media type")
    if media_type.name not in _ENGINES:
        _ENGINES[media_type.name] = ContentSearch(cfg, media_type)
    return _ENGINES[media_type.name]
