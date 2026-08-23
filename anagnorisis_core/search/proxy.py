"""proxy.py — a text stand-in for a file's content embedding.

Turns the vector a content embedder produced into text, so that files can be
searched, scored and rated by the text pipeline long before the (much slower)
OmniDescriptor pass reaches them. Two components are combined:

  1. Zero-shot semantic tags — cosine similarity between the file embedding and
     the media type's tag vocabulary, filtered by a configurable threshold (not
     top-N, to avoid false positive labels).

  2. Quantized fingerprint — each embedding dimension mapped to one of 32
     characters, preserving fine-grained neighbourhood structure in the token
     space. Similar embeddings produce nearly identical character sequences,
     which the text embedder in turn encodes to similar vectors.

Building a section requires no model and no file access: the embedding is read
from the cache the module engines already fill, via the ``source`` handed in at
construction. A model is loaded only to embed the tag vocabulary, once per
vocabulary. That is what lets ``generate_full_description`` include a proxy
section for any file, at any time, without disturbing whatever is in VRAM.

Callers get a proxy from ``anagnorisis_core.media.models.get_proxy(cfg, media_type)``
rather than constructing one.
"""

import glob
import hashlib
import inspect
import os
import re
from typing import List, Optional, Protocol, Sequence

import numpy as np
import torch

from anagnorisis_core.storage.caching import get_two_level_cache

# ── Character alphabet for quantized fingerprint ─────────────────────────────
_LEVELS = 32
_CHARS  = 'abcdefghijklmnopqrstuvwxyz012345'  # 26 + 6 = 32 chars


def quantize_embedding(emb: np.ndarray, levels: int = _LEVELS) -> str:
    """Rank-based (histogram-equalisation) quantisation of *emb*.

    After L2-normalisation each component of a high-dimensional embedding is
    very small (std ≈ 1/√d), so a linear mapping from [-1,1] produces a nearly
    constant sequence (all 'p'/'q' for 32 levels).  Rank-based quantisation
    avoids this by assigning each dimension to a percentile bucket, guaranteeing
    that every character in the 32-symbol alphabet appears roughly equally often.

    Similarity is still preserved: if two embeddings are close in cosine space
    their component-wise orderings are similar, so they produce similar strings.

    Returns an empty string for zero/near-zero embeddings (failed processing),
    so that files that could not be embedded are never assigned a fingerprint
    or tags — a zero vector would produce the same fingerprint for every failed
    file, making the proxy misleadingly identical across unrelated files.

    CLAP: 512 dims → 512 tokens (~2 text chunks)
    SigLIP: 768 dims → 768 tokens (~3 text chunks)
    """
    emb = np.array(emb, dtype=np.float32).ravel()
    # Guard: skip quantisation for zero/near-zero embeddings (failed files).
    if np.linalg.norm(emb) < 1e-6:
        return ''
    n = len(emb)
    # Double argsort gives each element its rank (0 = smallest value).
    ranks = np.argsort(np.argsort(emb))
    # Map rank [0, n-1] → bucket [0, levels-1] uniformly.
    indices = np.clip((ranks * levels // n).astype(np.int32), 0, levels - 1)
    return ' '.join(_CHARS[i] for i in indices)


# ── Algorithm version hash ────────────────────────────────────────────────────
# Derived from the source of quantize_embedding so the cache key changes
# automatically whenever the fingerprint algorithm is modified.  No manual
# version bumping is needed — stale entries are simply bypassed on the next
# cache miss and recomputed with the new algorithm.
_ALGO_HASH = hashlib.md5(inspect.getsource(quantize_embedding).encode()).hexdigest()[:8]


class ProxySource(Protocol):
    """The embedding model, as much of it as the proxy needs.

    Implemented by ``anagnorisis_core.media.models._ProxySource``. Kept this narrow on
    purpose: the proxy used to take a whole search engine and reach into its
    private cache, which tied it to the modules and made the lookup easy to
    break from either side.
    """

    name: str

    @property
    def model_hash(self) -> Optional[str]:
        """Hash of the loaded model, or None while nothing has been loaded."""

    def embedding_key(self, file_path: str) -> Optional[str]:
        """Cache key of the file's content embedding, or None if unknowable."""

    def cached_embedding(self, file_path: str) -> Optional[np.ndarray]:
        """The file's embedding if already computed, else None. Loads nothing."""

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a tag string. May load the model."""


class EmbeddingProxyGenerator:
    """Builds and caches proxy sections for one media type."""

    def __init__(
        self,
        source: ProxySource,
        tags: Sequence[str],
        threshold: Optional[float],
        cache_path: str,
    ):
        self.source = source
        self.tags: List[str] = [str(tag) for tag in (tags or [])]
        self.threshold = float(threshold) if threshold is not None else None

        # Stable hash of the tag vocabulary for cache-key versioning
        joined = '\n'.join(sorted(self.tags))
        self._vocab_hash = hashlib.md5(joined.encode()).hexdigest()[:12]

        # Used in the tag-matrix filename, so it must not introduce separators of
        # its own — the name is parsed back apart when pruning.
        self._slug = re.sub(r'[^A-Za-z0-9]+', '-', str(getattr(source, 'name', ''))
                            ) or 'unnamed'

        # Per-file proxy text cache (RAM + disk, long TTL)
        self._cache = get_two_level_cache(
            cache_dir=os.path.join(cache_path, 'embedding_proxy_cache'),
            name="embedding_proxy",
        )

        # Tag-embedding matrix saved as a .pt file, once per vocabulary *and*
        # model — see _tag_embs_path_for.
        os.makedirs(cache_path, exist_ok=True)
        self._cache_path = cache_path
        self._tag_embs: Optional[np.ndarray] = None  # lazy-loaded
        self._tag_embs_hash: Optional[str] = None    # model it was built with

    # ── public API ────────────────────────────────────────────────────────────

    def get_proxy_text(self, file_path: str) -> str:
        """Return the proxy section for *file_path*, or '' if unavailable.

        Lookup order, none of which loads a model or reads the file:
          1. Proxy text cache (RAM → disk).
          2. The content-embedding cache, from which the section is rebuilt.
          3. '' — nothing has embedded this file yet.
        """
        cache_key = self._cache_key(file_path)
        if cache_key is None:
            return ''

        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        embedding = self.source.cached_embedding(file_path)
        if embedding is None:
            return ''

        text = self._build_section(embedding)
        # Don't cache error messages — embedding may succeed on a later pass.
        # Don't cache if tags should have been produced but weren't — tag
        # embeddings may not have been available (model unloaded, .pt missing).
        has_error = '[Error:' in text
        tags_expected = bool(self.tags)
        tags_present = 'Tags:' in text
        if not has_error and (not tags_expected or tags_present):
            self._cache.set(cache_key, text)
        return text

    # ── internals ─────────────────────────────────────────────────────────────

    def _cache_key(self, file_path: str) -> Optional[str]:
        """Key the proxy text by the embedding it is derived from.

        Everything that can change the section is in here: the fingerprint
        algorithm, the vocabulary, the threshold, and the embedding's own
        identity (path + model + version). No stat call and no content hash —
        the section is exactly as fresh as the embedding behind it.
        """
        embedding_key = self.source.embedding_key(file_path)
        if embedding_key is None:
            return None
        thresh = 'none' if self.threshold is None else f'{self.threshold:.4f}'
        return f"proxy::{_ALGO_HASH}::{self._vocab_hash}::{thresh}::{embedding_key}"

    # tag_embeddings_<media type>_<vocabulary>_<model>.pt
    _TAG_FILE_GLOB = 'tag_embeddings_*.pt'

    def _tag_embs_path_for(self, model_hash: str) -> str:
        """Where the tag matrix for *model_hash* lives.

        Tag vectors are compared against file embeddings, so they only mean
        anything in the same vector space — the model is part of their identity,
        not just the vocabulary. Keyed on the vocabulary alone, swapping the
        embedding model would silently reuse the old model's tag vectors and
        every tag would be noise.

        The media type leads the name so the file says who owns it. That is what
        makes the old ones removable: edit a vocabulary and the hash changes, and
        without an owner in the name there is no way to tell which of the
        leftovers were the previous version of *this* vocabulary and which belong
        to another media type entirely.
        """
        return os.path.join(
            self._cache_path,
            f'tag_embeddings_{self._slug}_{self._vocab_hash}_{model_hash[:12]}.pt',
        )

    def _prune_stale_tag_files(self, keep: str) -> None:
        """Delete this vocabulary's earlier tag matrices.

        Each is several megabytes, and a new one is written whenever the
        vocabulary is edited or the embedding model changes — so without this
        they accumulate one per change, forever, and none of them will ever be
        read again.

        Only files this proxy owns are touched: same media type, different
        vocabulary or model. Another media type's current file is not ours to
        delete. Files in the older name that carried no media type are removed
        too — nothing can read them any more, so keeping them is pure cost.

        Deleting a file another process is reading is safe: the reader holds the
        inode open until it is done.
        """
        prefix = f'tag_embeddings_{self._slug}_'
        removed = freed = 0
        for path in glob.glob(os.path.join(self._cache_path, self._TAG_FILE_GLOB)):
            if os.path.abspath(path) == os.path.abspath(keep):
                continue
            parts = os.path.basename(path)[:-len('.pt')].split('_')
            # 'tag', 'embeddings', <type>, <vocab>, <model>  — four means the old
            # name, which has no media type in it.
            legacy = len(parts) == 4
            if not (legacy or os.path.basename(path).startswith(prefix)):
                continue
            try:
                size = os.path.getsize(path)
                os.remove(path)
                removed += 1
                freed += size
            except OSError as exc:
                print(f"[EmbeddingProxy] Could not remove {path}: {exc}")
        if removed:
            print(f"[EmbeddingProxy] Removed {removed} superseded tag file(s), "
                  f"{freed / 1e6:.1f} MB.")

    def ensure_tag_embeddings(self, embed_batch=None) -> bool:
        """Embed the tag vocabulary now, if it is not already cached.

        Worth calling deliberately before a long run: the vocabulary is a couple
        of thousand short strings, it is needed the moment a description is
        assembled, and left to happen lazily it fires in the middle of the
        pipeline with a large model already resident.

        *embed_batch* is an optional callable taking a list of strings and
        returning one vector per string. Pass one in background work and the whole
        vocabulary goes through the model in batches on the GPU — seconds instead
        of many minutes. Without it, tags are embedded one at a time on the CPU,
        which is the right choice on the search path (searching must never wake
        the GPU) and the wrong one anywhere else.
        """
        if embed_batch is not None:
            return self._compute_tag_embeddings_batched(embed_batch)
        return self._get_tag_embeddings().shape[0] > 0

    def _compute_tag_embeddings_batched(self, embed_batch) -> bool:
        """Embed the whole vocabulary in batches and cache it."""
        model_hash = self.source.model_hash
        if not model_hash or not self.tags:
            return False
        path = self._tag_embs_path_for(model_hash)
        if os.path.exists(path) or self._tag_embs_hash == model_hash:
            return True     # already done, on disk or in memory

        print(f"[EmbeddingProxy] Embedding {len(self.tags)} '{self.source.name}' "
              f"tags in batches…")
        try:
            vectors = embed_batch(list(self.tags))
        except Exception as exc:
            print(f"[EmbeddingProxy] Batched tag embedding failed ({exc}); "
                  f"leaving '{self.source.name}' tags for a later pass.")
            return False

        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != len(self.tags):
            print(f"[EmbeddingProxy] Unexpected tag embedding shape {arr.shape} "
                  f"for {len(self.tags)} tags; discarding.")
            return False

        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.where(norms > 0, norms, 1.0)
        torch.save(torch.from_numpy(arr), path)
        self._tag_embs = arr
        self._tag_embs_hash = model_hash
        print(f"[EmbeddingProxy] Saved tag embeddings to {path}")
        self._prune_stale_tag_files(keep=path)
        return True

    def _get_tag_embeddings(self) -> np.ndarray:
        """Return [N_tags, D] L2-normalised float32 array.

        Loaded from disk on first call; computed via the source's text encoder
        and saved to disk if the cache file is missing. This is the only path
        that loads an embedding model.

        Row *i* always corresponds to ``self.tags[i]`` — callers map similarity
        scores back to tag names by index.
        """
        model_hash = self.source.model_hash
        if not model_hash:
            # Nothing has loaded a model yet, so there is no space to embed
            # into. Don't cache this; a later pass will succeed.
            return np.zeros((0, 1), dtype=np.float32)

        if self._tag_embs is not None and self._tag_embs_hash == model_hash:
            return self._tag_embs

        tag_embs_path = self._tag_embs_path_for(model_hash)
        if os.path.exists(tag_embs_path):
            try:
                data = torch.load(tag_embs_path, map_location='cpu', weights_only=True)
                arr  = (
                    data.numpy().astype(np.float32)
                    if isinstance(data, torch.Tensor)
                    else np.array(data, dtype=np.float32)
                )
                self._tag_embs = arr
                self._tag_embs_hash = model_hash
                print(
                    f"[EmbeddingProxy] Loaded {arr.shape[0]} tag embeddings "
                    f"for '{self.source.name}' from {tag_embs_path}"
                )
                return self._tag_embs
            except Exception as e:
                print(f"[EmbeddingProxy] Cache load failed ({e}), recomputing tag embeddings.")

        if not self.tags:
            self._tag_embs = np.zeros((0, 1), dtype=np.float32)
            self._tag_embs_hash = model_hash
            return self._tag_embs

        print(
            f"[EmbeddingProxy] Computing embeddings for {len(self.tags)} "
            f"'{self.source.name}' tags…"
        )
        # Probe with one tag before committing to thousands. If the embedder
        # cannot produce a vector — no model downloaded yet, or a load failure —
        # then grinding through the whole vocabulary produces nothing but a long
        # wait and a wall of identical errors.
        try:
            probe = self.source.embed_text(self.tags[0])
        except Exception as exc:
            probe = None
            print(f"[EmbeddingProxy] Could not encode tags: {exc}")
        if probe is None:
            print(f"[EmbeddingProxy] No embedder available; leaving "
                  f"'{self.source.name}' tags for a later pass.")
            return np.zeros((0, 1), dtype=np.float32)

        rows: List[Optional[np.ndarray]] = [probe]
        for i, tag in enumerate(self.tags[1:], start=1):
            if i % 50 == 0:
                print(f"[EmbeddingProxy]   {i}/{len(self.tags)} tags processed")
            try:
                rows.append(self.source.embed_text(tag))
            except Exception as exc:
                print(f"[EmbeddingProxy] Error encoding tag '{tag}': {exc}")
                rows.append(None)

        valid = [r for r in rows if r is not None]
        if not valid:
            self._tag_embs = np.zeros((0, 1), dtype=np.float32)
            self._tag_embs_hash = model_hash
            return self._tag_embs

        # Substitute a zero row for a tag that failed to encode rather than
        # dropping it: callers read tag names back as self.tags[row], so a
        # missing row would shift the name of every tag after it. A zero row
        # scores 0 against everything, so it simply never wins.
        dim = valid[0].shape[0]
        arr = np.stack(
            [r if r is not None else np.zeros(dim, dtype=np.float32) for r in rows],
            axis=0,
        )  # [N_tags, D]
        # L2-normalise; the model does not guarantee unit-length outputs.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr   = arr / np.where(norms > 0, norms, 1.0)

        torch.save(torch.from_numpy(arr), tag_embs_path)
        self._tag_embs = arr
        self._tag_embs_hash = model_hash
        print(f"[EmbeddingProxy] Saved tag embeddings to {tag_embs_path}")
        self._prune_stale_tag_files(keep=tag_embs_path)
        return self._tag_embs

    def _build_section(self, emb: np.ndarray) -> str:
        """Build the proxy text section from a raw 1-D embedding vector.

        Returns an error section for zero/near-zero embeddings (failed files)
        so the problem is visible rather than silently showing nothing.
        """
        # ── Component 2: quantised fingerprint ───────────────────────────────
        fingerprint = quantize_embedding(emb)
        # quantize_embedding returns '' for zero/near-zero vectors, which means
        # the file could not be processed by the embedding model.
        if not fingerprint:
            return (
                f'# Embedding proxy ({self.source.name}):\n'
                f'[Error: embedding unavailable — '
                f'file could not be processed by the embedding model]\n'
            )

        # ── Component 1: zero-shot semantic tags ─────────────────────────────
        tags_line = ''
        tag_embs  = self._get_tag_embeddings()
        if tag_embs.shape[0] > 0:
            norm  = float(np.linalg.norm(emb))
            emb_n = emb / norm if norm > 0 else emb
            sims  = emb_n @ tag_embs.T          # cosine similarity [N_tags]
            if self.threshold is None:
                # No threshold: always show top 15 tags by cosine similarity.
                # Use >= 0 so zero-embeddings (failed files) still get tags
                # based on default orientation rather than silently showing nothing.
                top_idx   = np.argsort(sims)[::-1][:15]
                matching  = [self.tags[i] for i in top_idx if sims[i] >= 0]
                tags_line = 'Tags: ' + ', '.join(matching) if matching else ''
            else:
                above = np.where(sims >= self.threshold)[0]
                if len(above) > 0:
                    sorted_idx = above[np.argsort(sims[above])[::-1]]
                    matching   = [self.tags[i] for i in sorted_idx]
                    tags_line  = 'Tags: ' + ', '.join(matching)
                else:
                    # Diagnostic: log top-5 matches so threshold can be tuned
                    top5_idx  = np.argsort(sims)[::-1][:5]
                    top5      = [(self.tags[i], float(sims[i])) for i in top5_idx]
                    top5_str  = ', '.join(f'{t}={s:.3f}' for t, s in top5)
                    print(
                        f'[EmbeddingProxy] No tags above threshold {self.threshold:.4f} '
                        f'for {self.source.name}. Top-5: {top5_str}'
                    )

        lines = [f'# Embedding proxy ({self.source.name}):']
        if tags_line:
            lines.append(tags_line)
        lines.append(f'Fingerprint: {fingerprint}')
        return '\n'.join(lines) + '\n'
