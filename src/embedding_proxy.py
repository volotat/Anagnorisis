"""embedding_proxy.py

Generates a textual proxy section from CLAP/SigLIP embeddings for files that
do not yet have an OmniDescriptor-generated description.  The proxy allows the
universal evaluator (text-based Jina + TransformerEvaluator) to score audio and
image files immediately, without waiting for the slow LLM description pipeline.

Two components are combined into a single text section:

  1. Zero-shot semantic tags — cosine similarity between the file embedding and a
     vocabulary of tag strings defined in config.yaml, filtered by a configurable
     threshold (not top-N, to avoid false positive labels).

  2. Quantized fingerprint — each embedding dimension mapped to one of 32
     characters, preserving fine-grained neighbourhood structure in the token
     space.  Similar embeddings produce nearly identical character sequences,
     which Jina encodes to similar vectors.

Tag vocabulary embeddings are precomputed once per vocab-hash and persisted as a
.pt file in cache_path.  Per-file proxy texts are stored in a TwoLevelCache so
that subsequent calls are instant and model-free.

Typical flow in serve.py
────────────────────────
  # At module init:
  proxy_gen = EmbeddingProxyGenerator(
      engine=music_search_engine,
      tag_list=list(cfg.music.embedding_tags),
      threshold=cfg.music.embedding_tags_threshold,
      cache_path=cfg.main.cache_path,
      model_name=cfg.music.embedding_model,
  )
  metadata_search_engine.embedding_proxy = proxy_gen

  # After computing CLAP/SigLIP embeddings in bulk (pre-populate cache before
  # the Jina embedding phase so generate_full_description hits only the cache):
  for i, fp in enumerate(filtered_files_list):
      proxy_gen.compute_proxy_section(files_hash_map[fp], embeddings[i].cpu().numpy())

  # generate_full_description() then reads from cache transparently via
  # get_cached_proxy_text(), never loading any model.
"""

import os
import hashlib
import numpy as np
import torch
from typing import Optional, List

import inspect
from src.caching import TwoLevelCache

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

    CLAP: 512 dims → 512 tokens (~2 Jina chunks)
    SigLIP: 768 dims → 768 tokens (~3 Jina chunks)
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


class EmbeddingProxyGenerator:
    """Generates and caches textual proxy sections from CLAP/SigLIP embeddings.

    Parameters
    ----------
    engine : MusicSearch | ImageSearch
        Already-initiated search engine.  Used for ``get_file_hash`` and
        ``process_text`` (to compute tag vocabulary embeddings once).
    tag_list : list[str]
        Vocabulary of tag strings (from ``cfg.music.embedding_tags`` etc.).
    threshold : float
        Cosine-similarity threshold in [0, 1].  Only tags whose similarity to
        the file embedding meets this threshold are included.
    cache_path : str
        Root cache directory (``cfg.main.cache_path``).
    model_name : str
        Display name for the header line in the proxy section.
    """

    def __init__(
        self,
        engine,
        tag_list: List[str],
        threshold: float,
        cache_path: str,
        model_name: str = "",
    ):
        self.engine     = engine
        # Flatten one level of nesting — config.yaml uses "- [a, b, c]" rows for
        # compact formatting, which YAML/OmegaConf parses as a list-of-lists.
        # OmegaConf's ListConfig is iterable but is not list/tuple, so we check
        # for any non-string iterable instead.
        raw = list(tag_list) if tag_list else []
        self.tag_list: List[str] = []
        for item in raw:
            if isinstance(item, str):
                self.tag_list.append(item)
            else:
                try:
                    for t in item:
                        self.tag_list.append(str(t))
                except TypeError:
                    self.tag_list.append(str(item))
        self.threshold  = float(threshold) if threshold is not None else None
        self.model_name = model_name

        # Stable hash of the tag vocabulary for cache-key versioning
        joined = '\n'.join(sorted(self.tag_list))
        self._vocab_hash = hashlib.md5(joined.encode()).hexdigest()[:12]

        # Per-file proxy text cache (RAM + disk, long TTL)
        proxy_cache_dir = os.path.join(cache_path, 'embedding_proxy_cache')
        self._cache = TwoLevelCache(
            cache_dir=proxy_cache_dir,
            name="embedding_proxy",
        )

        # Tag-embedding matrix saved as a .pt file once per vocab hash
        os.makedirs(cache_path, exist_ok=True)
        self._tag_embs_path = os.path.join(
            cache_path, f'tag_embeddings_{self._vocab_hash}.pt'
        )
        self._tag_embs: Optional[np.ndarray] = None  # lazy-loaded

    # ── tag embeddings ────────────────────────────────────────────────────────

    def _get_tag_embeddings(self) -> np.ndarray:
        """Return [N_tags, D] L2-normalised float32 array.

        Loaded from disk on first call; computed via the engine's text encoder
        and saved to disk if the cache file is missing.
        """
        if self._tag_embs is not None:
            return self._tag_embs

        if os.path.exists(self._tag_embs_path):
            try:
                data = torch.load(self._tag_embs_path, map_location='cpu', weights_only=True)
                arr  = (
                    data.numpy().astype(np.float32)
                    if isinstance(data, torch.Tensor)
                    else np.array(data, dtype=np.float32)
                )
                self._tag_embs = arr
                print(
                    f"[EmbeddingProxy] Loaded {arr.shape[0]} tag embeddings "
                    f"for '{self.model_name}' from {self._tag_embs_path}"
                )
                return self._tag_embs
            except Exception as e:
                print(f"[EmbeddingProxy] Cache load failed ({e}), recomputing tag embeddings.")

        if not self.tag_list:
            self._tag_embs = np.zeros((0, 1), dtype=np.float32)
            return self._tag_embs

        print(
            f"[EmbeddingProxy] Computing embeddings for {len(self.tag_list)} tags "
            f"(model: {self.model_name})…"
        )
        rows: List[Optional[np.ndarray]] = []
        for i, tag in enumerate(self.tag_list):
            if i % 50 == 0:
                print(f"[EmbeddingProxy]   {i}/{len(self.tag_list)} tags processed")
            try:
                emb = self.engine.process_text(tag)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                rows.append(np.array(emb, dtype=np.float32).ravel())
            except Exception as exc:
                print(f"[EmbeddingProxy] Error encoding tag '{tag}': {exc}")
                rows.append(None)

        valid = [r for r in rows if r is not None]
        if not valid:
            self._tag_embs = np.zeros((0, 1), dtype=np.float32)
            return self._tag_embs

        arr = np.stack(valid, axis=0)  # [N, D]
        # L2-normalise (SigLIP text embeddings are already normalised; CLAP's are not)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr   = arr / np.where(norms > 0, norms, 1.0)

        torch.save(torch.from_numpy(arr), self._tag_embs_path)
        self._tag_embs = arr
        print(f"[EmbeddingProxy] Saved tag embeddings to {self._tag_embs_path}")
        return self._tag_embs

    # ── public API ────────────────────────────────────────────────────────────

    def get_cached_proxy_text(self, file_path: str) -> str:
        """Return the proxy section for *file_path*, computing it on-the-fly if needed.

        Lookup order:
          1. Proxy cache (RAM → disk) — instant, no model.
          2. Engine's embedding cache — reads the already-computed CLAP/SigLIP
             embedding and builds the proxy section without loading any model.
          3. Returns '' if neither cache has data for this file.

        This means the proxy section appears immediately in "show full search
        description" for any file whose embedding was previously cached, even if
        the background rating pass has not yet run since a cache-key change.
        """
        try:
            file_hash = self.engine.get_file_hash(file_path)
        except Exception:
            return ''
        thresh_str = 'none' if self.threshold is None else f'{self.threshold:.4f}'
        cache_key = f"proxy::{_ALGO_HASH}::{self._vocab_hash}::{thresh_str}::{file_hash}"

        # 1. Proxy cache hit
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        # 2. Fall back to engine's embedding cache (no model load)
        try:
            model_hash = getattr(self.engine, 'model_hash', None)
            postfix    = self.engine._get_model_hash_postfix() if hasattr(self.engine, '_get_model_hash_postfix') else ''
            if model_hash:
                emb_key = f"{file_hash}::{model_hash}{postfix}"
                emb_cached = self.engine._fast_cache.get(emb_key)
                if emb_cached is not None:
                    # emb_cached is a CPU tensor, shape [1, D] or [D]
                    if isinstance(emb_cached, torch.Tensor):
                        emb_np = emb_cached.detach().cpu().numpy().ravel().astype(np.float32)
                    else:
                        emb_np = np.array(emb_cached, dtype=np.float32).ravel()
                    return self.compute_proxy_section(file_hash, emb_np)
        except Exception as exc:
            print(f"[EmbeddingProxy] Embedding cache fallback failed for {file_path}: {exc}")

        return ''

    def compute_proxy_section(self, file_hash: str, file_emb: np.ndarray) -> str:
        """Compute (and cache) the proxy section given a pre-computed embedding.

        Call this after computing CLAP/SigLIP embeddings in bulk, before the
        Jina embedding phase, so that ``get_cached_proxy_text`` always hits the
        cache without needing any model.

        Parameters
        ----------
        file_hash : str
            Content hash of the file (``engine.get_file_hash(path)``).
        file_emb : np.ndarray
            1-D float32 embedding vector, shape [D].

        Returns
        -------
        str
            Formatted proxy text section (always non-empty — contains at least
            the fingerprint line).
        """
        thresh_str = 'none' if self.threshold is None else f'{self.threshold:.4f}'
        cache_key = f"proxy::{_ALGO_HASH}::{self._vocab_hash}::{thresh_str}::{file_hash}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        text = self._build_section(np.array(file_emb, dtype=np.float32).ravel())
        # Don't cache error messages — embedding may succeed on a later pass.
        # Don't cache if tags should have been produced but weren't — tag
        # embeddings may not have been available (model unloaded, .pt missing).
        has_error     = '[Error:' in text
        tags_expected = bool(self.tag_list)
        tags_present  = 'Tags:' in text
        if not has_error and (not tags_expected or tags_present):
            self._cache.set(cache_key, text)
        return text

    # ── internals ─────────────────────────────────────────────────────────────

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
                f'# Embedding proxy ({self.model_name}):\n'
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
                matching  = [self.tag_list[i] for i in top_idx if sims[i] >= 0]
                tags_line = 'Tags: ' + ', '.join(matching) if matching else ''
            else:
                above = np.where(sims >= self.threshold)[0]
                if len(above) > 0:
                    sorted_idx = above[np.argsort(sims[above])[::-1]]
                    matching   = [self.tag_list[i] for i in sorted_idx]
                    tags_line  = 'Tags: ' + ', '.join(matching)
                else:
                    # Diagnostic: log top-5 matches so threshold can be tuned
                    top5_idx  = np.argsort(sims)[::-1][:5]
                    top5      = [(self.tag_list[i], float(sims[i])) for i in top5_idx]
                    top5_str  = ', '.join(f'{t}={s:.3f}' for t, s in top5)
                    print(
                        f'[EmbeddingProxy] No tags above threshold {self.threshold:.4f} '
                        f'(model: {self.model_name}). Top-5: {top5_str}'
                    )

        lines = [f'# Embedding proxy ({self.model_name}):']
        if tags_line:
            lines.append(tags_line)
        lines.append(f'Fingerprint: {fingerprint}')
        return '\n'.join(lines) + '\n'
