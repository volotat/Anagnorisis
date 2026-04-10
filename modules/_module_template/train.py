"""
train.py — Universal evaluator training contribution (OPTIONAL)

Create this file in your module folder if your module has user-rated items
that should contribute to universal evaluator training.

The orchestrator (modules/train/universal_train.py) auto-discovers this file
via a modules/*/train.py glob and calls get_training_pairs() if present.

CONTRACT
--------
Expose exactly one function:

    get_training_pairs(cfg, text_embedder, status_callback=None)
        → yields (chunk_embeddings: np.ndarray[chunks, dim], user_rating: float)

The orchestrator will:
  1. Scan modules/*/train.py for all installed modules
  2. Call get_training_pairs(cfg, text_embedder, status_callback) on each
  3. Collect all yielded pairs and train a single shared evaluator on them
"""

import os
import numpy as np
from omegaconf import OmegaConf


def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (chunk_embeddings, user_rating) pairs for universal evaluator training.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
        The merged application config.  Read your module's section via
        ``OmegaConf.select(cfg, "my_module.some_key", default=...)``.
    text_embedder : TextEmbedder
        Shared, already-initiated text embedder.
        Call ``text_embedder.embed_text(text) -> np.ndarray[chunks, dim]``.
    status_callback : callable(str) or None
        Optional; call it periodically to report progress.

    Yields
    ------
    chunk_embeddings : np.ndarray of shape [num_chunks, embedding_dim]
        Text embeddings for one rated item, obtained via embed_text().
        For text-native modules (like WebSearch) embed the file content
        directly.  For media modules (images, music) embed a generated
        text description instead.
    user_rating : float
        The user's explicit score for this item (typically 0–10).

    Notes
    -----
    - Only yield items where the user has explicitly rated them
      (user_rating IS NOT NULL in the DB).
    - Skip missing files silently.
    - Do NOT load heavy ML models here; use text_embedder for all embedding.
    - The function must be safe to call multiple times (idempotent).
    """
    # TODO: replace with your module's db_models import
    import modules._module_template.db_models as db_models

    # Example: read a config value for your module
    # media_dir = OmegaConf.select(cfg, "my_module.media_directory", default=None)
    # if not media_dir:
    #     print("[my_module/train] media_directory not configured, skipping.")
    #     return

    try:
        entries = db_models.ExampleLibrary.query.filter(
            db_models.ExampleLibrary.user_rating.isnot(None)
        ).all()
    except Exception as exc:
        print(f"[my_module/train] DB query failed: {exc}")
        return

    total = len(entries)
    if total == 0:
        return

    if status_callback:
        status_callback(f"my_module: found {total} user-rated items.")

    for i, entry in enumerate(entries):
        # --- Build a text representation for this item ---
        #
        # Text-native module (file IS the text):
        #   full_path = os.path.join(storage_dir, entry.file_path)
        #   with open(full_path, 'r', encoding='utf-8') as f:
        #       text = f.read()
        #
        # Media module (generate a description string):
        #   text = metadata_search.generate_full_description(
        #       full_path, media_folder=media_dir,
        #       generate_desc_if_not_in_cache=False,
        #   )
        #
        # Replace the placeholder below with your real implementation:
        text = f"Placeholder for item: {entry.file_path}"

        if not text or len(text.strip()) < 10:
            continue

        try:
            chunk_embeddings = text_embedder.embed_text(text)
        except Exception as exc:
            print(f"[my_module/train] Embedding error for item {entry.id}: {exc}")
            continue

        if chunk_embeddings is None or len(chunk_embeddings) == 0:
            continue

        yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))

        if status_callback:
            status_callback(f"my_module: embedded {i + 1}/{total} items...")
