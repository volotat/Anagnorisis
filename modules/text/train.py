"""
text/train.py — Universal evaluator training contribution.

Exposes get_training_pairs() so that universal_train.py can collect
(chunk_embeddings, user_rating) pairs from all user-rated text files.
"""

import os
import numpy as np
from omegaconf import OmegaConf


def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (chunk_embeddings, user_rating) pairs from user-rated text files.

    Two embedding strategies are supported, selected via
    ``cfg.evaluator.text_embedding_method``:

    ``"full_text"`` (default)
        Read each file directly and embed its full content with
        text_embedder. Gives the most direct signal about what the user
        found valuable in the text itself.

    ``"metadata"``
        Generate a short description via MetadataSearch (filename + cached
        OmniDescriptor summary + internal metadata) and embed that instead.
        Useful when you want the evaluator to reason about summaries rather
        than raw content, or when files are too large for direct embedding.

    Parameters
    ----------
    cfg : OmegaConf DictConfig
    text_embedder : TextEmbedder
        Shared, already-initiated text embedder.
    status_callback : callable(str) or None

    Yields
    ------
    (np.ndarray of shape [chunks, dim], float)
    """
    import modules.text.db_models as db_models

    media_dir = getattr(cfg.text, 'media_directory', None)
    if not media_dir or not os.path.isdir(media_dir):
        print("[text/train] media_directory not configured or missing, skipping.")
        return

    try:
        entries = db_models.TextLibrary.query.filter(
            db_models.TextLibrary.user_rating.isnot(None)
        ).all()
    except Exception as exc:
        print(f"[text/train] DB query failed: {exc}")
        return

    total = len(entries)
    if total == 0:
        print("[text/train] No user-rated text files found.")
        return

    print(f"[text/train] {total} user-rated text files found.")
    if status_callback:
        status_callback(f"text: found {total} user-rated files.")

    embedding_method = OmegaConf.select(cfg, "evaluator.text_embedding_method", default="full_text")
    print(f"[text/train] Using embedding strategy: '{embedding_method}'")

    # -------------------------------------------------------------------------
    # "metadata" strategy: generate a description via MetadataSearch and embed it.
    # Good for summary-based training or when raw file content is impractical.
    # -------------------------------------------------------------------------
    if embedding_method == "metadata":
        from modules.text.engine import TextSearch
        from src.metadata_search import MetadataSearch

        engine = TextSearch(cfg=cfg)
        engine.initiate(
            models_folder=cfg.main.embedding_models_path,
            cache_folder=cfg.main.cache_path,
        )
        meta_search = MetadataSearch(engine=engine)

        for i, entry in enumerate(entries):
            if entry.file_path is None:
                continue
            fp = os.path.join(media_dir, entry.file_path)
            if not os.path.isfile(fp):
                continue
            try:
                description = meta_search.generate_full_description(
                    fp, media_folder=media_dir, generate_desc_if_not_in_cache=False
                )
                if not description or len(description.strip()) < 10:
                    continue
                chunk_embeddings = text_embedder.embed_text(description)
                if chunk_embeddings is None or len(chunk_embeddings) == 0:
                    continue
                yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))
            except Exception as exc:
                print(f"[text/train] Error processing {entry.file_path}: {exc}")
                continue

            status_callback(f"text: embedded {i + 1}/{total} files...")
        return

    # -------------------------------------------------------------------------
    # "full_text" strategy (default): read file content directly and embed it.
    # Most representative of the actual content the user rated.
    # -------------------------------------------------------------------------
    for i, entry in enumerate(entries):
        if entry.file_path is None:
            continue
        fp = os.path.join(media_dir, entry.file_path)
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                content = f.read()
            if not content or len(content.strip()) < 10:
                continue
            chunk_embeddings = text_embedder.embed_text(content)
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                continue
            yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))
        except Exception as exc:
            print(f"[text/train] Error processing {entry.file_path}: {exc}")
            continue

        status_callback(f"text: embedded {i + 1}/{total} files...")
