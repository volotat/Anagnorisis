"""
videos/train.py — Universal evaluator training contribution.

NOTE: Training of the universal evaluator now reads from the durable
memory/ folder (see modules/train/universal_train.py::_gather_from_memory),
so this function is no longer called by the training pipeline. It is kept
for backwards-compatibility / direct invocation only.

User ratings now live in the shared FilesLibrary table (keyed by file_path,
stored as a full VFS URL), consistent with modules/music and modules/images.
"""

import os
import numpy as np


def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (chunk_embeddings, user_rating) pairs from user-rated videos.

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
    import src.db_models as main_db_models
    from modules.videos.engine import VideoSearch
    from src.metadata_search import MetadataSearch
    import fs
    import src.virtual_file_system as vfs

    media_dir = getattr(cfg.videos, 'media_directory', None)
    if not media_dir:
        print("[videos/train] media_directory not configured, skipping.")
        return

    try:
        entries = main_db_models.FilesLibrary.query.filter(
            main_db_models.FilesLibrary.user_rating.isnot(None)
        ).all()
    except Exception as exc:
        print(f"[videos/train] DB query failed: {exc}")
        return

    total = len(entries)
    if total == 0:
        print("[videos/train] No user-rated videos found.")
        return

    print(f"[videos/train] {total} user-rated videos found.")
    if status_callback:
        status_callback(f"videos: found {total} user-rated videos.")

    engine = VideoSearch(cfg=cfg)
    engine.initiate(
        models_folder=cfg.main.embedding_models_path,
        cache_folder=cfg.main.cache_path,
    )
    meta_search = MetadataSearch(engine=engine)

    for i, entry in enumerate(entries):
        if not entry.file_path:
            continue
        # entry.file_path is a full VFS URL
        try:
            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(entry.file_path)
            with fs.open_fs(base_url) as probe_fs:
                if not probe_fs.exists(path_in_fs):
                    continue
        except Exception:
            continue

        try:
            description = meta_search.generate_full_description(
                entry.file_path, media_folder=media_dir, generate_desc_if_not_in_cache=False
            )
            if not description or len(description.strip()) < 10:
                continue
            chunk_embeddings = text_embedder.embed_text(description)
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                continue
            yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))
        except Exception as exc:
            print(f"[videos/train] Error processing {entry.file_path}: {exc}")
            continue

        status_callback(f"videos: embedded {i + 1}/{total} videos...")
