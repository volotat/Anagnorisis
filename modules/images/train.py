from tqdm import tqdm  # noqa: F401
import numpy as np
import modules.images.db_models as db_models
# import src.scoring_models        # Removed: ImageEvaluator training is no longer used
# from sklearn.model_selection import train_test_split  # Removed
from modules.images.engine import ImageSearch  # ImageEvaluator removed — only ImageSearch needed
import os
# import pickle  # Removed
# import torch   # Removed


# ───────────────────────────────────────────────────────────────────────────────
# train_image_evaluator() has been removed.
#
# Previously this function trained a dedicated ImageEvaluator (3-layer MLP) on
# raw SigLIP image embeddings of user-rated images and saved image_evaluator.pt.
# Scoring is now handled entirely by UniversalEvaluator (TransformerEvaluator
# on Jina text embeddings of file descriptions + SigLIP embedding proxy).
# The “Train image evaluator” button and its socket handler have also been
# removed from the Train page UI.
#
# image_evaluator.pt on disk is no longer used and can be safely deleted.
# ───────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Universal evaluator training contribution
# ---------------------------------------------------------------------------

def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (chunk_embeddings, user_rating) pairs from user-rated images.

    Uses the "metadata" strategy: generates a text description for each image
    via MetadataSearch (filename + cached OmniDescriptor summary + internal
    metadata) and embeds it with the shared text_embedder.

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
    import modules.images.db_models as db_models
    from modules.images.engine import ImageSearch
    from src.metadata_search import MetadataSearch

    media_dir = getattr(cfg.images, 'media_directory', None)
    if not media_dir or not os.path.isdir(media_dir):
        print("[images/train] media_directory not configured or missing, skipping.")
        return

    try:
        entries = db_models.ImagesLibrary.query.filter(
            db_models.ImagesLibrary.user_rating.isnot(None)
        ).all()
    except Exception as exc:
        print(f"[images/train] DB query failed: {exc}")
        return

    total = len(entries)
    if total == 0:
        print("[images/train] No user-rated images found.")
        return

    print(f"[images/train] {total} user-rated images found.")
    if status_callback:
        status_callback(f"images: found {total} user-rated images.")

    engine = ImageSearch(cfg=cfg)
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
            # With the embedding proxy, generate_full_description() always includes at
            # least a fingerprint + tag section for files whose SigLIP embedding is cached,
            # so a short/empty description typically means the file has no embedding yet
            # and no cached OmniDescriptor output.  Skip those to avoid noise.
            if not description or len(description.strip()) < 10:
                continue
            chunk_embeddings = text_embedder.embed_text(description)
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                continue
            yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))
        except Exception as exc:
            print(f"[images/train] Error processing {entry.file_path}: {exc}")
            continue

        status_callback(f"images: embedded {i + 1}/{total} images...")