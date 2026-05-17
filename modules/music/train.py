from tqdm import tqdm  # noqa: F401
import numpy as np
import modules.music.db_models as db_models
# import src.scoring_models        # Removed: MusicEvaluator training is no longer used
# from sklearn.model_selection import train_test_split  # Removed
from modules.music.engine import MusicSearch  # MusicEvaluator removed — only MusicSearch needed
import os
# import pickle  # Removed
# import torch   # Removed

# from src.socket_events import CommonSocketEvents  # Removed (train_music_evaluator removed)


# ───────────────────────────────────────────────────────────────────────────────
# train_music_evaluator() has been removed.
#
# Previously this function trained a dedicated MusicEvaluator (3-layer MLP) on
# raw CLAP audio embeddings of user-rated tracks and saved music_evaluator.pt.
# Scoring is now handled entirely by UniversalEvaluator (TransformerEvaluator
# on Jina text embeddings of file descriptions + CLAP embedding proxy).
# The “Train music evaluator” button and its socket handler have also been
# removed from the Train page UI.
#
# music_evaluator.pt on disk is no longer used and can be safely deleted.
# ───────────────────────────────────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Universal evaluator training contribution
# ---------------------------------------------------------------------------

def get_training_pairs(cfg, text_embedder, status_callback=None):
    """
    Yield (chunk_embeddings, user_rating) pairs from user-rated music tracks.

    Uses the "metadata" strategy: generates a text description for each track
    via MetadataSearch (filename + cached OmniDescriptor summary + internal
    metadata tags) and embeds it with the shared text_embedder.

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
    import modules.music.db_models as db_models
    from modules.music.engine import MusicSearch
    from src.metadata_search import MetadataSearch

    media_dir = getattr(cfg.music, 'media_directory', None)
    if not media_dir or not os.path.isdir(media_dir):
        print("[music/train] media_directory not configured or missing, skipping.")
        return

    try:
        entries = db_models.MusicLibrary.query.filter(
            db_models.MusicLibrary.user_rating.isnot(None)
        ).all()
    except Exception as exc:
        print(f"[music/train] DB query failed: {exc}")
        return

    total = len(entries)
    if total == 0:
        print("[music/train] No user-rated tracks found.")
        return

    print(f"[music/train] {total} user-rated tracks found.")
    if status_callback:
        status_callback(f"music: found {total} user-rated tracks.")

    engine = MusicSearch(cfg=cfg)
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
            # least a fingerprint + tag section for files whose CLAP embedding is cached,
            # so a short/empty description typically means the file has no embedding yet
            # and no cached OmniDescriptor output.  Skip those to avoid noise.
            if not description or len(description.strip()) < 10:
                continue
            chunk_embeddings = text_embedder.embed_text(description)
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                continue
            yield (np.array(chunk_embeddings, dtype=np.float32), float(entry.user_rating))
        except Exception as exc:
            print(f"[music/train] Error processing {entry.file_path}: {exc}")
            continue

        status_callback(f"music: embedded {i + 1}/{total} tracks...")