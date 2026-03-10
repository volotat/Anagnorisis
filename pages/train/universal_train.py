"""
Universal evaluator training module.

Gathers user-rated files from ALL media modules (music, images, videos, text),
generates text descriptions via MetadataSearch.generate_full_description()
(using only cached auto-descriptions for speed), converts them to text
embeddings, and trains a single TransformerEvaluator on the combined data.
"""

from tqdm import tqdm
import numpy as np
import os
import time

from sklearn.model_selection import train_test_split
from src.scoring_models import TransformerEvaluator
from src.metadata_search import MetadataSearch
from src.text_embedder import TextEmbedder


# ---------------------------------------------------------------------------
# Module registry: (config_section, db_models_path, engine_class_path, media_dir_attr)
# Each entry describes how to find rated files for a given media module.
# ---------------------------------------------------------------------------
_MODULE_DEFS = [
    {
        "name": "music",
        "config_attr": "music",
        "db_import": "pages.music.db_models",
        "db_class": "MusicLibrary",
        "engine_import": "pages.music.engine",
        "engine_class": "MusicSearch",
        "embedding_method": "metadata",  # use metadata/summary description
    },
    {
        "name": "images",
        "config_attr": "images",
        "db_import": "pages.images.db_models",
        "db_class": "ImagesLibrary",
        "engine_import": "pages.images.engine",
        "engine_class": "ImageSearch",
        "embedding_method": "metadata",  # use metadata/summary description
    },
    {
        "name": "videos",
        "config_attr": "videos",
        "db_import": "pages.videos.db_models",
        "db_class": "VideosLibrary",
        "engine_import": "pages.videos.engine",
        "engine_class": "VideoSearch",
        "embedding_method": "metadata",  # use metadata/summary description
    },
    {
        "name": "text",
        "config_attr": "text",
        "db_import": "pages.text.db_models",
        "db_class": "TextLibrary",
        "engine_import": "pages.text.engine",
        "engine_class": "TextSearch",
        "embedding_method": "full_text",  # use full file content (controlled by cfg.evaluator.text_embedding_method)
    },
]


# Modules handled by the legacy _MODULE_DEFS path — excluded from auto-discovery
# to avoid counting their training data twice.
_LEGACY_MODULE_NAMES = {mdef["name"] for mdef in _MODULE_DEFS}


# ---------------------------------------------------------------------------
# Singleton universal evaluator
# ---------------------------------------------------------------------------
class UniversalEvaluator(TransformerEvaluator):
    """Singleton TransformerEvaluator shared across the application."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(UniversalEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=1024, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super().__init__(embedding_dim, rate_classes, name="UniversalEvaluator")
            self._initialized = True


# ---------------------------------------------------------------------------
# Auto-discovery: gather training pairs from pages/*/train.py
# ---------------------------------------------------------------------------
def _gather_from_module_train_files(cfg, text_embedder, status_callback=None):
    """
    Scan pages/*/train.py for modules that expose get_training_pairs() and
    collect already-embedded (chunk_embeddings, score) pairs from each.

    Modules listed in _LEGACY_MODULE_NAMES (music, images, videos, text) are
    skipped here because they are already handled by _gather_rated_files().
    """
    import importlib
    import glob

    # pages/ directory is one level above this file (pages/train/)
    pages_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    )

    extra_embeddings = []
    extra_scores = []

    train_files = sorted(glob.glob(os.path.join(pages_dir, '*', 'train.py')))

    for train_file in train_files:
        module_name = os.path.basename(os.path.dirname(train_file))

        # Skip template / hidden folders
        if module_name.startswith('_'):
            continue

        # Skip modules already gathered by the legacy path
        if module_name in _LEGACY_MODULE_NAMES:
            continue

        module_import_path = f"pages.{module_name}.train"
        try:
            mod = importlib.import_module(module_import_path)
        except Exception as exc:
            print(f"[UniversalTrain] Could not import {module_import_path}: {exc}")
            continue

        if not hasattr(mod, 'get_training_pairs'):
            print(f"[UniversalTrain] {module_import_path} has no get_training_pairs(), skipping.")
            continue

        print(f"[UniversalTrain] Gathering training pairs from '{module_name}'...")
        if status_callback:
            status_callback(f"Gathering training pairs from '{module_name}'...")

        count = 0
        try:
            for chunk_embeddings, score in mod.get_training_pairs(
                cfg, text_embedder, status_callback
            ):
                extra_embeddings.append(chunk_embeddings)
                extra_scores.append(score)
                count += 1
        except Exception as exc:
            print(f"[UniversalTrain] Error in get_training_pairs for '{module_name}': {exc}")
            continue

        print(f"[UniversalTrain] '{module_name}': collected {count} training pairs.")

    return extra_embeddings, extra_scores


# ---------------------------------------------------------------------------
# Helper: lazily import a module attribute
# ---------------------------------------------------------------------------
def _import_attr(module_path: str, attr_name: str):
    """Import *attr_name* from *module_path* (dot-separated)."""
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


# ---------------------------------------------------------------------------
# Gather rated files across all modules
# ---------------------------------------------------------------------------
def _gather_rated_files(cfg, status_callback=None):
    """
    Returns a list of dicts:
        [{
            "file_path": str,           # absolute path
            "user_rating": float,
            "media_folder": str,        # media directory for this module
            "engine": BaseSearchEngine, # singleton engine instance (for get_metadata / get_file_hash)
            "module_name": str,
            "embedding_method": str,    # "full_text" or "metadata"
        }, ...]
    """
    all_rated = []

    for mdef in _MODULE_DEFS:
        module_name = mdef["name"]
        config_attr = mdef["config_attr"]

        # ---- check if the module is configured ----
        if not hasattr(cfg, config_attr):
            print(f"[UniversalTrain] Module '{module_name}' not in config, skipping.")
            continue

        module_cfg = getattr(cfg, config_attr)
        media_dir = getattr(module_cfg, "media_directory", None)
        if not media_dir or not os.path.isdir(media_dir):
            print(f"[UniversalTrain] Module '{module_name}': media directory "
                  f"'{media_dir}' missing or invalid, skipping.")
            continue

        # ---- import DB model class ----
        try:
            DbModel = _import_attr(mdef["db_import"], mdef["db_class"])
        except Exception as e:
            print(f"[UniversalTrain] Could not import DB model for '{module_name}': {e}")
            continue

        # ---- query rated entries ----
        try:
            entries = DbModel.query.filter(DbModel.user_rating.isnot(None)).all()
        except Exception as e:
            print(f"[UniversalTrain] DB query failed for '{module_name}': {e}")
            continue

        if not entries:
            print(f"[UniversalTrain] No rated files in '{module_name}'.")
            continue

        # ---- get or create the engine singleton (needed by MetadataSearch for get_file_hash / get_metadata) ----
        try:
            EngineClass = _import_attr(mdef["engine_import"], mdef["engine_class"])
            engine = EngineClass(cfg=cfg)
            # Ensure the engine is initiated (safe to call multiple times due to singleton guard)
            engine.initiate(
                models_folder=cfg.main.embedding_models_path,
                cache_folder=cfg.main.cache_path,
            )
        except Exception as e:
            print(f"[UniversalTrain] Could not init engine for '{module_name}': {e}")
            continue

        # ---- build file list ----
        for entry in entries:
            fp = os.path.join(media_dir, entry.file_path)
            if not os.path.isfile(fp):
                continue
            all_rated.append({
                "file_path": fp,
                "user_rating": entry.user_rating,
                "media_folder": media_dir,
                "engine": engine,
                "module_name": module_name,
                "embedding_method": mdef["embedding_method"],
            })

        if status_callback:
            status_callback(f"Found {len(entries)} rated files in '{module_name}' module.")

        print(f"[UniversalTrain] '{module_name}': {len(entries)} rated entries "
              f"({sum(1 for e in entries if os.path.isfile(os.path.join(media_dir, e.file_path)))} files exist).")

    return all_rated


# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------
def train_universal_evaluator(cfg, callback=None):
    """
    Train a single TransformerEvaluator on text-embedding representations of
    all user-rated files across every configured media module.

    Parameters
    ----------
    cfg : OmegaConf config
    callback : optional ``(status, percent, baseline_accuracy, train_accuracy, test_accuracy)``
    """

    print("=" * 60)
    print("[UniversalTrain] Starting universal evaluator training")
    print("=" * 60)

    # 1. Gather rated files from all modules
    rated_files = _gather_rated_files(cfg, status_callback=lambda msg: callback(msg, 0, 0) if callback else None)

    if not rated_files:
        msg = "No rated files found across any module. Abort training."
        print(f"[UniversalTrain] {msg}")
        if callback:
            callback(msg, 100, 0)
        return

    print(f"[UniversalTrain] Total rated files across all modules: {len(rated_files)}")

    # 2. Build embeddings for all rated files.
    #
    #    Two strategies are used depending on the module's embedding_method:
    #
    #    "full_text" (text module, when cfg.evaluator.text_embedding_method == "full_text"):
    #        Use engine.process_files() — reads the full file content, chunks it,
    #        and embeds with TextEmbedder. Results are cached by (file_hash, model_hash)
    #        so re-running training is fast.
    #
    #    "metadata" (all other modules, or text when text_embedding_method == "metadata"):
    #        Use MetadataSearch.generate_full_description() — filename + path +
    #        cached OmniDescriptor summary + internal metadata → single short text.
    #        generate_desc_if_not_in_cache=False keeps this fast (no inference).

    text_embedder = TextEmbedder(cfg)
    text_embedder.initiate(models_folder=cfg.main.embedding_models_path)

    # -------------------------------------------------------------------------
    # Phase A: pre-compute full-text embeddings for the text module (if enabled).
    # Batching via engine.process_files() reuses its per-file cache.
    # -------------------------------------------------------------------------
    precomputed_embeddings: dict[str, list] = {}  # file_path -> list[np.ndarray]

    if cfg.evaluator.text_embedding_method == "full_text":
        text_items = [item for item in rated_files if item["embedding_method"] == "full_text"]
        if text_items:
            text_engine = text_items[0]["engine"]  # TextSearch singleton
            text_file_paths = [item["file_path"] for item in text_items]
            text_media_folder = text_items[0]["media_folder"]

            print(f"[UniversalTrain] Pre-computing full-text embeddings for "
                  f"{len(text_file_paths)} text files via TextSearch cache...")
            if callback:
                callback(f"Pre-computing full-text embeddings for {len(text_file_paths)} text files...", 0, 0)

            text_chunk_embeddings = text_engine.process_files(
                text_file_paths, media_folder=text_media_folder
            )
            precomputed_embeddings = {
                fp: embs for fp, embs in zip(text_file_paths, text_chunk_embeddings)
            }
            print(f"[UniversalTrain] Full-text embeddings ready.")

    # -------------------------------------------------------------------------
    # Phase B: iterate all rated files, picking the right embedding per item.
    # -------------------------------------------------------------------------

    # Cache MetadataSearch instances per engine id (one per module)
    meta_search_by_engine: dict[int, MetadataSearch] = {}

    valid_embeddings = []
    valid_scores = []
    skipped = 0

    total = len(rated_files)
    start_time = time.time()

    for idx, item in enumerate(rated_files):
        file_path = item["file_path"]
        engine = item["engine"]
        embedding_method = item["embedding_method"]

        if callback:
            elapsed = time.time() - start_time
            eta = (elapsed / (idx + 1)) * (total - idx - 1) if idx > 0 else 0
            eta_str = f"{eta:.0f}s" if eta < 120 else f"{eta / 60:.1f}min"
            callback(
                f"Generating descriptions & embeddings: {idx + 1}/{total} "
                f"(ETA: {eta_str})",
                (idx + 1) / total * 0.5,  # first 50% of progress
                0,
            )

        try:
            # ------------------------------------------------------------------
            # Full-text path: use pre-computed embeddings from TextSearch cache.
            # ------------------------------------------------------------------
            if embedding_method == "full_text" and cfg.evaluator.text_embedding_method == "full_text":
                chunk_embeddings = precomputed_embeddings.get(file_path)
                if chunk_embeddings is None or len(chunk_embeddings) == 0:
                    skipped += 1
                    continue

            # ------------------------------------------------------------------
            # Metadata path: generate description → embed with TextEmbedder.
            # ------------------------------------------------------------------
            else:
                # Get or create MetadataSearch for this engine
                eid = id(engine)
                if eid not in meta_search_by_engine:
                    ms = MetadataSearch.__new__(MetadataSearch)
                    # Manually init without the singleton guard to allow per-engine instances
                    ms.engine = engine
                    ms.cfg = cfg
                    ms.text_embedder = text_embedder
                    ms._initialized = True

                    # Reuse shared cache; create fresh OmniDescriptor reference
                    from src.omni_descriptor import OmniDescriptor
                    ms.omni_descriptor = OmniDescriptor(cfg)

                    cache_folder = os.path.join(cfg.main.cache_path, 'metadata_cache')
                    if MetadataSearch._shared_cache is None:
                        from src.caching import TwoLevelCache
                        import threading
                        with MetadataSearch._cache_lock:
                            if MetadataSearch._shared_cache is None:
                                MetadataSearch._shared_cache = TwoLevelCache(
                                    cache_dir=cache_folder, name="metadata_cache"
                                )
                    ms._fast_cache = MetadataSearch._shared_cache

                    meta_search_by_engine[eid] = ms

                meta_search = meta_search_by_engine[eid]

                # Generate description using only cached auto-descriptions (fast)
                description = meta_search.generate_full_description(
                    file_path,
                    media_folder=item["media_folder"],
                    generate_desc_if_not_in_cache=False,
                )

                if not description or len(description.strip()) < 10:
                    skipped += 1
                    continue

                # Unload OmniDescriptor (just in case it was touched)
                meta_search.omni_descriptor.unload()

                # Convert description to text embeddings (list of chunk embeddings)
                chunk_embeddings = text_embedder.embed_text(description)

                if chunk_embeddings is None or len(chunk_embeddings) == 0:
                    skipped += 1
                    continue

            valid_embeddings.append(np.array(chunk_embeddings, dtype=np.float32))
            valid_scores.append(item["user_rating"])

        except Exception as e:
            print(f"[UniversalTrain] Error processing {file_path}: {e}")
            skipped += 1
            continue

    print(f"[UniversalTrain] Valid: {len(valid_embeddings)}, Skipped: {skipped}")

    # -------------------------------------------------------------------------
    # Phase C: collect pre-embedded pairs from auto-discovered module train.py
    # files (e.g. web_search).  These modules embed their own data, so no
    # further description generation or embedding is needed here.
    # -------------------------------------------------------------------------
    extra_embeddings, extra_scores = _gather_from_module_train_files(
        cfg, text_embedder,
        status_callback=lambda msg: callback(msg, 0, 0) if callback else None,
    )
    if extra_embeddings:
        print(f"[UniversalTrain] Auto-discovered modules: +{len(extra_embeddings)} additional pairs.")
        valid_embeddings.extend(extra_embeddings)
        valid_scores.extend(extra_scores)

    if len(valid_embeddings) < 2:
        msg = (f"Not enough valid embeddings ({len(valid_embeddings)}) to train. "
               "Need at least 2. Abort training.")
        print(f"[UniversalTrain] {msg}")
        if callback:
            callback(msg, 100, 0)
        return

    # 3. Split into train / test sets
    status = 'Training the universal evaluator model...'
    print(f"[UniversalTrain] {status}")
    print(f"[UniversalTrain] Training on {len(valid_embeddings)} files.")

    evaluator = UniversalEvaluator()
    evaluator.reinitialize()

    X_train, X_test, y_train, y_test = train_test_split(
        valid_embeddings, valid_scores, test_size=0.1, random_state=42
    )

    print(f"[UniversalTrain] X_train: {len(X_train)}, X_test: {len(X_test)}")
    print(f"[UniversalTrain] y_train min/max: {min(y_train)}/{max(y_train)}")

    # Baseline accuracy (predict mean)
    mean_score = np.mean(y_train)
    baseline_accuracy = 1 - np.mean(
        np.abs(mean_score - np.array(y_test)) / (np.array(y_test) + evaluator.mape_bias)
    )

    # 4. Training loop
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_epoch = 0
    total_epochs = 5001
    batch_size = 16

    model_save_path = os.path.join(cfg.main.personal_models_path, 'universal_evaluator.pt')

    pbar = tqdm(range(total_epochs), desc="Universal training")

    for epoch in pbar:
        train_accuracy, test_accuracy = evaluator.train(
            X_train, y_train, X_test, y_test, batch_size=batch_size
        )

        pbar.set_description(
            f'Epoch: {epoch + 1}, '
            f'Train: {train_accuracy * 100:.2f}%, '
            f'Test: {test_accuracy * 100:.2f}%'
        )

        if callback:
            percent = 0.5 + (epoch + 1) / total_epochs * 0.5  # second 50% of progress
            callback(status, percent, baseline_accuracy, train_accuracy, test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            evaluator.save(model_save_path)

    status = (
        f'Best Epoch: {best_epoch}, '
        f'Train Accuracy: {best_train_accuracy * 100:.2f}%, '
        f'Test Accuracy: {best_test_accuracy * 100:.2f}%'
    )
    print(f"[UniversalTrain] {status}")
    if callback:
        callback(status, 100, baseline_accuracy)

    # Reload best checkpoint
    evaluator.load(model_save_path)
    print('[UniversalTrain] Training complete! Universal evaluator is ready.')
