"""
Universal evaluator training module.

Gathers (chunk_embeddings, user_rating) pairs from ALL media modules by
auto-discovering modules/*/train.py files and calling get_training_pairs()
on each.  Every module is fully responsible for its own DB queries and
text embedding — no central registry required.  New modules contribute to
training simply by adding a train.py with get_training_pairs().
"""

from tqdm import tqdm
import numpy as np
import os
import time
from collections import Counter

from sklearn.model_selection import train_test_split
from src.scoring_models import TransformerEvaluator
from src.text_embedder import TextEmbedder


# ---------------------------------------------------------------------------
# Training augmentation switches
# ---------------------------------------------------------------------------
# Add random (nonsensical) embeddings mapped to score 0.  This teaches the
# evaluator that meaningless content should receive the lowest rating.
ENABLE_NONSENSICAL_NEGATIVES = True
NONSENSICAL_COUNT = 2000  # number of synthetic zero-score samples to inject

# Oversample underrepresented score bins up to the median bin count so that
# the model sees a roughly balanced distribution during training.
ENABLE_OVERSAMPLING = False

# Apply per-sample inverse-frequency loss weighting so rare scores contribute
# proportionally more to the gradient even without duplicating data.
# This might hurt the performance as the task essentially a regression, so 
# weights might skew the values.
ENABLE_LOSS_WEIGHTING = False



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
# Auto-discovery: gather training pairs from modules/*/train.py
# ---------------------------------------------------------------------------
def _gather_from_module_train_files(cfg, text_embedder, status_callback=None):
    """
    Scan modules/*/train.py for modules that expose get_training_pairs() and
    collect already-embedded (chunk_embeddings, score) pairs from each.

    All modules — including built-in ones (music, images, videos, text) — are
    handled here.  Each module's train.py is fully responsible for querying
    its own DB and producing ready-to-use text embeddings.
    """
    import importlib
    import glob

    # modules/ directory is one level above this file (modules/train/)
    pages_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    )

    all_embeddings = []
    all_scores = []

    train_files = sorted(glob.glob(os.path.join(pages_dir, '*', 'train.py')))

    for train_file in train_files:
        module_name = os.path.basename(os.path.dirname(train_file))

        # Skip template / hidden folders
        if module_name.startswith('_'):
            continue

        module_import_path = f"modules.{module_name}.train"
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
                all_embeddings.append(chunk_embeddings)
                all_scores.append(score)
                count += 1
        except Exception as exc:
            print(f"[UniversalTrain] Error in get_training_pairs for '{module_name}': {exc}")
            continue

        print(f"[UniversalTrain] '{module_name}': collected {count} training pairs.")

    return all_embeddings, all_scores




# ---------------------------------------------------------------------------
# Main training entry-point
# ---------------------------------------------------------------------------
def train_universal_evaluator(cfg, callback=None, max_steps=None, time_budget_seconds=None):
    """
    Train a single TransformerEvaluator on text-embedding representations of
    all user-rated files across every configured media module.

    Parameters
    ----------
    cfg : OmegaConf config
    callback : optional ``(status, percent, baseline_accuracy, train_accuracy, test_accuracy)``
    max_steps : int or None
        Maximum number of training epochs.  Defaults to 5001 when None.
    time_budget_seconds : float or None
        Wall-clock seconds allowed for the training loop.  Counting starts
        at the first epoch (after all data preparation is done).  When both
        *max_steps* and *time_budget_seconds* are supplied whichever limit is
        hit first stops training.
    """

    print("=" * 60)
    print("[UniversalTrain] Starting universal evaluator training")
    print("=" * 60)

    # 1. Initialise text embedder (shared across all module get_training_pairs calls)
    text_embedder = TextEmbedder(cfg)
    text_embedder.initiate(models_folder=cfg.main.embedding_models_path)

    # 2. Gather (chunk_embeddings, score) pairs from all modules via auto-discovery.
    #    Each module's train.py is responsible for querying its own DB, reading
    #    its files, and producing ready-to-use text embeddings.
    valid_embeddings, valid_scores = _gather_from_module_train_files(
        cfg, text_embedder,
        status_callback=lambda msg: callback(msg, 0, 0) if callback else None,
    )

    print(f"[UniversalTrain] Total training pairs across all modules: {len(valid_embeddings)}")

    if len(valid_embeddings) == 0:
        msg = "No rated files found across any module. Abort training."
        print(f"[UniversalTrain] {msg}")
        if callback:
            callback(msg, 100, 0)
        return

    if len(valid_embeddings) < 2:
        msg = (f"Not enough valid embeddings ({len(valid_embeddings)}) to train. "
               "Need at least 2. Abort training.")
        print(f"[UniversalTrain] {msg}")
        if callback:
            callback(msg, 100, 0)
        return

    # 3. Split into train / test sets FIRST (on original, unaugmented data)
    #    so the test set is a clean, unseen holdout with no duplicates.
    status = 'Training the universal evaluator model...'
    print(f"[UniversalTrain] {status}")

    evaluator = UniversalEvaluator()
    evaluator.reinitialize()

    X_train, X_test, y_train, y_test = train_test_split(
        valid_embeddings, valid_scores, test_size=0.1, random_state=42
    )

    print(f"[UniversalTrain] Original split — X_train: {len(X_train)}, X_test: {len(X_test)}")
    print(f"[UniversalTrain] y_train min/max: {min(y_train)}/{max(y_train)}")

    # -------------------------------------------------------------------------
    # Phase D: augment training split; mirror a proportional share onto test
    # so both metrics are computed on the same score distribution and the
    # accuracy graph is directly comparable between the two curves.
    # -------------------------------------------------------------------------
    embedding_dim = X_train[0].shape[-1]  # typically 1024
    n_real_train = len(X_train)   # pre-augmentation size, used for ratio below

    # D-1) Inject nonsensical (random noise) embeddings → score 0
    if ENABLE_NONSENSICAL_NEGATIVES and NONSENSICAL_COUNT > 0:
        rng = np.random.default_rng(seed=42)
        for _ in range(NONSENSICAL_COUNT):
            n_chunks = rng.integers(1, 6)  # 1-5 random chunks
            noise = rng.standard_normal((n_chunks, embedding_dim)).astype(np.float32)
            X_train.append(noise)
            y_train.append(0.0)
        print(f"[UniversalTrain] Injected {NONSENSICAL_COUNT} nonsensical zero-score samples into train set.")

        # Mirror a proportional amount into the test set so both curves start
        # from the same baseline difficulty level and remain comparable.
        # Use the noise fraction of the AUGMENTED train set (noise / total),
        # applied directly to the original test size.  This prevents the test
        # set from being dominated when NONSENSICAL_COUNT >> n_real_train.
        # e.g. 2000 noise / 2090 total = 95.7% → add 0.957 * 10 ≈ 10 noise to
        # a test set of 10 real samples, giving 50% noise (vs 96% in train).
        noise_fraction = NONSENSICAL_COUNT / (n_real_train + NONSENSICAL_COUNT)
        n_test_nonsensical = round(len(X_test) * noise_fraction)
        rng_test = np.random.default_rng(seed=99)
        for _ in range(n_test_nonsensical):
            n_chunks = rng_test.integers(1, 6)
            noise = rng_test.standard_normal((n_chunks, embedding_dim)).astype(np.float32)
            X_test.append(noise)
            y_test.append(0.0)
        print(f"[UniversalTrain] Injected {n_test_nonsensical} nonsensical zero-score samples into test set.")

    # D-2) Oversample underrepresented score bins to median bin count
    if ENABLE_OVERSAMPLING:
        bins = [round(s) for s in y_train]
        bin_counts = Counter(bins)
        median_count = int(np.median(list(bin_counts.values())))
        oversampled_embs = []
        oversampled_scores = []
        rng_os = np.random.default_rng(seed=123)
        for bin_val, count in sorted(bin_counts.items()):
            if count >= median_count:
                continue
            idxs = [i for i, s in enumerate(y_train) if round(s) == bin_val]
            need = median_count - count
            chosen = rng_os.choice(idxs, size=need, replace=True)
            for ci in chosen:
                oversampled_embs.append(X_train[ci])
                oversampled_scores.append(y_train[ci])
        if oversampled_embs:
            X_train.extend(oversampled_embs)
            y_train.extend(oversampled_scores)
            print(f"[UniversalTrain] Oversampled +{len(oversampled_embs)} samples "
                  f"(target median bin count: {median_count}).")
            final_bins = Counter(round(s) for s in y_train)
            print(f"[UniversalTrain] Train score distribution after augmentation: "
                  f"{dict(sorted(final_bins.items()))}")

    print(f"[UniversalTrain] Training on {len(X_train)} samples (test set: {len(X_test)}).")

    # D-3) Compute per-sample loss weights on the final augmented training set
    sample_weights = None
    if ENABLE_LOSS_WEIGHTING:
        train_bins = np.array([round(s) for s in y_train])
        bin_counts_train = Counter(train_bins.tolist())
        total_train = len(y_train)
        n_bins_present = len(bin_counts_train)
        # weight = total / (n_bins * count_of_this_bin)  → rare bins weigh more
        sample_weights = np.array([
            total_train / (n_bins_present * bin_counts_train[round(s)])
            for s in y_train
        ], dtype=np.float32)
        # Normalise so mean weight == 1 (keeps learning rate semantics stable)
        sample_weights = sample_weights / sample_weights.mean()
        print(f"[UniversalTrain] Loss weighting enabled.  "
              f"Weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")

    # Baseline accuracy (predict mean of augmented train) vs clean test set.
    # Uses the full augmented y_train distribution so the baseline is comparable
    # to what the model actually sees during training.
    mean_score = np.mean(y_train)
    baseline_accuracy = 1 - np.mean(
        np.abs(mean_score - np.array(y_test)) / (np.array(y_test) + evaluator.mape_bias)
    )

    # 4. Training loop
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_epoch = 0
    total_epochs = max_steps if max_steps is not None else 5001
    batch_size = 16

    model_save_path = os.path.join(cfg.main.personal_models_path, 'universal_evaluator.pt')

    # pbar = tqdm(range(total_epochs), desc="Universal training")
    training_loop_start = time.time()

    # Send intial train and test accuracy (before any training) to the callback so the UI can show a baseline point on the graph.
    if callback:
        train_accuracy, test_accuracy = evaluator.evaluate(X_train, y_train, X_test, y_test, batch_size=batch_size)
        callback(status, 0, baseline_accuracy, train_accuracy, test_accuracy)

    print(f"[UniversalTrain] Starting training loop for up to {total_epochs} epochs...")
    print(f"[UniversalTrain] Training samples: {len(X_train)}, Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

    for epoch in range(total_epochs):
        train_accuracy, test_accuracy = evaluator.train(
            X_train, y_train, X_test, y_test,
            batch_size=batch_size,
            sample_weights=sample_weights,
        )

        # pbar.set_description(
        #     f'Epoch: {epoch + 1}, '
        #     f'Train: {train_accuracy * 100:.2f}%, '
        #     f'Test: {test_accuracy * 100:.2f}%'
        # )

        if callback:
            percent = (epoch + 1) / total_epochs
            callback(status, percent, baseline_accuracy, train_accuracy, test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1
            evaluator.save(model_save_path)

        if time_budget_seconds is not None:
            elapsed_training = time.time() - training_loop_start
            if elapsed_training >= time_budget_seconds:
                print(f"[UniversalTrain] Time budget of {time_budget_seconds}s reached after epoch {epoch + 1}.")
                break

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
