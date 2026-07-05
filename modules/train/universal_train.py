"""
Universal evaluator training module.

Gathers (chunk_embeddings, user_rating) pairs from the durable memory folder
(``project_config/memory/<YYYY-MM-DD>/<soft_hash>.md``). Each memory file holds
the rich text description of a rated file (tags/fingerprint/omni/internal/.meta);
the rating itself lives in the ``FilesLibrary`` table (joined by soft hash) so
the evaluator never sees the score in the text it embeds and cannot cheat.
"""

from tqdm import tqdm
import numpy as np
import os
import time
import re
import glob
from collections import Counter

from sklearn.model_selection import train_test_split
from src.universal_evaluator import UniversalEvaluator
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
# Memory-folder gathering
# ---------------------------------------------------------------------------

def _parse_memory_file(text):
    """Split a memory .md into (rating, description_text).

    Line 1 must be ``Rating: <float>``. If it isn't (missing or unparseable),
    the file is considered broken/foreign and we return ``(None, None)`` so the
    caller skips it. The description is everything from line 2 onward — i.e.
    the whole file minus the rating line, so the embedder never sees the score.
    """
    lines = text.split('\n', 1)
    if not lines:
        return None, None
    m = re.match(r'^Rating:\s*(\S+)', lines[0].strip())
    if not m:
        return None, None  # no rating on line 1 → broken/unrelated file
    try:
        rating = float(m.group(1))
    except ValueError:
        return None, None
    description = lines[1] if len(lines) > 1 else ""
    return rating, description.strip()


def _gather_from_memory(cfg, text_embedder, status_callback=None):
    """Collect (chunk_embeddings, score) pairs from the memory folder.

    Walks ``cfg.main.memory_path/<YYYY-MM-DD>/*.md``, parses the rating from
    the first line of each file, deduplicates by soft hash (keeping the most
    recent dated folder = the user's latest opinion), and embeds the
    description text (everything after the header block, with the rating line
    stripped) with the shared Jina text embedder.

    No DB lookup is needed — the rating lives on line 1 of each memory file.
    """
    memory_path = cfg.main.memory_path
    if not os.path.isdir(memory_path):
        print(f"[UniversalTrain] Memory folder not found: {memory_path}")
        return [], []

    # Date folders sort chronologically: YYYY-MM-DD. Walk oldest -> newest so
    # that newer files overwrite older ones for the same soft hash.
    date_dirs = sorted(
        (d for d in glob.glob(os.path.join(memory_path, '*')) if os.path.isdir(d))
    )

    hash_to_entry = {}  # soft_hash -> (rating, description)
    for date_dir in date_dirs:
        for md_path in glob.glob(os.path.join(date_dir, '*.md')):
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            except Exception as exc:
                print(f"[UniversalTrain] Could not read {md_path}: {exc}")
                continue
            rating, soft_hash, description = _parse_memory_file(text)
            if soft_hash is None or rating is None or len(description) < 10:
                continue
            hash_to_entry[soft_hash] = (rating, description)  # latest wins

    if not hash_to_entry:
        print("[UniversalTrain] No usable memory files found.")
        return [], []

    print(f"[UniversalTrain] {len(hash_to_entry)} unique rated memory entries.")

    all_embeddings = []
    all_scores = []
    count = 0
    for rating, description in hash_to_entry.values():
        if count % 100 == 0 and status_callback:
            status_callback(f"Gathering memory pairs ({count})...")
        chunk_embeddings = text_embedder.embed_text(description)
        if chunk_embeddings is None or len(chunk_embeddings) == 0:
            continue
        all_embeddings.append(np.array(chunk_embeddings, dtype=np.float32))
        all_scores.append(float(rating))
        count += 1

    print(f"[UniversalTrain] Collected {count} training pairs from memory.")
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

    # 1. Initialise text embedder (shared across all memory description embeds)
    text_embedder = TextEmbedder(cfg)
    text_embedder.initiate(models_folder=cfg.main.embedding_models_path)

    # 2. Gather (chunk_embeddings, score) pairs from the durable memory folder.
    #    Descriptions come from memory/<date>/<soft_hash>.md; ratings are joined
    #    from FilesLibrary by soft hash (never stored inside the .md text).
    valid_embeddings, valid_scores = _gather_from_memory(
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

    # 4. Training loop — runs entirely inside the UniversalEvaluator subprocess.
    #    Progress messages are streamed back to the callback in real time.
    total_epochs = max_steps if max_steps is not None else 5001
    batch_size = 32

    model_save_path = os.path.join(cfg.main.personal_models_path, 'universal_evaluator.pt')

    print(f"[UniversalTrain] Starting training loop for up to {total_epochs} epochs via subprocess...")
    print(f"[UniversalTrain] Training samples: {len(X_train)}, Baseline Accuracy: {baseline_accuracy * 100:.2f}%")

    def _progress_handler(data):
        """Relay subprocess progress messages to the caller's callback."""
        if callback is None:
            return
        if data['type'] == 'initial_eval':
            # Epoch-0 baseline point for the UI chart
            callback(status, 0, baseline_accuracy, data['train_acc'], data['test_acc'])
        elif data['type'] == 'epoch':
            percent = (data['epoch'] + 1) / total_epochs
            callback(status, percent, baseline_accuracy, data['train_acc'], data['test_acc'])

    result = evaluator.train_full(
        X_train, y_train, X_test, y_test,
        sample_weights,
        total_epochs,
        batch_size,
        time_budget_seconds,
        model_save_path,
        progress_callback=_progress_handler,
    )

    best_epoch = result['best_epoch']
    best_train_accuracy = result['best_train_accuracy']
    best_test_accuracy = result['best_test_accuracy']

    status = (
        f'Best Epoch: {best_epoch}, '
        f'Train Accuracy: {best_train_accuracy * 100:.2f}%, '
        f'Test Accuracy: {best_test_accuracy * 100:.2f}%'
    )
    print(f"[UniversalTrain] {status}")
    if callback:
        callback(status, 100, baseline_accuracy)

    # The subprocess already reloaded the best checkpoint internally.
    # evaluator.hash is mirrored by train_full() so downstream hash checks work.
    print('[UniversalTrain] Training complete! Universal evaluator is ready.')
