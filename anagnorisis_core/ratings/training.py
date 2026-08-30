"""
Universal evaluator training module.

Gathers (chunk_embeddings, user_rating) pairs from the durable memory folder
(``project_config/memory/<YYYY-MM-DD>/<soft_hash>.md``). Each memory file holds
the rich text description of a rated file (tags/fingerprint/omni/internal/.meta);
the rating is line 1 of each memory file and is stripped before embedding, so
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
from anagnorisis_core.models.universal_evaluator import UniversalEvaluator
from anagnorisis_core.models.embedder import get_omni_embedder
from anagnorisis_core.storage.caching import get_two_level_cache


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


def _format_eta(seconds: float) -> str:
    """A short, human-readable remaining time for a status line."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m {seconds % 60:.0f}s"
    return f"{seconds // 3600:.0f}h {(seconds % 3600) / 60:.0f}m"


def _epoch_status(epoch, total_epochs, *, elapsed, time_budget_seconds,
                  best_accuracy, best_epoch, current_accuracy) -> str:
    """The status line for one training epoch.

    Says how far along the run is and what the best score so far is, because
    the number worth watching is the best rather than the current one: that is
    the epoch whose weights get kept.
    """
    remaining = max(total_epochs - epoch, 0)
    per_epoch = elapsed / epoch if epoch else 0.0
    eta = remaining * per_epoch

    # The loop stops at whichever limit comes first, so the estimate has to
    # respect both or it promises time the run will not take.
    if time_budget_seconds is not None:
        eta = min(eta, max(time_budget_seconds - elapsed, 0.0))

    return (f'Epoch {epoch}/{total_epochs}, {remaining} left (~{_format_eta(eta)}). '
            f'Best {best_accuracy * 100:.2f}% at epoch {best_epoch}, '
            f'now {current_accuracy * 100:.2f}%.')


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

    def report(message):
        """Hand a status line upwards.

        Called on every iteration rather than every N. Both consumers throttle
        already (the task context at 0.25s, the socket emit at 1s).
        """
        if status_callback:
            status_callback(message)

    # Date folders sort chronologically: YYYY-MM-DD. Walk oldest -> newest so
    # that newer files overwrite older ones for the same soft hash.
    date_dirs = sorted(
        (d for d in glob.glob(os.path.join(memory_path, '*')) if os.path.isdir(d))
    )

    # Listed up front so every message below can say what it counts towards.
    report("Looking for memory files...")
    md_paths = []
    for date_dir in date_dirs:
        md_paths.extend(sorted(glob.glob(os.path.join(date_dir, '*.md'))))

    hash_to_entry = {}  # soft_hash -> (rating, description)
    for i, md_path in enumerate(md_paths, 1):
        report(f"Reading memory files ({i}/{len(md_paths)})...")
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except Exception as exc:
            print(f"[UniversalTrain] Could not read {md_path}: {exc}")
            continue
        # The filename *is* the soft hash — memory files are written as
        # <soft_hash>.md — so the parser does not need to return it. It used
        # to be unpacked from here anyway, which raised ValueError on the
        # first file and meant training could not run at all.
        soft_hash = os.path.splitext(os.path.basename(md_path))[0]
        rating, description = _parse_memory_file(text)
        if rating is None or description is None or len(description) < 10:
            continue
        hash_to_entry[soft_hash] = (rating, description)  # latest wins

    if not hash_to_entry:
        print("[UniversalTrain] No usable memory files found.")
        return [], []

    print(f"[UniversalTrain] {len(hash_to_entry)} unique rated memory entries.")

    # Memory files never change: the filename is the soft hash of the file it
    # describes, and a new opinion is written as a new file in a new date
    # folder. So an embedding of one is valid forever, for as long as the model
    # that produced it is the model in use. Without this every training run
    # re-embedded the whole corpus from scratch.
    cache = get_two_level_cache(
        cache_dir=os.path.join(cfg.main.cache_path, 'memory_embeddings'),
        name="memory_training",
    )
    model_hash = getattr(text_embedder, 'model_hash', None) or 'unknown-model'

    all_embeddings = []
    all_scores = []
    count = 0
    reused = 0
    total = len(hash_to_entry)
    started = time.time()

    for i, (soft_hash, (rating, description)) in enumerate(hash_to_entry.items(), 1):
        # An estimate from one or two samples is worse than none: the first item
        # carries the whole warm-up and reads as "0s left" right before a
        # ten-minute wait. Wait until the average means something.
        elapsed = time.time() - started
        if i > 5 and elapsed > 0:
            eta = (total - i + 1) * (elapsed / (i - 1))
            tail = f", {_format_eta(eta)} left"
        else:
            tail = ""
        report(f"Embedding memory descriptions ({i}/{total}{tail})...")

        cache_key = f"memory_emb::{soft_hash}::{model_hash}"
        chunk_embeddings = cache.get(cache_key)
        if chunk_embeddings is None:
            # embed_long_text is the real name; embed_text is an alias for it on
            # the embedder. Spelled out here so it is obvious that a long
            # description is chunked and every chunk kept, rather than truncated.
            chunk_embeddings = text_embedder.embed_long_text(description)
            if chunk_embeddings is None or len(chunk_embeddings) == 0:
                continue
            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
            cache.set(cache_key, chunk_embeddings)
        else:
            chunk_embeddings = np.array(chunk_embeddings, dtype=np.float32)
            reused += 1

        all_embeddings.append(chunk_embeddings)
        all_scores.append(float(rating))
        count += 1

    print(f"[UniversalTrain] Collected {count} training pairs from memory "
          f"({reused} reused from cache, {count - reused} newly embedded) "
          f"in {time.time() - started:.1f}s.")
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
    #
    # Loading the model takes on the order of ten seconds and used to happen
    # with nothing said, so the page sat on whatever it had shown before and
    # the run looked stuck before it had even started.
    if callback:
        callback('Loading the embedding model...', 0, 0)
    load_started = time.time()
    text_embedder = get_omni_embedder(cfg)
    text_embedder.initiate(models_folder=cfg.main.embedding_models_path)
    print(f"[UniversalTrain] Embedding model ready in {time.time() - load_started:.1f}s.")

    # 2. Gather (chunk_embeddings, score) pairs from the durable memory folder.
    #    Descriptions come from memory/<date>/<soft_hash>.md; ratings are joined
    #    from line 1 of each memory file, which is stripped before embedding.
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
    print("[UniversalTrain] Training the universal evaluator model...")

    if callback:
        callback(f'Preparing {len(valid_embeddings)} training pairs...', 0, 0)

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

    # Mirrors the subprocess's own rule for "best": the highest test accuracy
    # seen so far, which is also the epoch whose weights were checkpointed.
    best = {'accuracy': 0.0, 'epoch': 0}
    loop_started = time.time()

    def _progress_handler(data):
        """Relay subprocess progress messages to the caller's callback.

        Reports the running best and how far along the run is, rather than one
        fixed sentence for the whole loop. Thousands of epochs behind a status
        that never changes gives no sign of whether the model is improving, or
        indeed still moving.
        """
        if callback is None:
            return

        if data['type'] == 'initial_eval':
            # Epoch-0 baseline point for the UI chart.
            callback(f'Scoring before training, epoch 0 of {total_epochs}...',
                     0, baseline_accuracy, data['train_acc'], data['test_acc'])
            return

        if data['type'] != 'epoch':
            return

        epoch = data['epoch'] + 1
        if data['test_acc'] > best['accuracy']:
            best['accuracy'] = data['test_acc']
            best['epoch'] = epoch

        callback(
            _epoch_status(
                epoch, total_epochs,
                elapsed=time.time() - loop_started,
                time_budget_seconds=time_budget_seconds,
                best_accuracy=best['accuracy'], best_epoch=best['epoch'],
                current_accuracy=data['test_acc'],
            ),
            epoch / total_epochs, baseline_accuracy,
            data['train_acc'], data['test_acc'],
        )

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
