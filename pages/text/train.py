from tqdm import tqdm
import numpy as np
import pages.text.db_models as db_models
import src.scoring_models
from sklearn.model_selection import train_test_split
from pages.text.engine import TextSearch, TextEvaluator
import os
import torch


def train_text_evaluator(cfg, callback=None):
    # Create / fetch the singleton evaluator
    evaluator = TextEvaluator()
    evaluator.reinitialize()  # Reset weights for training from scratch

    # Initialize TextSearch to access embeddings & model hash
    text_engine = TextSearch(cfg=cfg)
    text_engine.initiate(
        models_folder=cfg.main.embedding_models_path,
        cache_folder=cfg.main.cache_path,
    )

    # Fetch all text library entries that have a user rating
    text_library_entries = db_models.TextLibrary.query.filter(
        db_models.TextLibrary.user_rating.isnot(None)
    ).all()

    if not text_library_entries:
        print("No rated text files found in the database. Abort training.")
        return

    # Build file paths and labels from DB
    media_dir = cfg.text.media_directory
    file_paths = [os.path.join(media_dir, e.file_path) for e in text_library_entries]
    text_scores = [e.user_rating for e in text_library_entries]

    # Process files to get embeddings: list[list[np.ndarray]]
    embeddings = text_engine.process_files(file_paths, media_folder=media_dir)

    # Filter out files with empty embeddings (failed or missing)
    valid_embeddings = []
    valid_scores = []
    for emb, score in zip(embeddings, text_scores):
        if emb is not None and len(emb) > 0:
            valid_embeddings.append(np.array(emb, dtype=np.float32))
            valid_scores.append(score)

    if len(valid_embeddings) == 0:
        print("No valid embeddings found for rated text files. Abort training.")
        return

    print(f"Training on {len(valid_embeddings)} rated text files.")

    # Split into train and eval sets
    status = 'Training the text evaluator model...'
    print(status)
    X_train, X_test, y_train, y_test = train_test_split(
        valid_embeddings, valid_scores, test_size=0.1, random_state=42
    )

    print(f"X_train: {len(X_train)}, X_test: {len(X_test)}")
    print(f"y_train min/max: {min(y_train)}/{max(y_train)}")

    # Calculate baseline accuracy (predict mean for everything)
    mean_score = np.mean(y_train)
    baseline_accuracy = 1 - np.mean(
        np.abs(mean_score - np.array(y_test)) / (np.array(y_test) + evaluator.mape_bias)
    )

    # Training loop
    best_train_accuracy = 0
    best_test_accuracy = 0
    best_epoch = 0
    total_epochs = 5001
    batch_size = 16

    pbar = tqdm(range(total_epochs))

    for epoch in pbar:
        train_accuracy, test_accuracy = evaluator.train(
            X_train, y_train, X_test, y_test, batch_size=batch_size
        )

        pbar.set_description(
            f'Epoch: {epoch+1}, '
            f'Train Metric: {train_accuracy * 100:.2f}%, '
            f'Test Metric: {test_accuracy * 100:.2f}%'
        )

        if callback:
            percent = (epoch + 1) / total_epochs
            callback(status, percent, baseline_accuracy, train_accuracy, test_accuracy)

        if test_accuracy > best_test_accuracy:
            best_train_accuracy = train_accuracy
            best_test_accuracy = test_accuracy
            best_epoch = epoch + 1

            # Save the best model
            evaluator.save(
                os.path.join(cfg.main.personal_models_path, 'text_evaluator.pt')
            )

    status = (
        f'Best Epoch: {best_epoch}, '
        f'Train Accuracy: {best_train_accuracy * 100:.2f}%, '
        f'Test Accuracy: {best_test_accuracy * 100:.2f}%'
    )
    print(status)
    if callback:
        callback(status, 100, baseline_accuracy)

    # Reload the best checkpoint
    evaluator.load(
        os.path.join(cfg.main.personal_models_path, 'text_evaluator.pt')
    )
    print('Training complete! Now you can use the new model to evaluate text files.')
