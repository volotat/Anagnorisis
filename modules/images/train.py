from tqdm import tqdm
import numpy as np
import modules.images.db_models as db_models
import src.scoring_models
from sklearn.model_selection import train_test_split
from modules.images.engine import ImageSearch, ImageEvaluator
import os
import pickle
import torch  

def train_image_evaluator(cfg, callback=None):
  # Create the model
  evaluator = ImageEvaluator() #src.scoring_models.Evaluator(embedding_dim=768, rate_classes=11)
  evaluator.reinitialize() # In case the model was already loaded 

  # Initialize ImagesSearch to access the cache and model hash
  images_engine = ImageSearch(cfg=cfg)
  images_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)

  # Create dataset from DB, select only images with user rating
  images_library_entries = db_models.ImagesLibrary.query.filter(
    db_models.ImagesLibrary.user_rating.isnot(None)
  ).all()

  # Build file paths and labels from DB, then extract embeddings via engine
  media_dir = cfg.images.media_directory
  file_paths = [os.path.join(media_dir, e.file_path) for e in images_library_entries]
  image_scores = [e.user_rating for e in images_library_entries]

  embeddings = images_engine.process_files(file_paths, media_folder=media_dir)
  # Keep only non-zero embeddings (failed or missing files become zero vectors)
  mask = embeddings.abs().sum(dim=1) > 0
  if mask.sum().item() == 0:
    print("No valid embeddings found for rated tracks. Abort training.")
    return
  image_embeddings = embeddings[mask].to(evaluator.device)
  image_scores = [s for s, m in zip(image_scores, mask.tolist()) if m]

  # Split to train and eval sets
  status = 'Training the model...'
  print(status)
  X_train, X_test, y_train, y_test = train_test_split(image_embeddings, image_scores, test_size=0.1, random_state=42)

  print("X_train:", len(X_train), "X_test:", len(X_test))
  print("y_train min max:", min(y_train), max(y_train))

  # Calculate the mean score of all train scores
  mean_score = np.mean(y_train)
  # Calculate baseline accuracy
  baseline_accuracy = 1 - np.mean(np.abs(mean_score - np.array(y_test)) / (np.array(y_test) + evaluator.mape_bias))

  # Train the model
  best_train_accuracy = 0
  best_test_accuracy = 0
  best_epoch = 0
  total_epochs = 5001

  # Initialize the progress bar
  pbar = tqdm(range(total_epochs))

  for epoch in pbar:
    # Train the model
    train_accuracy, test_accuracy = evaluator.train(X_train, y_train, X_test, y_test, batch_size=64)

    # Update the progress bar description
    pbar.set_description(f'Epoch: {epoch+1}, Train Metric: {train_accuracy * 100:.2f}%, Test Metric: {test_accuracy * 100:.2f}%')

    if callback:
      percent = (epoch+1) / total_epochs
      callback(status, percent, baseline_accuracy, train_accuracy, test_accuracy)

    # Check if this epoch's accuracy is the best
    if test_accuracy > best_test_accuracy:
      best_train_accuracy = train_accuracy
      best_test_accuracy = test_accuracy
      best_epoch = epoch + 1

      # Save the model
      evaluator.save(os.path.join(cfg.main.personal_models_path, 'image_evaluator.pt'))

  status = f'Best Epoch: {best_epoch}, Train Accuracy: {best_train_accuracy * 100:.2f}%, Test Accuracy: {best_test_accuracy * 100:.2f}%'
  print(status)
  if callback: 
    callback(status, 100, baseline_accuracy)

  evaluator.load(os.path.join(cfg.main.personal_models_path, 'image_evaluator.pt'))
  print('Training complete! Now you can use new model to evaluate images.')


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