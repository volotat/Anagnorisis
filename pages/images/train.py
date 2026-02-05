from tqdm import tqdm
import numpy as np
import pages.images.db_models as db_models
import src.scoring_models
from sklearn.model_selection import train_test_split
from pages.images.engine import ImageSearch, ImageEvaluator
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