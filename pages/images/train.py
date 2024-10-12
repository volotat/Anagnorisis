from tqdm import tqdm
import numpy as np
import pages.images.db_models as db_models
import src.scoring_models
from sklearn.model_selection import train_test_split
from pages.images.engine import ImageSearch
import os



def train_image_evaluator(cfg, callback=None):
  # Create dataset from DB, select only images with user rating
  images_library_entries = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.user_rating != None).all()


  image_files = [entry.file_path for entry in images_library_entries]
  image_scores = [entry.user_rating for entry in images_library_entries]

  # get embeddings for images
  status = "Step 1/2: Gathering embeddings for image files..."

  def send_emb_extracting_status(num_extracted, num_total):
    percent = num_extracted / num_total
    if callback: callback(status, percent, 0)

  image_files = [os.path.join(cfg.images.media_directory, file) for file in image_files]

  embeddings = ImageSearch.process_images(image_files, callback=send_emb_extracting_status).to('cpu')


  # Split to train and eval sets
  status = 'Step 2/2: Training the model...'
  print(status)
  X_train, X_test, y_train, y_test = train_test_split(embeddings, image_scores, test_size=0.1, random_state=42)

  print("X_train:", len(X_train), "X_test:", len(X_test))

  # Calculate the mean score of all train scores
  mean_score = np.mean(y_train)
  # Calculate baseline accuracy
  baseline_accuracy = 1 - np.mean(np.abs(mean_score - np.array(y_test)) / (np.array(y_test) + 1))

  # Create the model
  evaluator = src.scoring_models.Evaluator(embedding_dim=768, rate_classes=11)

  # Train the model
  best_train_accuracy = 0
  best_test_accuracy = 0
  best_epoch = 0
  total_epochs = 2001

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
      evaluator.save('./models/image_evaluator.pt')

  status = f'Best Epoch: {best_epoch}, Train Accuracy: {best_train_accuracy * 100:.2f}%, Test Accuracy: {best_test_accuracy * 100:.2f}%'
  print(status)
  if callback: 
    callback(status, 100, baseline_accuracy)