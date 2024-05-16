from tqdm import tqdm
import numpy as np
import pages.music.db_models as db_models
import src.scoring_models
from sklearn.model_selection import train_test_split

def train_audio_evaluator(callback=None):
  # Create dataset from DB, select only music with user rating
  music_library_entries = db_models.MusicLibrary.query.filter(db_models.MusicLibrary.user_rating != None).all()

  # filter all non-mp3 files
  music_library_entries = [entry for entry in music_library_entries if entry.file_path.endswith('.mp3')]

  music_files = [entry.file_path for entry in music_library_entries]
  music_scores = [entry.user_rating for entry in music_library_entries]

  # get embeddings for that music
  embedder = src.scoring_models.AudioEmbedder(audio_embedder_model_path = "./models/MERT-v1-95M")

  status = "Embedding music files..."
  print(status)
  embeddings = []
  for i, file_path in enumerate(tqdm(music_files)):
    embedding = embedder.embed_audio(file_path)
    #print(embedding.shape)
    embeddings.append(embedding)

    if callback:
      percent = (i+1) / len(music_files)
      callback(status, percent)

  # Split to train and eval sets
  status = 'Training the model...'
  print(status)
  X_train, X_test, y_train, y_test = train_test_split(embeddings, music_scores, test_size=0.1, random_state=42)

  print("X_train:", len(X_train), "X_test:", len(X_test))

  # Create the model
  evaluator = src.scoring_models.Evaluator(embedding_dim=embedder.embedding_dim, rate_classes=11)

  # Train the model
  best_train_accuracy = 0
  best_test_accuracy = 0
  best_epoch = 0
  total_epochs = 301

  # Initialize the progress bar
  pbar = tqdm(range(total_epochs))

  for epoch in pbar:
    # Train the model
    train_accuracy, test_accuracy = evaluator.train(X_train, y_train, X_test, y_test, batch_size=64)

    # Update the progress bar description
    pbar.set_description(f'Epoch: {epoch+1}, Train Metric: {train_accuracy * 100:.2f}%, Test Metric: {test_accuracy * 100:.2f}%')

    if callback:
      percent = (epoch+1) / total_epochs
      callback(status, percent, train_accuracy, test_accuracy)

    # Check if this epoch's accuracy is the best
    if test_accuracy > best_test_accuracy:
      best_train_accuracy = train_accuracy
      best_test_accuracy = test_accuracy
      best_epoch = epoch + 1

      # Save the model
      evaluator.save('./models/audio_evaluator.pt')

  status = f'Best Epoch: {best_epoch}, Train Accuracy: {best_train_accuracy * 100:.2f}%, Test Accuracy: {best_test_accuracy * 100:.2f}%'
  print(status)
  callback(status, 100)