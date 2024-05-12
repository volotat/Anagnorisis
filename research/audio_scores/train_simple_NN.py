import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm
import random
import pandas as pd

import torch.nn.functional as F
import utils

BATCH_SIZE = 32
embedding_dim = 768
rate_classes = 11




utils.set_seed(42)

# Load dataset
artists, titles, scores, embeddings = utils.load_data_from_csv("audio_scores_dataset.csv")
print("Dataset has been loaded")


'''# Create normalized, truncated and quantized reduced embeddings that will simulate the LLaMA model's input
mean = np.mean(embeddings, axis=0)
var = np.var(embeddings, axis=0)
normalized_reduced_embeddings = (embeddings - mean) / np.sqrt(var)

min_allowed_value = -1
max_allowed_value = 1
truncated_reduced_embeddings = np.clip(normalized_reduced_embeddings, min_allowed_value, max_allowed_value)
truncated_reduced_embeddings = (truncated_reduced_embeddings - min_allowed_value) / (max_allowed_value - min_allowed_value) # normalize to [0, 1]
granularity = 146
quantized_reduced_embeddings = (truncated_reduced_embeddings * granularity + 0.5).astype(int) 
quantized_reduced_embeddings = (quantized_reduced_embeddings.astype(np.float32) / granularity - 0.5) * 2 # results are in the range of [-1, 1]'''



# Split to train and eval sets
print('Training the model...')
X_train, X_test, y_train, y_test = train_test_split(embeddings, scores, test_size=0.15, random_state=42)

print("X_train:", len(X_train), "X_test:", len(X_test))

# Convert data to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create data loaders
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE)
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


def train_and_evaluate(num_epochs):
  # Define the network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(embedding_dim, 768)
      self.fc2 = nn.Linear(768, 256)
      self.fc3 = nn.Linear(256, rate_classes)

    def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = torch.relu(self.fc2(x))
      x = self.fc3(x)
      return x

  model = Net()
  criterion = nn.CrossEntropyLoss()

  # Train the model
  best_train_metric = np.inf
  best_test_metric = np.inf
  best_epoch = 0

  optimizer = torch.optim.Adam(model.parameters())

  # Initialize the progress bar
  pbar = tqdm(range(num_epochs))

  for epoch in pbar:
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      labels = labels.float()
      optimizer.zero_grad()
      outputs = model(inputs)
      predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, rate_classes)
      predicted = predicted.sum(dim=-1)

      loss = F.l1_loss(predicted, labels)

      #loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
    
    train_metric = utils.calculate_metric(model, train_loader, 11)
    test_metric = utils.calculate_metric(model, test_loader, 11)

    # Update the progress bar description
    pbar.set_description(f'Epoch: {epoch+1}, Train Metric: {train_metric:.2f}, Test Metric: {test_metric:.2f}')

    # Check if this epoch's accuracy is the best
    if test_metric < best_test_metric:
      best_train_metric = train_metric
      best_test_metric = test_metric
      best_epoch = epoch + 1

  print(f'Best Epoch: {best_epoch}, Train Metric: {best_train_metric:.4f}, Test Metric: {best_test_metric:.4f}')

train_and_evaluate(1000)





