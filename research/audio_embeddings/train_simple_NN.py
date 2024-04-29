import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import utils
from tqdm import tqdm

utils.set_seed(42)

# Assume embeddings is a list of embeddings and artists is a list of corresponding artists
embeddings, reduced_embeddings, artists, titles = utils.load_data_from_csv("dataset.csv")
print("Dataset has been loaded")

# Create normalized, truncated and quantized reduced embeddings that will simulate the LLaMA model's input
mean = np.mean(reduced_embeddings, axis=0)
var = np.var(reduced_embeddings, axis=0)
normalized_reduced_embeddings = (reduced_embeddings - mean) / np.sqrt(var)

min_allowed_value = -1
max_allowed_value = 1
truncated_reduced_embeddings = np.clip(normalized_reduced_embeddings, min_allowed_value, max_allowed_value)
truncated_reduced_embeddings = (truncated_reduced_embeddings - min_allowed_value) / (max_allowed_value - min_allowed_value) # normalize to [0, 1]
granularity = 146
quantized_reduced_embeddings = (truncated_reduced_embeddings * granularity).astype(int) 
quantized_reduced_embeddings = (quantized_reduced_embeddings.astype(np.float32) / granularity - 0.5) * 2 # results are in the range of [-1, 1]

# Convert artists to numerical labels
le = LabelEncoder()
artists = le.fit_transform(artists)


def calculate_accuracy(loader, model):
  correct = 0
  total = 0
  with torch.no_grad():
    for data in loader:
      inputs, labels = data
      outputs = model(inputs)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  return 100 * correct / total

def train_and_evaluate(X_train, X_test, y_train, y_test, num_epochs=200):
  # Convert data to tensors
  X_train = torch.tensor(X_train, dtype=torch.float32)
  y_train = torch.tensor(y_train, dtype=torch.long)
  X_test = torch.tensor(X_test, dtype=torch.float32)
  y_test = torch.tensor(y_test, dtype=torch.long)

  # Create data loaders
  train_data = TensorDataset(X_train, y_train)
  train_loader = DataLoader(train_data, batch_size=32)
  test_data = TensorDataset(X_test, y_test)
  test_loader = DataLoader(test_data, batch_size=32)

  # Define the network
  class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()
      self.fc1 = nn.Linear(X_train.shape[1], 128)
      self.fc2 = nn.Linear(128, len(le.classes_))

    def forward(self, x):
      x = torch.relu(self.fc1(x))
      x = self.fc2(x)
      return x

  # Train the network
  model = Net()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  # Initialize best accuracy and best epoch
  best_train_accuracy = 0
  best_test_accuracy = 0
  best_epoch = 0

  # Initialize the progress bar
  pbar = tqdm(range(num_epochs))

  for epoch in pbar:  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

    train_accuracy = calculate_accuracy(train_loader, model)
    test_accuracy = calculate_accuracy(test_loader, model)

    # Update the progress bar description
    pbar.set_description(f'Epoch: {epoch+1}, Train Accuracy: {train_accuracy:.2f}, Test Accuracy: {test_accuracy:.2f}')
    #if (epoch + 1) % 10 == 0:
    #  print(f'Epoch {epoch+1}, Train Accuracy: {train_accuracy}%, Test Accuracy: {test_accuracy}%')

    # Check if this epoch's accuracy is the best
    if test_accuracy > best_test_accuracy:
      best_train_accuracy = train_accuracy
      best_test_accuracy = test_accuracy
      best_epoch = epoch + 1

  return best_epoch, best_train_accuracy, best_test_accuracy

# Split the data into a training set and a test set for full embeddings
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(embeddings, artists, test_size=0.2, random_state=42)

# Train the NN on full embeddings and calculate accuracy
best_epoch_full, train_accuracy_full, test_accuracy_full = train_and_evaluate(X_train_full, X_test_full, y_train_full, y_test_full)
print('Finished Training on full embeddings \n')

# Split the data into a training set and a test set for reduced embeddings
X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_embeddings, artists, test_size=0.2, random_state=42)

# Train the NN on reduced embeddings and calculate accuracy
best_epoch_reduced, train_accuracy_reduced, test_accuracy_reduced = train_and_evaluate(X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced)
print('Finished Training on reduced embeddings \n')

# Split the data into a training set and a test set for quantized reduced embeddings
X_train_quantized, X_test_quantized, y_train_quantized, y_test_quantized = train_test_split(quantized_reduced_embeddings, artists, test_size=0.2, random_state=42)

# Train the NN on reduced embeddings and calculate accuracy
best_epoch_quantized, train_accuracy_quantized, test_accuracy_quantized = train_and_evaluate(X_train_quantized, X_test_quantized, y_train_quantized, y_test_quantized)
print('Finished Training on quantized reduced embeddings \n')

# Train the NN on reduced embeddings and calculate accuracy
_, train_accuracy_quantized_1, test_accuracy_quantized_1 = train_and_evaluate(X_train_quantized, X_test_quantized, y_train_quantized, y_test_quantized, num_epochs=1)
print('Finished Training on quantized reduced embeddings \n')

print(f'Best Epoch on full embeddings: {best_epoch_full}')
print(f'Train Accuracy: {train_accuracy_full :.2f}%, Test Accuracy: {test_accuracy_full :.2f}%')
print(f'Best Epoch on reduced embeddings: {best_epoch_reduced}')
print(f'Train Accuracy: {train_accuracy_reduced :.2f}%, Test Accuracy: {test_accuracy_reduced :.2f}%')
print(f'Best Epoch on quantized reduced embeddings: {best_epoch_quantized}')
print(f'Train Accuracy: {train_accuracy_quantized :.2f}%, Test Accuracy: {test_accuracy_quantized :.2f}%')
print(f'Results on single epoch training with quantized reduced embeddings: ')
print(f'Train Accuracy: {train_accuracy_quantized_1 :.2f}%, Test Accuracy: {test_accuracy_quantized_1 :.2f}%')