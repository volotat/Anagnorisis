import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import torch
from torch import nn
import numpy as np

import os
import hashlib

import torch.nn.functional as F


# Create scoring model class
class Evaluator():
  def __init__(self, embedding_dim=128, rate_classes=11):
    self.embedding_dim = embedding_dim
    self.rate_classes = rate_classes
    self.hash = None
    self.mape_bias = 2

    # Set the device
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    self.model = Net().to(self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters())

  # Not used in current implementation
  def calculate_metric(self, loader):
    maes = [] 
    with torch.no_grad():
      for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)

        # Calculate weighted average from all predictions to get the final prediction
        predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes, device=self.device)
        predicted = predicted.sum(dim=-1)

        maes.append(torch.mean(torch.abs(predicted - labels)).item())
    return np.mean(maes)
  
  # Not used in current implementation
  def calculate_accuracy(self, loader):
    accuracy_list = [] 
    with torch.no_grad():
      for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)

        # Calculate weighted average from all predictions to get the final prediction
        predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes, device=self.device)
        predicted = predicted.sum(dim=-1)

        # Calculate MAPE (Mean absolute percentage error) while increasing scores by 1 to avoid division by zero
        mape = torch.mean(torch.abs(predicted - labels) / (labels + self.mape_bias)).item()
        # Invert the MAPE to get the accuracy
        accuracy_list.append(1 - mape)
    return np.mean(accuracy_list)
  
  def calculate_accuracy_2(self, loader):
    accuracy_list = [] 
    with torch.no_grad():
      for data in loader:
        inputs, labels = data
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = self.model(inputs)

        # Calculate weighted average from all predictions to get the final prediction
        predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes, device=self.device)
        predicted = predicted.sum(dim=-1)

        # Calculate MAPE (Mean absolute percentage error) while increasing scores by 1 to avoid division by zero
        mape = torch.mean(torch.abs(predicted - labels) / (labels + 1)).item()
        # Alternative metric: transform MAPE to an accuracy in [0, 1]
        accuracy_list.append(1 / (1 + mape))
    return np.mean(accuracy_list)

  def train(self, X_train, y_train, X_test, y_test, batch_size=32):
    #le = LabelEncoder()
    #artists = le.fit_transform(y_train + y_test)

    #print(type(X_train), X_train[:3][:10])

    # Convert data to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # Convert to one-hot
    #y_train_one_hot = F.one_hot(y_train, num_classes=self.rate_classes)
    #y_test_one_hot = F.one_hot(y_test, num_classes=self.rate_classes)

    # Create data loaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    for i, data in enumerate(train_loader, 0):
      inputs, labels = data
      inputs, labels = inputs.to(self.device), labels.to(self.device)
      labels = labels.float()
      self.optimizer.zero_grad()
      outputs = self.model(inputs)
      #loss = self.criterion(outputs, labels)

      predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes, device=self.device)
      predicted = predicted.sum(dim=-1)
      loss = F.mse_loss(predicted, labels)

      
      loss.backward()
      self.optimizer.step()
    
    train_accuracy = self.calculate_accuracy_2(train_loader)
    test_accuracy = self.calculate_accuracy_2(test_loader)

    return train_accuracy, test_accuracy

  def load(self, model_path):
    # load the model from folder
    if os.path.exists(model_path):
      self.model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
      self.save(model_path)

    # If not in cache or file has been modified, calculate the hash
    with open(model_path, "rb") as f:
      bytes = f.read()  # Read the entire file as bytes
      self.hash = hashlib.md5(bytes).hexdigest()

  def save(self, model_path):
    # save the model to the folder
    torch.save(self.model.state_dict(), model_path)

  def predict(self, X):
    if not isinstance(X, torch.Tensor):
      X = torch.tensor(X, dtype=torch.float32).to(self.device)
      
    self.model.eval()
    with torch.no_grad():
      outputs = self.model(X)

      # Calculate weighted average from all predictions to get the final prediction
      predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, self.rate_classes, device=self.device)
      predicted = predicted.sum(dim=-1)


      #_, predicted = torch.max(outputs, dim=1)  # Get the index of the max log-probability
    return predicted.cpu().detach().numpy()
  
  # reinitialize the model weights to random values for training from scratch
  def reinitialize(self):
    # Reinitialize the model weights
    for m in self.model.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()
    
    # Reinitialize the optimizer
    self.optimizer = torch.optim.Adam(self.model.parameters())
    
    # Reinitialize the loss function (if necessary)
    self.criterion = nn.CrossEntropyLoss()