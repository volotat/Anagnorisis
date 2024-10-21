import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm import tqdm


from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
from datasets import load_dataset
import numpy as np
import time

import io
import torchaudio
from pydub import AudioSegment
import gc
import os
import hashlib

import torch.nn.functional as F

class AudioEmbedder:
  _instance = None

  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      self.model = None
      self.processor = None
      self.embedding_dim = 768
      #self.load_model(self)

      self.sampling_rate = 24000 # 24kHz is the REQUERED sampling rate for the MERT model used in this module
      self.context_window_seconds = 5 # MERT model is trained on 5 second tracks

      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      
    return self._instance
  
  def __init__(self, *args, **kwargs) -> None:
    self.load_model( *args, **kwargs  )
    pass

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    self.unload_model()

  def load_model(self, audio_embedder_model_path = "./models/MERT-v1-95M"):
    print('Loading model:', audio_embedder_model_path)
    # loading our model weights from https://huggingface.co/m-a-p/MERT-v1-95M
    self.model = AutoModel.from_pretrained(audio_embedder_model_path, local_files_only=True, trust_remote_code=True).to(self.device)

    # loading the corresponding preprocessor config
    self.processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_embedder_model_path, local_files_only=True)

    # set the model to evaluation mode (just to be sure)
    self.model.eval()

  def unload_model(self):
    self.model = None
    self.processor = None
    gc.collect()
    torch.cuda.empty_cache()


  # converts mp3 file into a waveform
  def mp3_to_waveform(self, mp3_file_path):
    audio = AudioSegment.from_mp3(mp3_file_path)
    audio = audio.set_frame_rate(self.sampling_rate)
    file_handle = io.BytesIO()
    audio.export(file_handle, format='wav')
    file_handle.seek(0)
    waveform, _ = torchaudio.load(file_handle)
    waveform = waveform.mean(dim=0)
    return waveform
  
  def get_time_reduced_embedding(self, outputs):
    time_reduced_mean = outputs.mean(-2)
    return time_reduced_mean
  #  print('last_hidden_state:', last_hidden_state.shape)
  #
  #  # find indexes of values that are the most distant from their mean on the time axis
  #  indexes = np.argmax(np.abs(last_hidden_state - time_reduced_mean), axis=-2)
  #
  #  # take only the most important values as a final embedding
  #  time_reduced_hidden_state = last_hidden_state[indexes, np.arange(len(indexes))]
  #
  #  return time_reduced_hidden_state

  def embed_audio(self, audio_path, embedder_sampling_points = 5):
    full_waveform = self.mp3_to_waveform(audio_path)

    window_size = self.sampling_rate * self.context_window_seconds

    # Generate random start positions for each part
    start_poses = np.random.choice(len(full_waveform) - window_size, size=embedder_sampling_points)

    # Extract the waveform for each part
    part_waveforms = [full_waveform[start:start+window_size] for start in start_poses]

    # Stack the part waveforms into a batch
    batch_waveform = torch.stack(part_waveforms).to(self.device)

    # Reshape the batch waveform to be 3D
    batch_waveform = batch_waveform.view(-1, batch_waveform.shape[-1])

    # Process the batch waveform
    inputs = self.processor(batch_waveform, sampling_rate=self.sampling_rate, return_tensors="pt", padding=True)

    # Move the inputs to the GPU
    inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
    
    # Squeeze the leading dimension from the input_values tensor and its attention mask
    inputs["input_values"] = inputs["input_values"].squeeze(0)
    inputs["attention_mask"] = inputs["attention_mask"].squeeze(0)

    # Calculate the embeddings for the batch
    with torch.no_grad():
      outputs = self.model(**inputs)

    # Get time-reduced embeddings for each part
    sampled_embeddings = self.get_time_reduced_embedding(outputs.last_hidden_state)
    mean_embedding = sampled_embeddings.mean(0)

    # Move the mean_embedding back to the CPU
    mean_embedding = mean_embedding.cpu().detach().numpy()

    return mean_embedding

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
    
    train_accuracy = self.calculate_accuracy(train_loader)
    test_accuracy = self.calculate_accuracy(test_loader)

    return train_accuracy, test_accuracy

  def load(self, model_path):
    # load the model from folder
    if os.path.exists(model_path):
      self.model.load_state_dict(torch.load(model_path))

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