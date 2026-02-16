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

from src.model_manager import ModelManager


# Create scoring model class
class Evaluator():
  def __init__(self, embedding_dim=128, rate_classes=11, name="Evaluator"):
    self.embedding_dim = embedding_dim
    self.rate_classes = rate_classes
    self.hash = None
    self.mape_bias = 2
    self.name = name

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

    self._model = Net().cpu()  # Initialize on CPU
    # Tag the module instance; used by logs and repr
    setattr(self._model, "_name", self.name)

    self.model = ModelManager(self._model, device=self.device)
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters())

    # Force unload after wrapping as this usually cause the model to be loaded to GPU
    self.model.unload_model()

  # Optional helper to rename after creation
  def set_name(self, name: str):
    self.name = name
    setattr(self._model, "_name", name)

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


# ---------------------------------------------------------------------------
# Transformer-based evaluator for variable-length embedding sequences
# (e.g. text files represented as a list of chunk embeddings)
# ---------------------------------------------------------------------------

import math

class TransformerEvaluator:
  """
  Lightweight transformer that ingests a *sequence* of embedding vectors
  (one per chunk) and produces a scalar rating.

  Architecture
  ------------
  1.  Linear projection: embedding_dim → d_model
  2.  Learnable [CLS] token prepended to the sequence
  3.  Sinusoidal positional encoding (capped at max_seq_len)
  4.  N TransformerEncoder layers
  5.  [CLS] output → small MLP head → rate_classes logits
  6.  Weighted-softmax → scalar prediction (same convention as Evaluator)
  """

  def __init__(
      self,
      embedding_dim: int = 1024,
      rate_classes: int = 11,
      d_model: int = 1024,
      nhead: int = 4,
      num_layers: int = 2,
      dim_feedforward: int = 512,
      max_seq_len: int = 512,
      dropout: float = 0.1,
      name: str = "TransformerEvaluator",
  ):
    self.embedding_dim = embedding_dim
    self.rate_classes = rate_classes
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    self.hash = None
    self.mape_bias = 2
    self.name = name

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- inner network ----
    class TransformerNet(nn.Module):
      def __init__(inner_self):
        super().__init__()
        # Project high-dim embeddings down to d_model
        inner_self.input_proj = nn.Linear(embedding_dim, d_model)

        # Learnable [CLS] token
        inner_self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Sinusoidal positional encoding buffer (max_seq_len + 1 for CLS)
        pe = inner_self._sinusoidal_pe(max_seq_len + 1, d_model)
        inner_self.register_buffer("pe", pe)  # [1, max_seq_len+1, d_model]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        inner_self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # MLP classification head
        inner_self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, rate_classes),
        )

      @staticmethod
      def _sinusoidal_pe(length: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(length).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, length, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

      def forward(inner_self, x: torch.Tensor, padding_mask: torch.Tensor = None):
        """
        Parameters
        ----------
        x : Tensor [B, S, embedding_dim]
        padding_mask : BoolTensor [B, S]  True = padded (ignored) position

        Returns
        -------
        logits : Tensor [B, rate_classes]
        """
        B, S, _ = x.shape
        x = inner_self.input_proj(x)                         # [B, S, d_model]

        # Prepend [CLS]
        cls_tokens = inner_self.cls_token.expand(B, -1, -1)  # [B, 1, d_model]
        x = torch.cat([cls_tokens, x], dim=1)                # [B, S+1, d_model]

        # Positional encoding (truncate if sequence is shorter than buffer)
        x = x + inner_self.pe[:, :S + 1, :]

        # Extend padding mask for the CLS position (never masked)
        if padding_mask is not None:
          cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
          padding_mask = torch.cat([cls_mask, padding_mask], dim=1)  # [B, S+1]

        x = inner_self.encoder(x, src_key_padding_mask=padding_mask)

        # Take [CLS] output
        cls_out = x[:, 0, :]  # [B, d_model]
        return inner_self.head(cls_out)

    self._model = TransformerNet().cpu()
    setattr(self._model, "_name", self.name)

    self.model = ModelManager(self._model, device=self.device)
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)

    # Force unload after wrapping
    self.model.unload_model()

  # --------------- helpers for variable-length batching ---------------

  @staticmethod
  def _collate_variable_length(batch_embeddings, batch_labels, max_seq_len):
    """
    Pad a list of variable-length np.ndarray / Tensor items into a batch.

    Parameters
    ----------
    batch_embeddings : list of np.ndarray  each [S_i, D]
    batch_labels     : list / array of scalars
    max_seq_len      : int – hard cap on sequence length

    Returns
    -------
    padded : Tensor  [B, S_max, D]
    mask   : BoolTensor  [B, S_max]   True = padding
    labels : Tensor  [B]
    """
    processed = []
    for emb in batch_embeddings:
      if isinstance(emb, torch.Tensor):
        emb = emb.numpy() if emb.device.type == 'cpu' else emb.cpu().numpy()
      if isinstance(emb, list):
        emb = np.array(emb, dtype=np.float32)
      # Truncate long sequences
      if len(emb) > max_seq_len:
        emb = emb[:max_seq_len]
      processed.append(emb)

    lengths = [len(e) for e in processed]
    S_max = max(lengths)
    D = processed[0].shape[-1]
    B = len(processed)

    padded = np.zeros((B, S_max, D), dtype=np.float32)
    mask = np.ones((B, S_max), dtype=bool)  # True = pad

    for i, emb in enumerate(processed):
      L = len(emb)
      padded[i, :L, :] = emb
      mask[i, :L] = False

    labels_t = torch.tensor(batch_labels, dtype=torch.float32)
    return torch.from_numpy(padded), torch.from_numpy(mask), labels_t

  # --------------- public API (mirrors Evaluator) ---------------

  def set_name(self, name: str):
    self.name = name
    setattr(self._model, "_name", name)

  def _run_epoch(self, X, y, batch_size, train_mode=True):
    """Run one pass over data, return list of (predicted, label) pairs."""
    indices = list(range(len(X)))
    if train_mode:
      import random
      random.shuffle(indices)

    all_preds = []
    all_labels = []

    if train_mode:
      self.model.train()
    else:
      self.model.eval()

    for start in range(0, len(indices), batch_size):
      batch_idx = indices[start:start + batch_size]
      batch_emb = [X[i] for i in batch_idx]
      batch_lbl = [y[i] for i in batch_idx]

      padded, mask, labels = self._collate_variable_length(batch_emb, batch_lbl, self.max_seq_len)
      padded = padded.to(self.device)
      mask = mask.to(self.device)
      labels = labels.to(self.device)

      if train_mode:
        self.optimizer.zero_grad()
        logits = self.model(padded, padding_mask=mask)
        predicted = (torch.softmax(logits, dim=-1) * torch.arange(self.rate_classes, device=self.device)).sum(dim=-1)
        loss = F.mse_loss(predicted, labels)
        loss.backward()
        self.optimizer.step()
      else:
        with torch.no_grad():
          logits = self.model(padded, padding_mask=mask)
          predicted = (torch.softmax(logits, dim=-1) * torch.arange(self.rate_classes, device=self.device)).sum(dim=-1)

      all_preds.extend(predicted.detach().cpu().tolist())
      all_labels.extend(labels.detach().cpu().tolist())

    return all_preds, all_labels

  def _accuracy_from_preds(self, preds, labels):
    preds = np.array(preds)
    labels = np.array(labels)
    mape = np.mean(np.abs(preds - labels) / (labels + 1.0))
    return 1.0 / (1.0 + mape)

  def train(self, X_train, y_train, X_test, y_test, batch_size=16):
    """
    One epoch of training + evaluation.

    Parameters
    ----------
    X_train : list of np.ndarray  – each element [S_i, D]
    y_train : list of int/float ratings
    X_test  : list of np.ndarray
    y_test  : list of int/float ratings
    batch_size : int

    Returns
    -------
    (train_accuracy, test_accuracy)
    """
    # Training pass
    train_preds, train_labels = self._run_epoch(X_train, y_train, batch_size, train_mode=True)
    train_acc = self._accuracy_from_preds(train_preds, train_labels)

    # Eval pass
    test_preds, test_labels = self._run_epoch(X_test, y_test, batch_size, train_mode=False)
    test_acc = self._accuracy_from_preds(test_preds, test_labels)

    return train_acc, test_acc

  def predict(self, X):
    """
    Predict ratings for a list of variable-length embedding sequences.

    Parameters
    ----------
    X : list of (list of np.ndarray) or list of np.ndarray
        Each element represents one file's chunk embeddings: [S_i, D]

    Returns
    -------
    np.ndarray of shape [len(X)] with predicted ratings
    """
    self.model.eval()
    all_preds = []

    with torch.no_grad():
      for file_embs in X:
        if file_embs is None or (isinstance(file_embs, (list, np.ndarray)) and len(file_embs) == 0):
          all_preds.append(0.0)
          continue

        # Convert to np array if list of arrays
        if isinstance(file_embs, list):
          file_embs = np.array(file_embs, dtype=np.float32)
        if isinstance(file_embs, torch.Tensor):
          file_embs = file_embs.cpu().numpy()

        # Truncate
        if len(file_embs) > self.max_seq_len:
          file_embs = file_embs[:self.max_seq_len]

        # Single-sample batch
        x = torch.from_numpy(file_embs).unsqueeze(0).to(self.device)  # [1, S, D]
        logits = self.model(x, padding_mask=None)
        predicted = (torch.softmax(logits, dim=-1) * torch.arange(self.rate_classes, device=self.device)).sum(dim=-1)
        all_preds.append(predicted.item())

    return np.array(all_preds)

  def load(self, model_path):
    if os.path.exists(model_path):
      self.model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
      self.save(model_path)

    with open(model_path, "rb") as f:
      data = f.read()
      self.hash = hashlib.md5(data).hexdigest()

  def save(self, model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(self.model.state_dict(), model_path)

  def reinitialize(self):
    for m in self.model.modules():
      if isinstance(m, (nn.Linear, nn.LayerNorm)):
        if hasattr(m, 'reset_parameters'):
          m.reset_parameters()
    # Re-init [CLS] token
    if hasattr(self._model, 'cls_token'):
      nn.init.normal_(self._model.cls_token, std=0.02)
    self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-2)