import pandas as pd
import numpy as np
import torch
import random

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)

def load_data_from_csv(csv_file):
  df = pd.read_csv(csv_file)
  embeddings = np.array(df['embedding'].apply(eval).tolist())
  #reduced_embeddings = np.array(df['reduced_embeddings'].apply(eval).tolist())
  artists = df['artist'].tolist()
  titles = df['title'].tolist()
  scores = df['score'].tolist()
  return artists, titles, scores, embeddings

def calculate_metric(model, loader, rate_classes = 11):
  maes = [] 
  with torch.no_grad():
    for data in loader:
      inputs, labels = data
      outputs = model(inputs)

      # Calculate weighted average from all predictions to get the final prediction
      predicted = torch.softmax(outputs, dim=-1) * torch.arange(0, rate_classes)
      predicted = predicted.sum(dim=-1)

      maes.append(torch.mean(torch.abs(predicted - labels)).item())
  return np.mean(maes)