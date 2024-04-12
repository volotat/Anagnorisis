import pandas as pd
import numpy as np
import torch
import random

def load_data_from_csv(csv_file):
  df = pd.read_csv(csv_file)
  embeddings = np.array(df['embeddings'].apply(eval).tolist())
  reduced_embeddings = np.array(df['reduced_embeddings'].apply(eval).tolist())
  artists = df['artist'].tolist()
  titles = df['title'].tolist()
  return embeddings, reduced_embeddings, artists, titles

def set_seed(seed):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(seed)
  random.seed(seed)