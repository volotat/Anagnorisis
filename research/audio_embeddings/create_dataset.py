import os
import json
import pandas as pd
from tinytag import TinyTag
from audio_embedder import AudioEmbedder

def get_metadata(file_path):
  tag = TinyTag.get(file_path)
  return {"artist": tag.artist, "title": tag.title}

def get_embeddings(file_path, embedder):
  return {"embeddings": embedder.embed_audio(file_path).tolist()}

embedder = AudioEmbedder(audio_embedder_model_path = "../../models/MERT-v1-95M")

data = []
directory = "/media/alex/Expansion/Music/"
mp3_files = [os.path.join(root, name) for root, dirs, files in os.walk(directory) for name in files if name.endswith(".mp3")]
num_files = len(mp3_files)
ind = 0
for ind, filename in enumerate(mp3_files):
  if filename.endswith(".mp3"):
    file_path = os.path.join(directory, filename)
    metadata = get_metadata(file_path)
    if metadata["artist"] == None or metadata["title"] == None:
      continue
    print(f"{ind+1}/{num_files} Processing {metadata['artist']} - {metadata['title']}...")
    try:
      embeddings = get_embeddings(file_path, embedder)
      data.append({**metadata, **embeddings})
    except Exception as e:
      print(f"Failed to process {metadata['artist']} - {metadata['title']}: {e}")

df = pd.DataFrame(data)
df.to_csv("dataset.csv", index=False)

