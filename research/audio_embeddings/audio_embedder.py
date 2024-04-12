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

class AudioEmbedder:
  _instance = None

  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      self.model = None
      self.processor = None
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
    mean_embedding = mean_embedding.to('cpu')

    return mean_embedding