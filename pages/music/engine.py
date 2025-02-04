from PIL import Image
import requests
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel, ClapProcessor
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import hashlib
import datetime
import io
import time
import cv2
import imageio
import torchaudio

import src.scoring_models
import pages.music.db_models as db_models
import pages.file_manager as file_manager

files_embeds_fast_cache = {}


class MusicSearch ():
  device = None
  model = None
  processor = None
  is_busy = False

  @staticmethod
  def initiate():
    if MusicSearch.model is not None:
      return
    
    MusicSearch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MusicSearch.model = ClapModel.from_pretrained("./models/clap-htsat-fused", local_files_only=True).to(MusicSearch.device)
    MusicSearch.processor = ClapProcessor.from_pretrained("./models/clap-htsat-fused", local_files_only=True)
    MusicSearch.feature_extractor = AutoFeatureExtractor.from_pretrained("./models/clap-htsat-fused", local_files_only=True)

    MusicSearch.model_hash = MusicSearch.get_model_hash()
    MusicSearch.embedding_dim = MusicSearch.model.config.text_config.projection_dim

    MusicSearch.cached_file_list = file_manager.CachedFileList('cache/music_file_list.pkl')
    MusicSearch.cached_file_hash = file_manager.CachedFileHash('cache/music_file_hash.pkl')

  '''_instance = None
  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      
    return self._instance
    
  def __init__(self) -> None:
    pass'''

  @staticmethod
  def get_model_hash():
    state_dict = MusicSearch.model.state_dict()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    model_hash = hashlib.md5(buffer.read()).hexdigest()
    return model_hash
  
  @staticmethod
  def read_audio(audio_path):
    # Implement audio reading and preprocessing here
    # For example, using torchaudio to read the audio file
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    return waveform, sample_rate

  @staticmethod
  def process_audio(audio_files, batch_size=32, callback=None, media_folder=None):
    MusicSearch.initiate()

    # Check if the MusicSearch is busy processing another request
    if MusicSearch.is_busy: 
      raise Exception("MusicSearch is busy")
    
    # Set the busy flag to prevent multiple calls
    MusicSearch.is_busy = True

    all_audio_embeds = []
    new_items = []

    for ind, audio_path in enumerate(tqdm(audio_files)):
      try:
        # Compute the hash of the audio file
        audio_hash = MusicSearch.cached_file_hash.get_file_hash(audio_path)
        
        # If the cache file exists, load the embeddings from it
        if audio_hash in files_embeds_fast_cache:
          audio_embeds = files_embeds_fast_cache[audio_hash]
        else:
          # Check if the embedding exists in the database
          audio_record = db_models.MusicLibrary.query.filter_by(hash=audio_hash).first()

          if audio_record and audio_record.embedding:
            # Load the embeddings from the database
            audio_embeds = pickle.loads(audio_record.embedding)
            # Save the embeddings to the fast cache (RAM)
            files_embeds_fast_cache[audio_hash] = audio_embeds
          else:
            # Process the audio and generate embeddings
            waveform, sample_rate = MusicSearch.read_audio(audio_path)
            
            if waveform is None: 
              raise Exception(f"Error reading audio file: {audio_path}")

            # convert waveform to 48kHz if not
            if sample_rate != 48000:
              waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(waveform)
              sample_rate = 48000

            # comvert to mono if not
            waveform = waveform.mean(dim=0, keepdim=False)
            
            # Use the waveform directly with the processor
            #inputs_audio = MusicSearch.processor(audios=[waveform], sampling_rate=sample_rate, return_tensors="pt").to(MusicSearch.device)

            # Get the audio embeddings
            with torch.no_grad():
                inputs_audio = MusicSearch.feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(MusicSearch.device)
                outputs = MusicSearch.model.get_audio_features(**inputs_audio)

            # Get the audio embeddings
            audio_embeds = outputs

            # Save the embeddings to the database
            if audio_record:
              audio_record.embedding = pickle.dumps(audio_embeds)
              audio_record.embedder_hash = MusicSearch.model_hash
            else:
              # This require knowledge of the media folder to find the relative path
              if media_folder:
                audio_data = {
                  'hash': audio_hash,
                  'file_path': os.path.relpath(audio_path, media_folder),
                  'embedding': pickle.dumps(audio_embeds),
                  'embedder_hash': MusicSearch.model_hash
                }
                new_items.append(db_models.MusicLibrary(**audio_data))

            # Save the embeddings to the fast cache (RAM)
            files_embeds_fast_cache[audio_hash] = audio_embeds
      except Exception as e:
        print(f"Error processing audio: {audio_path}: {e}")
        audio_embeds = torch.zeros(1, MusicSearch.embedding_dim).to(MusicSearch.device)

      all_audio_embeds.append(audio_embeds)
      if callback: callback(ind + 1, len(audio_files))

    # Save the embeddings to the database
    if new_items: db_models.db.session.bulk_save_objects(new_items)
    db_models.db.session.commit()

    # Concatenate all embeddings
    all_audio_embeds = torch.cat(all_audio_embeds, dim=0)

    # Reset the busy flag
    MusicSearch.is_busy = False

    return all_audio_embeds
  
  @staticmethod
  def process_text(text):
    MusicSearch.initiate()
    
    inputs_text = MusicSearch.processor(text=text, padding=True, return_tensors="pt").to(MusicSearch.device)

    with torch.no_grad():
      outputs = MusicSearch.model.get_text_features(**inputs_text)

    # get the text embeddings
    text_embeds = outputs
    
    return text_embeds

  @staticmethod
  def compare(embeds_audio, embeds_text):
    print(embeds_audio.shape, embeds_text.shape)  
    '''
    logits_per_text = torch.matmul(embeds_text, embeds_audio.t()) * MusicSearch.model.logit_scale.exp() + MusicSearch.model.logit_bias
    logits_per_audio = logits_per_text.t()

    probs = torch.sigmoid(logits_per_audio)
    '''

    # cosine similarity as logits
    logit_scale_text = MusicSearch.model.logit_scale_t.exp()
    logit_scale_audio = MusicSearch.model.logit_scale_a.exp()
    logits_per_text = torch.matmul(embeds_text, embeds_audio.t()) * logit_scale_text
    logits_per_audio = torch.matmul(embeds_audio, embeds_text.t()) * logit_scale_audio

    logits_avg = (logits_per_text + logits_per_audio.t()) / 2.0

    probs = torch.sigmoid(logits_avg)[0]
    return probs.cpu().detach().numpy() 

# Create scoring model singleton class so it easily accessible from other modules
class MusicEvaluator(src.scoring_models.Evaluator):
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(MusicEvaluator, cls).__new__(cls)
    return cls._instance

  def __init__(self, embedding_dim=768, rate_classes=11):
    if not hasattr(self, '_initialized'):
      super(MusicEvaluator, self).__init__(embedding_dim, rate_classes)
      self._initialized = True