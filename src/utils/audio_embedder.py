from transformers import Wav2Vec2FeatureExtractor
from transformers import AutoModel
import torch
from torch import nn
from datasets import load_dataset
import numpy as np
import time


# loading our model weights from https://huggingface.co/m-a-p/MERT-v1-95M
model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# loading the corresponding preprocessor config
processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
# set the model to evaluation mode (just to be sure)
model.eval()



sampling_rate = 24000 # 24kHz is the REQUERED sampling rate for the MERT model used in this module
context_window_seconds = 5 # MERT model is trained on 5 second tracks

# converts mp3 file into a waveform
def mp3_to_waveform(mp3_file_path):
  audio = AudioSegment.from_mp3(mp3_file_path)
  audio = audio.set_frame_rate(sampling_rate)
  file_handle = io.BytesIO()
  audio.export(file_handle, format='wav')
  file_handle.seek(0)
  waveform, _ = torchaudio.load(file_handle)
  waveform = waveform.mean(dim=0)
  return waveform

def get_time_reduced_embedding(outputs):
  last_hidden_state = outputs.last_hidden_state.squeeze()
  time_reduced_mean = last_hidden_state.mean(-2)
  print('last_hidden_state:', last_hidden_state.shape)
  # find indexes of values that are the most distant from their mean on the time axis
  indexes = np.argmax(np.abs(last_hidden_state - time_reduced_mean), axis=-2)
  # take only the most important values as a final embedding
  time_reduced_hidden_state = last_hidden_state[indexes, np.arange(len(indexes))]

  return time_reduced_hidden_state

def calculate_cosine_similarity(embedding_1, embedding_2):
  cosine_similarity = nn.CosineSimilarity(dim=0)
  return cosine_similarity(embedding_1, embedding_2).item()

def convert_vector_to_letters(weighted_avg_hidden_states, min_allowed_value = -9.5, max_allowed_value = 4.5, granularity = 11):
  # list of characters that by itself generate only a single token in the llama model, special characters are not included
  charset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ë', 'ë', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ъ', 'ъ', 'Ы', 'ы', 'Ь', 'ь', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', 'ә', 'Ђ', 'ђ', 'Đ', 'đ', 'Љ', 'љ', 'Њ', 'њ', 'Ћ', 'ћ', 'Ć', 'ć', 'Č', 'č', 'Џ', 'џ', 'Š', 'š']

  weighted_avg_hidden_states = weighted_avg_hidden_states.detach().numpy()
  print('Extreams: ', np.amin(weighted_avg_hidden_states), np.amax(weighted_avg_hidden_states))

  if (np.amin(weighted_avg_hidden_states) < min_allowed_value):
    raise ValueError(f'Unsupported value in the embedding: {np.amin(weighted_avg_hidden_states)} < -3 !')
  
  if (np.amax(weighted_avg_hidden_states) > max_allowed_value):
    raise ValueError(f'Unsupported value in the embedding: {np.amax(weighted_avg_hidden_states)} > 3 !')

  weighted_avg_hidden_states = (weighted_avg_hidden_states - min_allowed_value) / (max_allowed_value - min_allowed_value) # normalize to [0, 1]
  gsqrt = int(granularity)
  print('gsqrt:', gsqrt, 'char count:', len(charset), 'max:', gsqrt * (gsqrt+1) + gsqrt)
  print('shape:', weighted_avg_hidden_states.shape, weighted_avg_hidden_states[::2].shape, weighted_avg_hidden_states[1::2].shape)

  states = (weighted_avg_hidden_states[::2] * gsqrt).astype(int) * (gsqrt + 1) + (weighted_avg_hidden_states[1::2] * gsqrt).astype(int)
  print('Extreams states: ', np.amin(states), np.amax(states))
  weighted_avg_hidden_states = (weighted_avg_hidden_states * granularity).astype(int) # quantize to [0, 11]
  letter_representation = ''.join([charset[i] for i in states])

  return letter_representation

def embed_audio(audio_path):
  full_waveform = mp3_to_waveform(audio_path)

  window_size = sampling_rate * context_window_seconds

  start_pos = 0 #? Random start position
  end_pos = start_pos+window_size

  # Extract the waveform for the selected parts
  part_waveform = full_waveform[start_pos:end_pos]

  # Process the waveforms
  inputs_1 = processor(waveform_1, sampling_rate=sampling_rate, return_tensors="pt")

  # Calculate the embeddings for the selected parts
  with torch.no_grad():
    outputs_1 = model(**inputs_1) 

  weighted_avg_hidden_states_1 = get_time_reduced_embedding(outputs_1)

  letter_representation = convert_vector_to_letters(weighted_avg_hidden_states_1)



#use example
#with AudioEmbedder() as audio_embedder:
#  vector, string = audio_embedder.embed_audio("path/to/audio.wav")