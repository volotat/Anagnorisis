import numpy as np
from tqdm import tqdm
import os

import llama2_engine
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

utils.set_seed(42)

# print better represented numpy arrays
np.set_printoptions(precision=3, suppress=True)

def convert_vector_to_letters(embeddings, min_allowed_value = -1, max_allowed_value = 1):
  # list of characters that by itself generate only a single token in the llama model, special characters are not included
  charset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ë', 'ë', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ъ', 'ъ', 'Ы', 'ы', 'Ь', 'ь', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', 'ә', 'Ђ', 'ђ', 'Đ', 'đ', 'Љ', 'љ', 'Њ', 'њ', 'Ћ', 'ћ', 'Ć', 'ć', 'Č', 'č', 'Џ', 'џ', 'Š', 'š']
  granularity = len(charset) - 1

  embeddings = np.clip(embeddings, min_allowed_value, max_allowed_value)

  embeddings = (embeddings - min_allowed_value) / (max_allowed_value - min_allowed_value) # normalize to [0, 1]

  states = (embeddings * granularity).astype(int)
  letter_representation = ''.join([charset[i] for i in states])

  return letter_representation

def produce_prompt_score_from_song(reduced_embedding_str, artist, title, score): #Here is the list of possible bands: {", ".join(set(artists))}.
  prompt = f'''### HUMAN:
Rate this song on a scale from 0 to 10.
{artist} - {title}
[music]{reduced_embedding_str}[/music]

### RESPONSE:
Score:'''
  response = f" {score}/10;"
  return prompt, response

def produce_prompt_score_from_emb(reduced_embedding_str, artist, title, score):
  prompt = f'''### HUMAN:
Rate this song on a scale from 0 to 10.
[music]{reduced_embedding_str}[/music]

### RESPONSE:
Score:'''
  response = f" {score}/10;"
  return prompt, response

def produce_prompt_artist_from_emb(reduced_embedding_str, artist, title, score):
  prompt = f'''### HUMAN:
Who are playing this song?
[music]{reduced_embedding_str}[/music]

### RESPONSE:
Artist:'''
  response = f" {artist};"
  return prompt, response

def produce_prompt_title_from_emb(reduced_embedding_str, artist, title, score):
  prompt = f'''### HUMAN:
What is the title of this song?
[music]{reduced_embedding_str}[/music]

### RESPONSE:
Title:'''
  response = f" {title};"
  return prompt, response

def produce_prompt_song_name_from_emb(reduced_embedding_str, artist, title, score):
  prompt = f'''### HUMAN:
What the song is this?
[music]{reduced_embedding_str}[/music]

### RESPONSE:
Song:'''
  response = f" {artist} - {title};"
  return prompt, response

# Load dataset
artists, titles, scores, embeddings = utils.load_data_from_csv("audio_scores_dataset.csv")
print("Dataset has been loaded")

# Create a PCA object
pca = PCA(n_components=32)
reduced_embeddings = pca.fit_transform(embeddings)

# Calculate reduced embeddings statistics for proper normalization
print("Reduced embeddings shape:", reduced_embeddings.shape)  
mean = np.mean(reduced_embeddings, axis=0)
var = np.var(reduced_embeddings, axis=0)
reduced_embeddings = (reduced_embeddings - mean) / np.sqrt(var)

# Convert reduced embeddings into a string representation 
reduced_embeddings_str = [convert_vector_to_letters(emb) for emb in reduced_embeddings]

# Split to train and eval sets
print('Training the model...')
X_train, X_test, y_train, y_test = train_test_split(list(zip(reduced_embeddings_str, artists, titles)), scores, test_size=0.15, random_state=42)

print("X_train len:", len(X_train), "X_test len:", len(X_test))

import datasets

# Create dataset for fine-tuning the LLaMA model
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/eval", exist_ok=True)

file_ind = 0
for ind in tqdm(range(len(X_train))):
  reduced_embeddings_str, artists, titles = X_train[ind]
  score = y_train[ind]

  prompt, response = produce_prompt_score_from_song(reduced_embeddings_str, artists, titles, score)
  with open(f"dataset/train/{file_ind}.txt", "w") as f:
    f.write(prompt + response)
    file_ind+=1

  prompt, response = produce_prompt_score_from_emb(reduced_embeddings_str, artists, titles, score)
  with open(f"dataset/train/{file_ind}.txt", "w") as f:
    f.write(prompt + response)
    file_ind+=1

  prompt, response = produce_prompt_artist_from_emb(reduced_embeddings_str, artists, titles, score)
  with open(f"dataset/train/{file_ind}.txt", "w") as f:
    f.write(prompt + response)
    file_ind+=1

  prompt, response = produce_prompt_title_from_emb(reduced_embeddings_str, artists, titles, score)
  with open(f"dataset/train/{file_ind}.txt", "w") as f:
    f.write(prompt + response)
    file_ind+=1

  prompt, response = produce_prompt_song_name_from_emb(reduced_embeddings_str, artists, titles, score)
  with open(f"dataset/train/{file_ind}.txt", "w") as f:
    f.write(prompt + response)
    file_ind+=1

# Load the datasets
train_dataset = datasets.load_dataset("text", data_dir="dataset/train", split="train", sample_by="document")
eval_dataset = datasets.load_dataset("text", data_dir="dataset/eval", split="train", sample_by="document")

# Continue with the rest of the code using the train and test sets
# Create a dataset for training the LLaMA model
predictor = llama2_engine.TextPredictor()
predictor.load_model(base_model_path="../../models/llama2_7b_chat_uncensored") #, lora_model_path="./lora"

def test_llama_model(X_test, y_test):
  # Test the LLaMA model before fine-tuning
  maes = []
  for ind in tqdm(range(len(X_test))):
    reduced_embeddings_str, artists, titles = X_test[ind]
    score = y_test[ind]

    prompt, response = produce_prompt_score_from_song(reduced_embeddings_str, artists, titles, score)
    llm_response = predictor.predict_from_text(prompt, temperature = 0, max_new_tokens = 16)

    predicted = float(llm_response.split(';')[0].split('/')[0]) # in case the float number was predicted
    predicted = int(round(predicted))
    #print("correct response", y_test[ind])
    #print("llm response", predicted)

    mae = np.mean(np.abs(predicted - score))
    maes.append(mae)

  return np.mean(maes)

metric = test_llama_model(X_test, y_test)
print(f"Before training metric value on test set: {metric :.4f}")

# Fine-tune the LLaMA model
predictor.fine_tune_model(train_dataset, None, lora_model_path="./lora", num_train_epochs = 20) 

metric = test_llama_model(X_test, y_test)
print(f"After training metric value on test set: {metric :.4f}")