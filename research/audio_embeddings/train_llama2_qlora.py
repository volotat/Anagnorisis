import numpy as np
from tqdm import tqdm
import os

import llama2_engine
import utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

utils.set_seed(42)

# print better represented numpy arrays
np.set_printoptions(precision=3, suppress=True)

def compare_strings_ignore_case_and_space(str1, str2):
  str1 = str1.split(';')[0]
  str2 = str2.split(';')[0]
  return str1.lower().strip()[:30] == str2.lower().strip()[:30]

def convert_vector_to_letters(embeddings, min_allowed_value = -1, max_allowed_value = 1):
  # list of characters that by itself generate only a single token in the llama model, special characters are not included
  charset = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'А', 'а', 'Б', 'б', 'В', 'в', 'Г', 'г', 'Д', 'д', 'Е', 'е', 'Ë', 'ë', 'Ж', 'ж', 'З', 'з', 'И', 'и', 'Й', 'й', 'К', 'к', 'Л', 'л', 'М', 'м', 'Н', 'н', 'О', 'о', 'П', 'п', 'Р', 'р', 'С', 'с', 'Т', 'т', 'У', 'у', 'Ф', 'ф', 'Х', 'х', 'Ц', 'ц', 'Ч', 'ч', 'Ш', 'ш', 'Щ', 'щ', 'Ъ', 'ъ', 'Ы', 'ы', 'Ь', 'ь', 'Э', 'э', 'Ю', 'ю', 'Я', 'я', 'ә', 'Ђ', 'ђ', 'Đ', 'đ', 'Љ', 'љ', 'Њ', 'њ', 'Ћ', 'ћ', 'Ć', 'ć', 'Č', 'č', 'Џ', 'џ', 'Š', 'š']
  granularity = len(charset) - 1

  embeddings = np.clip(embeddings, min_allowed_value, max_allowed_value)

  embeddings = (embeddings - min_allowed_value) / (max_allowed_value - min_allowed_value) # normalize to [0, 1]

  states = (embeddings * granularity).astype(int)
  letter_representation = ''.join([charset[i] for i in states])

  return letter_representation

def produce_prompt(reduced_embedding_str, artist): #Here is the list of possible bands: {", ".join(set(artists))}.
  prompt = f'''### HUMAN:
Who are playing this song? Please, tell me only the name of the artist or band without any punctuations or anything else.
[music]{reduced_embedding_str}[/music]
### RESPONSE:
Artist:'''
  response = f" {artist};"
  return prompt, response

# Load the dataset
embeddings, reduced_embeddings, artists, titles = utils.load_data_from_csv("dataset.csv")
le = LabelEncoder()
artists = le.fit_transform(artists)
print("Dataset has been loaded")

# Calculate reduced embeddings statistics for proper normalization
print("Reduced embeddings shape:", reduced_embeddings.shape)  
mean = np.mean(reduced_embeddings, axis=0)
var = np.var(reduced_embeddings, axis=0)
reduced_embeddings = (reduced_embeddings - mean) / np.sqrt(var)

# Convert reduced embeddings into a string representation 
reduced_embeddings_str = [convert_vector_to_letters(emb) for emb in reduced_embeddings]
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(reduced_embeddings_str, artists, test_size=0.1, random_state=42)

# Continue with the rest of the code using the train and test sets
# Create a dataset for training the LLaMA model
predictor = llama2_engine.TextPredictor()
predictor.load_model(base_model_path="../../models/llama2_7b_chat_uncensored") #, lora_model_path="./lora"

def test_llama_model(X_test, y_test):
  # Test the LLaMA model before fine-tuning
  right_answers = 0
  for ind in tqdm(range(len(X_test))):
    prompt, response = produce_prompt(X_test[ind], y_test[ind])
    llm_response = predictor.predict_from_text(prompt, temperature = 0, max_new_tokens = 16)
    #print("correct response", response)
    #print("llm response", len(llm_response), ":", llm_response)
    #print("Is it right?", compare_strings_ignore_case_and_space(response, llm_response))

    if compare_strings_ignore_case_and_space(response, llm_response):
      right_answers +=1

  return right_answers / len(X_test)


'''# Convert X_train and y_train to NumPy arrays for fast indexing
X_train = np.array(X_train)
y_train = np.array(y_train)

# Test the model accuracy on the train set
indexes = np.random.randint(0, len(X_train), size=len(X_test))
accuracy = test_llama_model(X_train[indexes], y_train[indexes])
print(f"Reduced embeddings Train Accuracy: {accuracy * 100 :.2f}%")

# Test the model accuracy on the test set
accuracy = test_llama_model(X_test, y_test)
print(f"Reduced embeddings Test Accuracy: {accuracy * 100 :.2f}%")

exit()'''

import datasets

# Create dataset for fine-tuning the LLaMA model
os.makedirs("dataset/train", exist_ok=True)
os.makedirs("dataset/eval", exist_ok=True)

for ind in tqdm(range(len(X_train))):
  prompt, response = produce_prompt(X_train[ind], y_train[ind])
  with open(f"dataset/train/{ind}.txt", "w") as f:
    f.write(prompt + response)

for ind in tqdm(range(len(X_test))):
  prompt, response = produce_prompt(X_test[ind], y_test[ind])
  with open(f"dataset/eval/{ind}.txt", "w") as f:
    f.write(prompt + response)

# Load the datasets
train_dataset = datasets.load_dataset("text", data_dir="dataset/train", split="train", sample_by="document")
eval_dataset = datasets.load_dataset("text", data_dir="dataset/eval", split="train", sample_by="document")

# Fine-tune the LLaMA model
predictor.fine_tune_model(train_dataset, eval_dataset, lora_model_path="./lora", num_train_epochs = 1) #eval_dataset

# Convert X_train and y_train to NumPy arrays for fast indexing
X_train = np.array(X_train)
y_train = np.array(y_train)

# Test the model accuracy on the train set
indexes = np.random.randint(0, len(X_train), size=len(X_test))
accuracy = test_llama_model(X_train[indexes], y_train[indexes])
print(f"Reduced embeddings Train Accuracy: {accuracy * 100 :.2f}%")

# Test the model accuracy on the test set
accuracy = test_llama_model(X_test, y_test)
print(f"Reduced embeddings Test Accuracy: {accuracy * 100 :.2f}%")