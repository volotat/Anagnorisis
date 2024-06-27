from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, SiglipVisionModel, SiglipTextModel
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import hashlib

class ImageSearch:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = AutoModel.from_pretrained("./models/siglip-base-patch16-224", local_files_only=True).to(device)
  processor = AutoProcessor.from_pretrained("./models/siglip-base-patch16-224", local_files_only=True)

  '''_instance = None
  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      
    return self._instance
    
  def __init__(self) -> None:
    pass'''

  @staticmethod
  def process_images(images, batch_size=32):
    # TODO: Rewrite caching functionality to use less files and speed up its performance

    # create cache directory
    os.makedirs("./cache/embeds_images", exist_ok=True) 

    # Split the images into batches
    # image_batches = [images[i:i + batch_size] for i in range(0, len(images), batch_size)]

    all_image_embeds = []

    for image_path in tqdm(images):
      # Compute the hash of the image file
      #with open(image_path, "rb") as f:
      #  bytes = f.read() # read entire file as bytes
      #  readable_hash = hashlib.sha256(bytes).hexdigest()
      readable_hash = hashlib.md5(image_path.encode()).hexdigest()

      # Define the path of the cache file
      cache_file = f"./cache/embeds_images/{readable_hash}.pkl"

      # If the cache file exists, load the embeddings from it
      if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
          image_embeds = pickle.load(f)
      else:
        # Otherwise, process the image and save the embeddings to the cache file
        image = Image.open(image_path).convert("RGB")
        inputs_images = ImageSearch.processor(images=[image], padding="max_length", return_tensors="pt").to(ImageSearch.device)

        with torch.no_grad():
          outputs = ImageSearch.model.get_image_features(**inputs_images)

        # get the image embeddings
        image_embeds = outputs

        # normalize the image embeddings
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)

        # Save the embeddings to the cache file
        with open(cache_file, "wb") as f:
          pickle.dump(image_embeds, f)

      all_image_embeds.append(image_embeds)

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)

    return all_image_embeds

  @staticmethod
  def process_text(text):
    inputs_text = ImageSearch.processor(text=text, padding="max_length", return_tensors="pt").to(ImageSearch.device)

    with torch.no_grad():
      outputs = ImageSearch.model.get_text_features(**inputs_text)

    # get the text embeddings
    text_embeds = outputs

    # normalize the text embeddings
    text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

    return text_embeds

  @staticmethod
  def compare(embeds_img, embeds_text):
    logits_per_text = torch.matmul(embeds_text, embeds_img.t()) * ImageSearch.model.logit_scale.exp() + ImageSearch.model.logit_bias
    logits_per_image = logits_per_text.t()

    probs = torch.sigmoid(logits_per_image)

    return probs.cpu().detach().numpy() 

