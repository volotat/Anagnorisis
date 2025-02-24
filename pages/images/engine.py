from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel, SiglipVisionModel, SiglipTextModel
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

import src.scoring_models
import pages.images.db_models as db_models
import pages.file_manager as file_manager


images_embeds_fast_cache = {}

def get_image_metadata(file_path):
    """
    Extracts relevant metadata from an image file.
    Currently, it only extracts the image resolution, but can be extended.
    """
    metadata = {}
    try:
        with Image.open(file_path) as img:
            metadata['resolution'] = img.size  # (width, height)
    except Exception as e:
        raise Exception(f"Error extracting metadata from {file_path}: {e}")
        print(f"Error extracting metadata from {file_path}: {e}")
        metadata['resolution'] = None  # Or some default value
    return metadata

class ImageSearch:
  device = None
  model = None
  processor = None
  is_busy = False

  @staticmethod
  def initiate():
    if ImageSearch.model is not None:
      return
    
    ImageSearch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ImageSearch.model = AutoModel.from_pretrained("./models/siglip-base-patch16-224", local_files_only=True).to(ImageSearch.device)
    ImageSearch.processor = AutoProcessor.from_pretrained("./models/siglip-base-patch16-224", local_files_only=True)
    ImageSearch.model_hash = ImageSearch.get_model_hash()
    ImageSearch.embedding_dim = ImageSearch.model.config.text_config.hidden_size

    ImageSearch.cached_file_list = file_manager.CachedFileList('cache/images_file_list.pkl')
    ImageSearch.cached_file_hash = file_manager.CachedFileHash('cache/images_file_hash.pkl')
    ImageSearch.cached_metadata = file_manager.CachedMetadata('cache/images_metadata.pkl', get_image_metadata)

  '''_instance = None
  def __new__(self, *args, **kwargs):
    if not self._instance:
      self._instance = super().__new__(self)
      
    return self._instance
    
  def __init__(self) -> None:
    pass'''

  @staticmethod
  def get_model_hash():
    state_dict = ImageSearch.model.state_dict()
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    model_hash = hashlib.md5(buffer.read()).hexdigest()
    return model_hash
  
  @staticmethod
  def read_image(image_path):
    try:
      # In case the image has non-typical format, we need to handle it separately
      if image_path.lower().endswith('.gif'):
        # Read GIF file using imageio
        gif = imageio.mimread(image_path)
        # Convert the first frame to a numpy array
        image = np.array(gif[0])
      else:
        # Read the image using OpenCV
        image = cv2.imread(image_path)

      if image is None:
        print(f"Unable to read image: {image_path}")
        return None

      # Check if the image is grayscale
      if len(image.shape) == 2 or image.shape[2] == 1:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
      elif image.shape[2] == 3:
        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      elif image.shape[2] == 4:
        # Convert from BGRA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
      else:
        raise ValueError("Invalid image shape")
      return image
    
    except OSError as e:
      print(f"Error reading image {image_path}: {e}")
      return None

  @staticmethod
  def process_images(images, batch_size=32, callback=None, media_folder=None):
    ImageSearch.initiate()

    # Check if the ImageSearch is busy processing another request
    if ImageSearch.is_busy: 
      raise Exception("ImageSearch is busy")
    
    # Set the busy flag to prevent multiple calls
    ImageSearch.is_busy = True

    all_image_embeds = []
    new_items = []

    # Initialize timing variables
    time_image_read = 0
    time_image_preprocess = 0
    time_get_embeddings = 0

    for ind, image_path in enumerate(tqdm(images)):
      try:
        # Compute the hash of the image file
        image_hash = ImageSearch.cached_file_hash.get_file_hash(image_path)
        
        # If the cache file exists, load the embeddings from it
        if image_hash in images_embeds_fast_cache:
          image_embeds = images_embeds_fast_cache[image_hash]
        else:
          # Check if the embedding exists in the database
          image_record = db_models.ImagesLibrary.query.filter_by(hash=image_hash).first()

          if image_record and image_record.embedding:
            # Load the embeddings from the database
            image_embeds = pickle.loads(image_record.embedding)
            # Save the embeddings to the fast cache (RAM)
            images_embeds_fast_cache[image_hash] = image_embeds
          else:
            start_time = time.time()
            # Process the image and generate embeddings
            image = ImageSearch.read_image(image_path)
            time_image_read += time.time() - start_time

            if image is None:
              raise Exception(f"Error reading image: {image_path}")

            # TODO: Find a way to preprocess and compute embeddings in one step in a bulk

            # Use the numpy array directly with the processor
            start_time = time.time()
            inputs_images = ImageSearch.processor(images=[image], padding="max_length", return_tensors="pt").to(ImageSearch.device)
            time_image_preprocess += time.time() - start_time

            # Get the image embeddings
            start_time = time.time()
            with torch.no_grad():
                outputs = ImageSearch.model.get_image_features(**inputs_images)

            # Get the image embeddings
            image_embeds = outputs

            # Normalize the image embeddings
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            time_get_embeddings += time.time() - start_time

            # Save the embeddings to the database
            if image_record:
              image_record.embedding = pickle.dumps(image_embeds)
              image_record.embedder_hash = ImageSearch.model_hash
            else:
              # This require knowledge of the media folder to find the relative path
              if media_folder:
                image_data = {
                  'hash': image_hash,
                  'file_path': os.path.relpath(image_path, media_folder),
                  'embedding': pickle.dumps(image_embeds),
                  'embedder_hash': ImageSearch.model_hash
                }
                new_items.append(db_models.ImagesLibrary(**image_data))

            # Save the embeddings to the fast cache (RAM)
            images_embeds_fast_cache[image_hash] = image_embeds
      except Exception as e:
        print(f"Error processing image: {image_path}: {e}")
        image_embeds = torch.zeros(1, ImageSearch.embedding_dim).to(ImageSearch.device)

      all_image_embeds.append(image_embeds)
      if callback: callback(ind + 1, len(images))

    # Calculate and print average times
    num_images = len(images)
    print(f"Average time to read images: {time_image_read / num_images:.4f} seconds")
    print(f"Average time to preprocess images: {time_image_preprocess / num_images:.4f} seconds")
    print(f"Average time to get image embeddings: {time_get_embeddings / num_images:.4f} seconds")

    # Save the embeddings to the database
    if new_items: db_models.db.session.bulk_save_objects(new_items)
    db_models.db.session.commit()

    # Concatenate all embeddings
    all_image_embeds = torch.cat(all_image_embeds, dim=0)

    # Reset the busy flag
    ImageSearch.is_busy = False

    return all_image_embeds

  @staticmethod
  def process_text(text):
    ImageSearch.initiate()
    
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
    print(embeds_img.shape, embeds_text.shape) 
    logits_per_text = torch.matmul(embeds_text, embeds_img.t()) * ImageSearch.model.logit_scale.exp() + ImageSearch.model.logit_bias
    logits_per_image = logits_per_text.t()

    probs = torch.sigmoid(logits_per_image)

    return probs.cpu().detach().numpy() 

# Create scoring model singleton class so it easily accessible from other modules
class ImageEvaluator(src.scoring_models.Evaluator):
  _instance = None

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(ImageEvaluator, cls).__new__(cls)
    return cls._instance

  def __init__(self, embedding_dim=768, rate_classes=11):
    if not hasattr(self, '_initialized'):
      super(ImageEvaluator, self).__init__(embedding_dim, rate_classes)
      self._initialized = True
