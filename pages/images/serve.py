
from flask import Flask, render_template, send_from_directory
import os
import sys
import glob
import subprocess
import hashlib
import numpy as np
from io import BytesIO

from pages.images.engine import ImageSearch
import math
from scipy.spatial import distance
import torch
import gc
import send2trash
import time
import datetime
import pages.images.db_models as db_models
import pickle

import src.scoring_models

# TODO: Move this class to into some separate files with callbacks or something like it
class EmbeddingGatheringCallback:
  def __init__(self, show_status_function = None):
    self.last_shown_time = 0
    self.show_status_function = show_status_function

  def __call__(self, num_extracted, num_total):
    current_time = time.time()
    if current_time - self.last_shown_time >= 1:
      self.show_status_function(f"Extracted embeddings for {num_extracted}/{num_total} images.")
      self.last_shown_time = current_time


def convert_size(size_bytes):
  if size_bytes == 0:
      return "0B"
  size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
  i = int(math.floor(math.log(size_bytes, 1024)))
  p = math.pow(1024, i)
  s = round(size_bytes / p, 2)
  return f"{s} {size_name[i]}"

def compute_distances_batched(embeds_img, batch_size=1024 * 24):
  # Ensure input is a torch tensor (on CPU)
  embeds_img = torch.tensor(embeds_img, dtype=torch.float32)

  num_images = embeds_img.shape[0]
  # Initialize the distances matrix on the CPU
  distances = torch.zeros((num_images, num_images), dtype=torch.float32)

  for start_row in range(0, num_images, batch_size):
    print(f"Processing row {start_row} of {num_images}")

    end_row = min(start_row + batch_size, num_images)
    # Move only the current batch to GPU
    batch = embeds_img[start_row:end_row].cuda()

    for start_col in range(0, num_images, batch_size):
      end_col = min(start_col + batch_size, num_images)
      # Move the comparison batch to GPU
      compare_batch = embeds_img[start_col:end_col].cuda()
      # Compute pairwise distances for the batch on GPU
      dists_batch = torch.cdist(batch, compare_batch, p=2).cpu()  # Move results back to CPU
      
      distances[start_row:end_row, start_col:end_col] = dists_batch
      if start_col != start_row:  # Fill the symmetric part of the matrix
        distances[start_col:end_col, start_row:end_row] = dists_batch.T

      del compare_batch  # Free the memory used by the comparison batch
      del dists_batch  # Free the memory used by the batch

    del batch  # Free the memory used by the batch

  distances.fill_diagonal_(float('inf'))  # Ignore self-distances
  distances = distances.detach().numpy()  # Convert to a NumPy array

  gc.collect()  # Force garbage collection to free memory
  torch.cuda.empty_cache()  # Free the memory used by the GPU
  
  return distances  # Convert to a NumPy

def get_folder_structure(root_folder):
  folder_dict = {}
  for root, dirs, _ in os.walk(root_folder):
    # Extract the relative path from the root_folder to the current root
    rel_path = os.path.relpath(root, root_folder)
    # Skip the root folder itself
    if rel_path == ".":
      rel_path = ""
    # Navigate/create nested dictionaries based on the relative path
    current_level = folder_dict
    for part in rel_path.split(os.sep):
      if part:  # Avoid empty strings
        current_level = current_level.setdefault(part, {})
    # Add subdirectories to the current level
    for d in dirs:
      current_level[d] = {}
  return folder_dict

'''
def get_file_descriptor(file_path):
  # This will only work on Unix-like systems
  # TODO: Add support for Windows
  stat_info = os.stat(file_path)
  inode = stat_info.st_ino
  device = stat_info.st_dev
  return f"{inode}-{device}"
'''

from PIL import Image
def get_image_resolution(file_path):
    with Image.open(file_path) as img:
        return img.size

'''
import pywintypes
import win32file

def get_windows_file_descriptor(file_path):
    handle = win32file.CreateFile(
        file_path,
        win32file.GENERIC_READ,
        win32file.FILE_SHARE_READ | win32file.FILE_SHARE_WRITE | win32file.FILE_SHARE_DELETE,
        None,
        win32file.OPEN_EXISTING,
        win32file.FILE_ATTRIBUTE_NORMAL,
        None
    )
    
    file_info = win32file.GetFileInformationByHandle(handle)
    handle.Close()
    
    file_index = file_info[8]  # File index (high and low combined)
    volume_serial = file_info[1]  # Volume serial number
    
    return f"{volume_serial}-{file_index}"
'''

# TODO: Move hash caching into utils as it will be useful for all modules

###########################################
# File Hash Caching

# Cache dictionary to store file path, last modified time, and hash value
file_hash_cache_file_path = 'cache/images_hashes.pkl'
file_hash_cache = {}

def load_hash_cache():
  global file_hash_cache

  if os.path.exists(file_hash_cache_file_path):
    with open(file_hash_cache_file_path, 'rb') as cache_file:
      file_hash_cache = pickle.load(cache_file)
    
    # Remove entries older than three months
    three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
    file_hash_cache = {k: v for k, v in file_hash_cache.items() if v[2] > three_months_ago}

def save_hash_cache():
  # Save the updated cache to the file
  with open(file_hash_cache_file_path, 'wb') as cache_file:
    pickle.dump(file_hash_cache, cache_file)

def get_file_hash(file_path):
  global file_hash_cache

  # Load the cache from the file if it exists and file_hash_cache is empty
  if not file_hash_cache: load_hash_cache()

  # Get the last modified time of the file
  last_modified_time = os.path.getmtime(file_path)
  
  # Check if the file is in the cache and if the last modified time matches
  if file_path in file_hash_cache:
    cached_last_modified_time, cached_hash, timestamp = file_hash_cache[file_path]
    if cached_last_modified_time == last_modified_time:
      return cached_hash
  
  # If not in cache or file has been modified, calculate the hash
  with open(file_path, "rb") as f:
    bytes = f.read()  # Read the entire file as bytes
    file_hash = hashlib.md5(bytes).hexdigest()
  
  # Update the cache
  file_hash_cache[file_path] = (last_modified_time, file_hash, datetime.datetime.now())

  return file_hash


###########################################
# File List Caching

file_list_cache_file_path = 'cache/images_file_list.pkl'
file_list_cache = {}

def load_file_list_cache():
  global file_list_cache
  if os.path.exists(file_list_cache_file_path):
    with open(file_list_cache_file_path, 'rb') as cache_file:
      file_list_cache = pickle.load(cache_file)

    # Remove entries older than three months
    three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
    file_list_cache = {k: v for k, v in file_list_cache.items() if v[2] > three_months_ago}

def save_file_list_cache():
  with open(file_list_cache_file_path, 'wb') as cache_file:
    pickle.dump(file_list_cache, cache_file)

def get_files_in_folder(folder_path, media_formats):
    global file_list_cache

    # Get the last modified time of the folder
    folder_last_modified_time = os.path.getmtime(folder_path)
    
    # Check if the folder is in the cache and if the last modified time matches
    if folder_path in file_list_cache:
        cached_last_modified_time, cached_file_list, timestamp = file_list_cache[folder_path]
        if cached_last_modified_time == folder_last_modified_time:
            return cached_file_list
    
    # If not in cache or folder has been modified, list the files in the folder
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(media_formats))]
    
    # Update the cache with the current timestamp and file list
    file_list_cache[folder_path] = (folder_last_modified_time, file_list, datetime.datetime.now())
    
    return file_list

def get_all_files(current_path, media_formats):
    global file_list_cache

    # Load the cache from the file if it exists and file_list_cache is empty
    if not file_list_cache:
        load_file_list_cache()

    all_files = []
    for root, dirs, files in os.walk(current_path):
        all_files.extend(get_files_in_folder(root, media_formats))

    # Save the updated cache to the file
    save_file_list_cache()
    
    return all_files

###########################################
# Metadata Caching
# TODO: Implement metadata caching
# file_size, resolution, glip_embedding


def init_socket_events(socketio, app=None, cfg=None):
  media_directory = cfg.images.media_directory

  image_evaluator = src.scoring_models.Evaluator(embedding_dim=768)
  image_evaluator.load('./models/image_evaluator.pt')

  def show_search_status(status):
    socketio.emit('emit_images_page_show_search_status', status)

  embedding_gathering_callback = EmbeddingGatheringCallback(show_search_status)
  

  def update_model_ratings(files_list):
    # filter out files that already have a rating in the DB
    # TODO: make it dependent on the hash of the model, so when the model is updated, the ratings are updated too
    file_hash_map = {file_path: get_file_hash(file_path) for file_path in files_list}
    hash_list = list(file_hash_map.values())

    rated_images_db_items = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.hash.in_(hash_list)).all()
    rated_images = {item.hash: item.model_rating for item in rated_images_db_items}

    # Filter out files that already have a rating in the database
    filtered_files_list = [file_path for file_path, file_hash in file_hash_map.items() if file_hash not in rated_images]
    if len(filtered_files_list) == 0: return

    # Rate all images in case they are not rated or model was updated
    embeddings = ImageSearch.process_images(filtered_files_list, callback=embedding_gathering_callback) #.cpu().detach().numpy() 
    model_ratings = image_evaluator.predict(embeddings)

    # Update the model ratings in the database
    show_search_status(f"Updating model ratings of images...") 
    for ind, full_path in enumerate(filtered_files_list):
      hash = file_hash_map[full_path]
      file_descriptor = "" #get_file_descriptor(full_path)

      image_db_item = db_models.ImagesLibrary.query.filter_by(hash=hash).first()
      model_rating = model_ratings[ind].item()
      if image_db_item is not None:
        image_db_item.model_rating = model_rating
      else:
        # Create new instance us there is no image in the database
        image_data = {
          "hash": hash,
          "file_path": os.path.relpath(full_path, media_directory),
          "file_descriptor": file_descriptor,
          "model_rating": model_rating
        }

        image_db_item = db_models.ImagesLibrary(**image_data)
        db_models.db.session.add(image_db_item)

    db_models.db.session.commit()

  # necessary to allow web application access to music files
  @app.route('/image_files/<path:filename>')
  def serve_image_files(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_images_page_get_files')
  def get_files(input_data):
    nonlocal media_directory
    start_time = time.time()

    path = input_data.get('path', '')
    pagination = input_data.get('pagination', 0)
    limit = input_data.get('limit', 100)
    text_query = input_data.get('text_query', None)
    
    files_data = []
    if path == "":
      current_path = media_directory
    else:
      current_path = os.path.join(media_directory, '..', path)


    folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
    print('folder_path', folder_path)

    show_search_status(f"Searching for images in '{folder_path}'.")
  
    # Walk with cache 1.5s for 66k files
    all_files = get_all_files(current_path, cfg.images.media_formats)

    show_search_status(f"Gathering hashes for {len(all_files)} images.")
    
    # Initialize the last shown time
    last_shown_time = 0

    # Get the hash of each file
    all_hashes = []
    for ind, file_path in enumerate(all_files):
      current_time = time.time()
      if current_time - last_shown_time >= 1:
        show_search_status(f"Gathering hashes for {ind+1}/{len(all_files)} images.")
        last_shown_time = current_time
      
      all_hashes.append(get_file_hash(file_path))

    save_hash_cache()


    # Sort image by text or image query
    show_search_status(f"Sorting images by {text_query}")
    if text_query and len(text_query) > 0:
      # Sort images by file size
      if text_query.lower().strip() == "file size":
        all_files = sorted(all_files, key=os.path.getsize)
      # Sort images by resolution
      elif text_query.lower().strip() == "resolution":
        if len(all_files) > 10000:
          raise Exception("Too many images to sort by proportion")
        # TODO: THIS IS VERY SLOW, NEED TO OPTIMIZE
        resolutions = {file_path: get_image_resolution(file_path) for file_path in all_files}
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])
      # Sort images by proportion
      elif text_query.lower().strip() == "proportion":
        if len(all_files) > 10000:
          raise Exception("Too many images to sort by proportion")
        # TODO: THIS IS VERY SLOW, NEED TO OPTIMIZE
        resolutions = {file_path: get_image_resolution(file_path) for file_path in all_files}
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] / resolutions[x][1])
      # Sort images by duplicates
      elif text_query.lower().strip() == "similarity":
        show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files, callback=embedding_gathering_callback).cpu().detach().numpy() 

        show_search_status(f"Computing distances between embeddings")
        # Assuming embeds_img is an array of image embeddings
        distances = compute_distances_batched(embeds_img)

        # memory inefficient but fast way to compute the pairwise distance matrix
        # distances = np.sqrt(((embeds_img[:, np.newaxis] - embeds_img[np.newaxis, :]) ** 2).sum(axis=2))

        # Set the diagonal to a large number to ignore self-distances
        #np.fill_diagonal(distances, np.inf)

        # Find the smallest distance for each embedding
        min_distances = np.min(distances, axis=1)
        
        show_search_status(f"Clustering images by similarity")
        ##################################################
        # Sort the embeddings by the smallest distance
        # if similarity is exactly zero, sort by file name

        # Step 1: Identify the most similar image for each image
        # Find the index of the smallest distance for each image (ignoring self-distances)
        target_indices = np.argmin(distances, axis=1)

        # Step 2: Use the min_distances of the target image
        # Extract the min_distances for the target images
        target_min_distances = min_distances[target_indices]

        # Step 3: Adjust the sorting logic
        # Get file sizes for all files
        file_sizes = [os.path.getsize(file_path) for file_path in all_files]
        # Create a list of tuples (target_min_distance, file_size, file_path) for each file
        files_with_target_distances_and_sizes = list(zip(target_min_distances, min_distances, file_sizes, all_files))
        # Sort the list of tuples by target_min_distance, then by file_size
        sorted_files = sorted(files_with_target_distances_and_sizes, key=lambda x: (x[0], x[1], x[2]))
        # Extract the sorted list of file paths
        all_files = [file_path for _, _, _, file_path in sorted_files]
      # Sort images randomly
      elif text_query.lower().strip() == "random":
        np.random.shuffle(all_files)
      # Sort images by rating
      elif text_query.lower().strip() == "rating": 
        # Update the model ratings of all current images
        update_model_ratings(all_files)

        # Get all ratings from the database
        all_rated_images_db_items = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.hash.in_(all_hashes)).all()
        if len(all_rated_images_db_items) > 0:
          # Create a dictionary of all ratings
          all_ratings = {}
          for image in all_rated_images_db_items:
            if image.user_rating is not None:
              all_ratings[image.hash] = image.user_rating
            else:
              all_ratings[image.hash] = image.model_rating

          # Calculate the mean score
          mean_score = np.mean(list(all_ratings.values()))

          # Sort images by user rating
          all_files = sorted(all_files, key=lambda x: all_ratings.get(get_file_hash(x), mean_score), reverse=True)
      # Present images as is
      else:
        show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files, callback=embedding_gathering_callback)
        embeds_text = ImageSearch.process_text(text_query)
        scores = ImageSearch.compare(embeds_img, embeds_text)

        # Create a list of indices sorted by their corresponding score
        show_search_status(f"Sorting by relevance")
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]

    #all_files = sorted(all_files, key=os.path.basename)


    # Extracting metadata for relevant batch of images
    show_search_status(f"Extracting metadata for relevant batch of images")
    page_files = all_files[pagination:limit]
    page_hashes = [get_file_hash(file_path) for file_path in page_files]

    # Extract DB data for the relevant batch of images
    image_db_items = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.hash.in_(page_hashes)).all()
    image_db_items_map = {item.hash: item for item in image_db_items}

    # Check if there images without model rating
    no_model_rating_images = [item.hash for item in image_db_items if item.model_rating is None]
    print('no_model_rating_images size', len(no_model_rating_images))

    if len(no_model_rating_images) > 0:
      # Update the model ratings of all current images
      update_model_ratings(page_files)

    for ind, full_path in enumerate(page_files):
      basename = os.path.basename(full_path)
      file_size = os.path.getsize(full_path)

      hash = page_hashes[ind]
      resolution = get_image_resolution(full_path)  # Returns a tuple (width, height)
    
      file_descriptor = "" #get_file_descriptor(full_path)

      user_rating = None
      model_rating = None

      if hash in image_db_items_map:
        image_db_item = image_db_items_map[hash]
        user_rating = image_db_item.user_rating
        model_rating = image_db_item.model_rating

      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": os.path.relpath(full_path, media_directory),
        "file_descriptor": file_descriptor,
        "base_name": basename,
        "hash": hash,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "file_size": convert_size(file_size),
        "resolution": f"{resolution[0]}x{resolution[1]}",
      }
      files_data.append(data)
                 
    # Extract subfolders structure from the path into a dict
    folders = get_folder_structure(media_directory)

    # Extract main folder name
    main_folder_name = os.path.basename(os.path.normpath(media_directory))
    folders = {main_folder_name: folders}
    
    socketio.emit('emit_images_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders})

    show_search_status(f'{len(all_files)} images processed in {time.time() - start_time:.4f} seconds.')

  @socketio.on('emit_images_page_open_file_in_folder')
  def open_file_in_folder(file_path):
    file_path = os.path.normpath(file_path)
    print(f'Opening file with path: "{file_path}"')
    
    # Assuming file_path is the full path to the file
    folder_path = os.path.dirname(file_path)
    if os.path.isfile(file_path):
      if sys.platform == "win32":  # Windows
        subprocess.run(["explorer", "/select,", file_path], check=True)
      elif sys.platform == "darwin":  # macOS
        subprocess.run(["open", "-R", file_path], check=True)
      else:  # Linux and other Unix-like OS
        # Convert the file path to an absolute path
        abs_path = os.path.abspath(file_path)

         # Check for the file manager and use the appropriate command on Linux
        if os.environ.get('XDG_CURRENT_DESKTOP') in ['GNOME', 'Unity']:
          subprocess.run(['nautilus', '--no-desktop', abs_path])
        elif os.environ.get('XDG_CURRENT_DESKTOP') == 'KDE':
          subprocess.run(['dolphin', '--select', abs_path])
        else:
          print("Unsupported desktop environment. Please add support for your file manager.")
    else:
      print("Error: File does not exist.")

  @socketio.on('emit_images_page_send_file_to_trash')
  def send_file_to_trash(file_path):
    if os.path.isfile(file_path):
      send2trash.send2trash(file_path)
      '''if sys.platform == "win32":  # Windows
        # Move the file to the recycle bin
        subprocess.run(["cmd", "/c", "del", "/q", "/f", file_path], check=True)
      elif sys.platform == "darwin":  # macOS
        # Move the file to the trash
        subprocess.run(["trash", file_path], check=True)
      else:  # Linux and other Unix-like OS
        # Move the file to the trash
        subprocess.run(["gio", "trash", file_path], check=True)'''
    else:
      print("Error: File does not exist.")  

  @socketio.on('emit_images_page_set_image_rating')
  def set_image_rating(data):
    image_hash = data['hash'] 
    image_rating = data['rating']
    image_path = data['file_path']
    image_file_descriptor = data['file_descriptor']
    
    print('Set image rating:', image_hash, image_rating)

    image_db_item = db_models.ImagesLibrary.query.filter_by(hash=image_hash, file_descriptor=image_file_descriptor).first()

    if image_db_item is None:  
      # Create new instance us there is no image in the database

      image_data = {
        "hash": image_hash,
        "file_path": image_path,
        "file_descriptor": image_file_descriptor,
        "user_rating": int(image_rating),
        "user_rating_date": datetime.datetime.now()
      }

      image_db_item = db_models.ImagesLibrary(**image_data)
      db_models.db.session.add(image_db_item)
      db_models.db.session.commit()
    else: 
      # Update the existing instance

      image_db_item.file_path = image_path # in case the file was moved
      image_db_item.user_rating = int(image_rating)
      image_db_item.user_rating_date = datetime.datetime.now()
      db_models.db.session.commit()