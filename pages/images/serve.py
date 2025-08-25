
from flask import Flask, render_template, send_from_directory
import os
import sys
import glob
import subprocess
import hashlib
import numpy as np
from io import BytesIO

from pages.images.engine import ImageSearch, ImageEvaluator
import math
from scipy.spatial import distance
import torch
import gc
import send2trash
import time
import datetime
import pages.images.db_models as db_models
import pickle
from omegaconf import OmegaConf

import src.scoring_models
from pages.utils import convert_size
from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback
from pages.socket_events import CommonSocketEvents
import pages.file_manager as file_manager

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

def get_folder_structure(folder_path, image_extensions=None):
  def count_images(folder):
    return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in image_extensions)

  def build_structure(path):
    folder_dict = {
      'name': os.path.basename(path),
      'num_images': count_images(path),
      'total_images': 0,
      'subfolders': {}
    }
    folder_dict['total_images'] = folder_dict['num_images']
    
    for subfolder in os.listdir(path):
      subfolder_path = os.path.join(path, subfolder)
      if os.path.isdir(subfolder_path):
        subfolder_structure = build_structure(subfolder_path)
        folder_dict['subfolders'][subfolder] = subfolder_structure
        folder_dict['total_images'] += subfolder_structure['total_images']
    
    return folder_dict
  return build_structure(folder_path)

def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
  if cfg.images.media_directory is None:
    print("Images media folder is not set.")
    media_directory = None
  else:
    # media_directory = os.path.join(data_folder, cfg.images.media_directory)
    media_directory = cfg.images.media_directory
  
  image_search_engine = ImageSearch(cfg=cfg)
  image_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)

  image_evaluator = ImageEvaluator() #src.scoring_models.Evaluator(embedding_dim=768)
  image_evaluator.load(os.path.join(cfg.main.personal_models_path, 'image_evaluator.pt'))

  def show_search_status(status):
    socketio.emit('emit_images_page_show_search_status', status)

  def gather_resolutions(all_files, all_hashes, media_directory):
    resolutions = {}
    progress_callback = SortingProgressCallback(show_search_status, operation_name="Gathering images resolution") # Create callback

    for ind, full_path in enumerate(all_files):
      file_path = os.path.relpath(full_path, media_directory)
      file_hash = all_hashes[ind]

      image_metadata = image_search_engine.cached_metadata.get_metadata(full_path, file_hash)
      if 'resolution' not in image_metadata:
        raise Exception(f"Resolution not found for image: {file_path}")
      
      resolutions[full_path] = image_metadata['resolution']
      progress_callback(ind + 1, len(all_files)) # Update progress

    return resolutions

  embedding_gathering_callback = EmbeddingGatheringCallback(show_search_status)
  

  def update_model_ratings(files_list):
    print(files_list)
    # filter out files that already have a rating in the DB
    files_list_hash_map = {file_path: image_search_engine.cached_file_hash.get_file_hash(file_path) for file_path in files_list}
    hash_list = list(files_list_hash_map.values())

    # Fetch rated images from the database in a single query
    rated_images_db_items = db_models.ImagesLibrary.query.filter(
      db_models.ImagesLibrary.hash.in_(hash_list),
      db_models.ImagesLibrary.model_rating.isnot(None),
      db_models.ImagesLibrary.model_hash.is_(image_evaluator.hash)
    ).all()

    # Create a list of hashes for rated images
    rated_images_hashes = {item.hash for item in rated_images_db_items}

    # Filter out files that already have a rating in the database
    filtered_files_list = [file_path for file_path, file_hash in files_list_hash_map.items() if file_hash not in rated_images_hashes]
    if not filtered_files_list: return
    

    # Rate all images in case they are not rated or model was updated
    embeddings = image_search_engine.process_images(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
    model_ratings = image_evaluator.predict(embeddings)

    # Update the model ratings in the database
    show_search_status(f"Updating model ratings of images...") 
    new_items = []
    update_items = []
    last_shown_time = 0
    for ind, full_path in enumerate(filtered_files_list):
      print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} images.")

      hash = files_list_hash_map[full_path]
      model_rating = model_ratings[ind].item()

      image_db_item = db_models.ImagesLibrary.query.filter_by(hash=hash).first()
      if image_db_item:
        image_db_item.model_rating = model_rating
        image_db_item.model_hash = image_evaluator.hash
        update_items.append(image_db_item)
      else:
        image_data = {
            "hash": hash,
            "file_path": os.path.relpath(full_path, media_directory),
            "model_rating": model_rating,
            "model_hash": image_evaluator.hash
        }
        new_items.append(db_models.ImagesLibrary(**image_data))

      current_time = time.time()
      if current_time - last_shown_time >= 1:
        show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} images.")
        last_shown_time = current_time   

    # Bulk update and insert
    if update_items:
        db_models.db.session.bulk_save_objects(update_items)
    if new_items:
        db_models.db.session.bulk_save_objects(new_items)

    # Commit the transaction
    db_models.db.session.commit()

  # necessary to allow web application access to image files
  @app.route('/image_files/<path:filename>')
  def serve_image_files(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_images_page_get_files')
  def get_files(input_data):
    nonlocal media_directory

    if media_directory is None:
      show_search_status("Images media folder is not set.")
      return

    start_time = time.time()

    path = input_data.get('path', '')
    pagination = input_data.get('pagination', 0)
    limit = input_data.get('limit', 100)
    text_query = input_data.get('text_query', None)
    seed = input_data.get('seed', None)

    if seed is not None:
      np.random.seed(int(seed))
    
    files_data = []

    # --- Directory Traversal Prevention ---
    # Resolve the real, canonical path of the safe base directory.
    safe_base_dir = os.path.realpath(media_directory)
    
    # Safely join the user-provided path to the base directory.
    if path == "":
        unsafe_path = safe_base_dir
    else:
        unsafe_path = os.path.join(safe_base_dir, os.pardir, path)
    
    # Resolve the absolute path, processing any '..' and symbolic links.
    current_path = os.path.realpath(unsafe_path)
    
    # Check if the final resolved path is within the safe base directory.
    # This is the crucial security check.
    if os.path.commonpath([current_path, safe_base_dir]) != safe_base_dir:
        show_search_status("Access denied: Directory traversal attempt detected.")
        # Default to the safe base directory if an invalid path is provided.
        current_path = safe_base_dir
    # --- End of Prevention ---

    # if path == "":
    #   current_path = media_directory
    # else:
    #   current_path = os.path.abspath(os.path.join(media_directory, '..', path))


    folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
    print('folder_path', folder_path)

    show_search_status(f"Searching for images in '{folder_path}'.")
  
    # Walk with cache 1.5s for 66k files
    all_files = image_search_engine.cached_file_list.get_all_files(current_path, cfg.images.media_formats)

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
      
      all_hashes.append(image_search_engine.cached_file_hash.get_file_hash(file_path))

    image_search_engine.cached_file_hash.save_hash_cache()


    # Sort image by text or image query
    show_search_status(f"Sorting images by {text_query}")
    if text_query and len(text_query) > 0:
      # If there is an image file with the path of the query, sort image by similarity to that image
      if os.path.isfile(text_query) and text_query.lower().endswith(tuple(cfg.images.media_formats)):
        target_path = text_query
        show_search_status(f"Extracting embeddings")
        embeds_img = image_search_engine.process_images(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        target_emb = image_search_engine.process_images([target_path], callback=embedding_gathering_callback, media_folder=media_directory)

        show_search_status(f"Computing distances between embeddings")
        scores = torch.cdist(embeds_img, target_emb, p=2).cpu().detach().numpy() 

        # Create a list of indices sorted by their corresponding score
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]
      # Sort images by file size
      elif text_query.lower().strip() == "file size":
        all_files = sorted(all_files, key=os.path.getsize)
      # Sort images by resolution
      elif text_query.lower().strip() == "resolution":
        show_search_status(f"Gathering resolutions for sorting...") # Initial status message
        resolutions = gather_resolutions(all_files, all_hashes, media_directory)
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])
      # Sort images by proportion
      elif text_query.lower().strip() == "proportion":
        show_search_status(f"Gathering resolutions for sorting...") # Initial status message
        resolutions = gather_resolutions(all_files, all_hashes, media_directory)
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] / resolutions[x][1])
      # Sort images by duplicates
      elif text_query.lower().strip() == "similarity":
        show_search_status(f"Extracting embeddings")
        embeds_img = image_search_engine.process_images(all_files, callback=embedding_gathering_callback, media_folder=media_directory).cpu().detach().numpy() 

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
        # Create a list of tuples (target_min_distance, min_distances, file_size, file_path) for each file
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
        items = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.hash.in_(all_hashes)).all()

        # Create a dictionary to map hashes to their ratings
        hash_to_rating = {item.hash: item.user_rating if item.user_rating is not None else item.model_rating for item in items}

        # Iterate over all_hashes and populate all_ratings using the dictionary
        all_ratings = [hash_to_rating.get(hash) for hash in all_hashes]

        # Calculate the mean score considering that there might be None values
        mean_score = np.mean([rating for rating in all_ratings if rating is not None])

        # Replace None values with the mean score
        all_ratings = [rating if rating is not None else mean_score for rating in all_ratings]

        # Sort the files by the ratings
        all_files = [file_path for _, file_path in sorted(zip(all_ratings, all_files), reverse=True)]
      # Sort images by the text query
      else:
        show_search_status(f"Extracting embeddings")
        embeds_img = image_search_engine.process_images(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        embeds_text = image_search_engine.process_text(text_query)
        scores = image_search_engine.compare(embeds_img, embeds_text)

        # Create a list of indices sorted by their corresponding score
        show_search_status(f"Sorting by relevance")
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]

    #all_files = sorted(all_files, key=os.path.basename)


    # Extracting metadata for relevant batch of images
    show_search_status(f"Extracting metadata for relevant batch of images")
    page_files = all_files[pagination:limit]
    page_hashes = [image_search_engine.cached_file_hash.get_file_hash(file_path) for file_path in page_files]

    # Extract DB data for the relevant batch of images
    image_db_items = db_models.ImagesLibrary.query.filter(db_models.ImagesLibrary.hash.in_(page_hashes)).all()
    image_db_items_map = {item.hash: item for item in image_db_items}

    # Check if there images without model rating
    no_model_rating_images = [os.path.join(media_directory, item.file_path) for item in image_db_items if item.model_rating is None]
    print('no_model_rating_images size', len(no_model_rating_images))

    if len(no_model_rating_images) > 0:
      # Update the model ratings of all current images
      update_model_ratings(no_model_rating_images)

    for ind, full_path in enumerate(page_files):
      file_path = os.path.relpath(full_path, media_directory)
      basename = os.path.basename(full_path)
      file_size = os.path.getsize(full_path)
      file_hash = page_hashes[ind]

      image_metadata = image_search_engine.cached_metadata.get_metadata(full_path, file_hash)
      resolution = image_metadata.get('resolution')  # Returns a tuple (width, height)
    
      user_rating = None
      model_rating = None

      if file_hash in image_db_items_map:
        image_db_item = image_db_items_map[file_hash]
        user_rating = image_db_item.user_rating
        model_rating = image_db_item.model_rating

      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": file_path,
        "base_name": basename,
        "hash": file_hash,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "file_size": convert_size(file_size),
        "resolution": f"{resolution[0]}x{resolution[1]}" if resolution else "N/A",
      }
      files_data.append(data)

    # Save all extracted metadata to the cache
    image_search_engine.cached_metadata.save_metadata_cache()
                 
    # Extract subfolders structure from the path into a dict
    folders = get_folder_structure(media_directory, cfg.images.media_formats)

    # Extract main folder name
    main_folder_name = os.path.basename(os.path.normpath(media_directory))
    folders['name'] = main_folder_name

    print(folders)
    
    socketio.emit('emit_images_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders})

    show_search_status(f'{len(all_files)} images processed in {time.time() - start_time:.4f} seconds.')

  @socketio.on('emit_images_page_move_files')
  def move_files(data):
    files = data['files']
    target_folder = data['target_folder']

    for file_path in files:
      target_path = os.path.join(target_folder, os.path.basename(file_path))

      # Check if the target path already exists add a suffix if necessary
      if os.path.exists(target_path):
        # If it is the same file, skip
        if os.path.samefile(file_path, target_path):
          continue

        file_name, file_extension = os.path.splitext(target_path)
        counter = 1
        new_file_name = f"{file_name}_{counter}{file_extension}"
        while os.path.exists(new_file_name):
          counter += 1
          new_file_name = f"{file_name}_{counter}{file_extension}"
        target_path = new_file_name

      os.rename(file_path, target_path)

  @socketio.on('emit_images_page_open_file_in_folder')
  def open_file_in_folder(file_path):
    file_manager.open_file_in_folder(file_path)

  @socketio.on('emit_images_page_send_file_to_trash')
  def send_file_to_trash(file_path):
    if os.path.isfile(file_path):
      send2trash.send2trash(file_path)
      print(f"File '{file_path}' sent to trash.")
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
      print(f"Error: File '{file_path}' does not exist.")  

  @socketio.on('emit_images_page_send_files_to_trash')
  def send_files_to_trash(files):
    for file_path in files:
      send_file_to_trash(file_path)

  @socketio.on('emit_images_page_set_image_rating')
  def set_image_rating(data):
    image_hash = data['hash'] 
    image_rating = data['rating']
    image_path = data['file_path']
    
    print('Set image rating:', image_hash, image_rating)

    image_db_item = db_models.ImagesLibrary.query.filter_by(hash=image_hash).first()

    if image_db_item is None:  
      # Create new instance us there is no image in the database

      image_data = {
        "hash": image_hash,
        "file_path": image_path,
        "user_rating": float(image_rating),
        "user_rating_date": datetime.datetime.now()
      }

      image_db_item = db_models.ImagesLibrary(**image_data)
      db_models.db.session.add(image_db_item)
      db_models.db.session.commit()
    else: 
      # Update the existing instance

      image_db_item.file_path = image_path # in case the file was moved
      image_db_item.user_rating = float(image_rating)
      image_db_item.user_rating_date = datetime.datetime.now()
      db_models.db.session.commit()

  @socketio.on('emit_images_page_get_path_to_media_folder')
  def get_path_to_media_folder():
    nonlocal media_directory
    socketio.emit('emit_images_page_show_path_to_media_folder', cfg.images.media_directory)

  @socketio.on('emit_images_page_update_path_to_media_folder')
  def update_path_to_media_folder(new_path):
    nonlocal media_directory
    print('Update path to media folder:', new_path)
    
    cfg.images.media_directory = new_path

    # Update the configuration file
    with open(os.path.join(data_folder, 'Anagnorisis-app', 'config.yaml'), 'w') as file:
      OmegaConf.save(cfg, file)

    media_directory = os.path.join(data_folder, cfg.images.media_directory)
    socketio.emit('emit_images_page_show_path_to_media_folder', cfg.images.media_directory)

    # Show files in new folder
    #get_files({})

  @socketio.on('emit_images_page_get_image_metadata_file_content')
  def get_image_metadata_file_content(file_path):
      """
      Reads the content of the .meta file associated with an image.
      If the file does not exist, returns an empty string.
      """
      nonlocal media_directory
      full_image_path = os.path.join(media_directory, file_path)
      metadata_file_path = full_image_path + ".meta"
      
      content = ""
      try:
          if os.path.exists(metadata_file_path):
              with open(metadata_file_path, 'r', encoding='utf-8') as f:
                  content = f.read()
          print(f"Read metadata for {file_path}")
      except Exception as e:
          print(f"Error reading metadata for {file_path}: {e}")
          # Optionally, emit an error status to the frontend
          socketio.emit('emit_images_page_show_search_status', f"Error reading metadata: {e}")
      
      socketio.emit('emit_images_page_show_image_metadata_content', {"content": content, "file_path": file_path})

  @socketio.on('emit_images_page_save_image_metadata_file_content')
  def save_image_metadata_file_content(data):
      """
      Saves the provided content to the .meta file associated with an image.
      """
      nonlocal media_directory
      file_path = data['file_path']
      metadata_content = data['metadata_content']
      
      full_image_path = os.path.join(media_directory, file_path)
      metadata_file_path = full_image_path + ".meta"

      try:
          # Ensure the directory exists before writing the file
          os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
          with open(metadata_file_path, 'w', encoding='utf-8') as f:
              f.write(metadata_content)
          print(f"Saved metadata for {file_path}")
          socketio.emit('emit_images_page_show_search_status', f"Metadata saved for {os.path.basename(file_path)}")
      except Exception as e:
          print(f"Error saving metadata for {file_path}: {e}")
          socketio.emit('emit_images_page_show_search_status', f"Error saving metadata: {e}")