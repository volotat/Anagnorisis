
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
from pages.common_filters import CommonFilters

from pages.utils import convert_size, convert_length, time_difference

# EVENTS:
# Incoming (handled with @socketio.on):

# emit_images_page_get_files
# emit_images_page_move_files
# emit_images_page_open_file_in_folder
# emit_images_page_send_file_to_trash
# emit_images_page_send_files_to_trash
# emit_images_page_set_image_rating
# emit_images_page_get_path_to_media_folder
# emit_images_page_update_path_to_media_folder
# emit_images_page_get_image_metadata_file_content
# emit_images_page_save_image_metadata_file_content

# Also should be:
# emit_images_page_get_folders

# Outgoing (emitted with socketio.emit):

# emit_images_page_show_search_status
# emit_images_page_show_files
# emit_images_page_show_path_to_media_folder
# emit_images_page_show_image_metadata_content

def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
  if cfg.images.media_directory is None:
    print("Images media folder is not set.")
    media_directory = None
  else:
    # media_directory = os.path.join(data_folder, cfg.images.media_directory)
    media_directory = cfg.images.media_directory
  
  images_search_engine = ImageSearch(cfg=cfg)
  images_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)

  # cached_file_hash = images_search_engine.cached_file_hash
  # cached_metadata = images_search_engine.cached_metadata

  image_evaluator = ImageEvaluator() #src.scoring_models.Evaluator(embedding_dim=768)
  image_evaluator.load(os.path.join(cfg.main.personal_models_path, 'image_evaluator.pt'))

  def show_search_status(status):
    socketio.emit('emit_images_page_show_search_status', status)

  def gather_resolutions(all_files, all_hashes, media_directory):
    resolutions = {}
    progress_callback = SortingProgressCallback(show_search_status, operation_name="Gathering images resolution") # Create callback

    for ind, full_path in enumerate(all_files):
      file_path = os.path.relpath(full_path, media_directory)
      # file_hash = all_hashes[ind]

      image_metadata = images_search_engine.get_metadata(full_path)
      if 'resolution' not in image_metadata:
        raise Exception(f"Resolution not found for image: {file_path}")
      
      resolutions[full_path] = image_metadata['resolution']
      progress_callback(ind + 1, len(all_files)) # Update progress

    return resolutions
  
  common_socket_events = CommonSocketEvents(socketio)

  embedding_gathering_callback = EmbeddingGatheringCallback(show_search_status)

  images_file_manager = file_manager.FileManager(
    cfg=cfg,
    media_directory=media_directory,
    engine=images_search_engine,
    module_name="images",
    media_formats=cfg.images.media_formats,
    socketio=socketio,
    db_schema=db_models.ImagesLibrary,
  )
  

  def update_model_ratings(files_list):
    print(files_list)
    # filter out files that already have a rating in the DB
    files_list_hash_map = {file_path: images_search_engine.get_file_hash(file_path) for file_path in files_list}
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
    embeddings = images_search_engine.process_images(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
    model_ratings = image_evaluator.predict(embeddings)

    # Update the model ratings in the database
    show_search_status(f"Updating model ratings of images...") 
    new_items = []
    update_items = []
    for ind, full_path in enumerate(filtered_files_list):
      # print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} images.")

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

      show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} images.")  

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
  
  @socketio.on('emit_images_page_get_folders')  
  def get_folders(data):
    path = data.get('path', '')
    return images_file_manager.get_folders(path)

  @socketio.on('emit_images_page_get_files')
  def get_files(input_data):
    # Define domain specific filters 
    def filter_by_resolution(all_files, text_query):
      show_search_status(f"Gathering resolutions for sorting...") # Initial status message
      all_hashes = [images_search_engine.get_file_hash(file_path) for file_path in all_files]
      resolutions = gather_resolutions(all_files, all_hashes, media_directory)
      # all_files_sorted = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])
      scores = [resolutions[x][0] * resolutions[x][1] for x in all_files]
      return scores

    def filter_by_proportion(all_files, text_query):
      show_search_status(f"Gathering resolutions for sorting...") # Initial status message
      all_hashes = [images_search_engine.get_file_hash(file_path) for file_path in all_files]
      resolutions = gather_resolutions(all_files, all_hashes, media_directory)
      #all_files_sorted = sorted(all_files, key=lambda x: resolutions[x][0] / resolutions[x][1])
      scores = [resolutions[x][0] / resolutions[x][1] for x in all_files]
      return scores

    # Create common filters instance
    common_filters = CommonFilters(
        engine=images_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.ImagesLibrary,
        update_model_ratings_func=update_model_ratings
    )

    # Get parameters
    # path = input_data.get('path', '')
    # pagination = input_data.get('pagination', 0)
    # limit = input_data.get('limit', 100)
    # text_query = input_data.get('text_query', None)
    # seed = input_data.get('seed', None)

    # Define available filters
    filters = {
      "by_file": common_filters.filter_by_file, # special sorting case when file path used as query
      "by_text": common_filters.filter_by_text, # special sorting case when text used as query, i.e. all other cases wasn't triggered
      "file_size": common_filters.filter_by_file_size,
      "resolution": filter_by_resolution,
      "proportion": filter_by_proportion,
      "similarity": common_filters.filter_by_similarity, 
      "random": common_filters.filter_by_random, 
      "rating": common_filters.filter_by_rating, 
      # "recommendation": filter_by_recommendation
    }

    # Define a method to gather domain specific file information
    def get_file_info(full_path, file_hash):
      file_path = os.path.relpath(full_path, media_directory)
      basename = os.path.basename(full_path)
      file_size = os.path.getsize(full_path)

      image_metadata = images_search_engine.get_metadata(full_path)
      resolution = image_metadata.get('resolution')  # Returns a tuple (width, height)
    
      user_rating = None
      model_rating = None

      db_item = db_models.ImagesLibrary.query.filter_by(hash=file_hash).first()

      if db_item:
        user_rating = db_item.user_rating
        model_rating = db_item.model_rating
        file_data = images_search_engine.get_metadata(full_path)
      else:
        raise Exception(f"File '{full_path}' with hash '{file_hash}' not found in the database.")

      return {
        "file_path": file_path,
        "base_name": basename,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "file_size": convert_size(file_size),
        "resolution": f"{resolution[0]}x{resolution[1]}" if resolution else "N/A",
        "file_data": file_data
      }

    # path, pagination, limit, text_query, seed, filters, get_file_info, update_model_ratings
    input_params = input_data.copy()
    input_params.update({
      "filters": filters,
      "get_file_info": get_file_info,
      "update_model_ratings": update_model_ratings,
    })
    return images_file_manager.get_files(**input_params)

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