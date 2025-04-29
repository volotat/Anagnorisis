import os
import time
from flask import send_from_directory

import math
import numpy as np
from omegaconf import OmegaConf
import datetime
import torch
import gc

import pages.file_manager as file_manager
import pages.music.db_models as db_models

from pages.music.engine import MusicSearch, MusicEvaluator

from pages.utils import convert_size, convert_length, time_difference

from pages.recommendation_engine import sort_files_by_recommendation

from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback

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

def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
  if cfg.music.media_directory is None:
    print("Music media folder is not set.")
    media_directory = None
  else:
    media_directory = os.path.join(data_folder, cfg.music.media_directory)

  print('Music media_directory:', media_directory)

  MusicSearch.initiate(models_folder=cfg.main.models_path, cache_folder=cfg.main.cache_path)
  cached_file_list = MusicSearch.cached_file_list
  cached_file_hash = MusicSearch.cached_file_hash
  cached_metadata = MusicSearch.cached_metadata

  music_evaluator = MusicEvaluator(embedding_dim=MusicSearch.embedding_dim) #src.scoring_models.Evaluator(embedding_dim=768)
  music_evaluator.load(os.path.join(cfg.main.models_path, 'music_evaluator.pt'))

  def show_search_status(status):
    socketio.emit('emit_music_page_show_search_status', status)

  embedding_gathering_callback = EmbeddingGatheringCallback(show_search_status)

  def show_search_status(status):
    socketio.emit('emit_music_page_show_search_status', status)

  def update_none_hashes_in_db(files_list, all_hashes):
    """
    Update the hash for all instances in the DB that do not have a hash.
    """
    for file_path, file_hash in zip(files_list, all_hashes):
        # Get the relative file path
        relative_file_path = os.path.relpath(file_path, media_directory)
        
        # Query the database for the file with the given file path and no hash
        db_item = db_models.MusicLibrary.query.filter_by(file_path=relative_file_path, hash=None).first()
        
        if db_item:
            # Update the hash
            db_item.hash = file_hash
            db_models.db.session.add(db_item)
    
    # Commit the changes to the database
    db_models.db.session.commit()

  def update_model_ratings(files_list):
    print('update_model_ratings')

    # filter out files that already have a rating in the DB
    files_list_hash_map = {file_path: cached_file_hash.get_file_hash(file_path) for file_path in files_list}
    hash_list = list(files_list_hash_map.values())

    # Fetch rated files from the database in a single query
    rated_files_db_items = db_models.MusicLibrary.query.filter(
      db_models.MusicLibrary.hash.in_(hash_list),
      db_models.MusicLibrary.model_rating.isnot(None),
      db_models.MusicLibrary.model_hash.is_(music_evaluator.hash)
    ).all()

    # Create a list of hashes for rated files
    rated_files_hashes = {item.hash for item in rated_files_db_items}

    # Filter out files that already have a rating in the database
    filtered_files_list = [file_path for file_path, file_hash in files_list_hash_map.items() if file_hash not in rated_files_hashes]
    if not filtered_files_list: return
    

    # Rate all files in case they are not rated or model was updated
    embeddings = MusicSearch.process_audio(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
    model_ratings = music_evaluator.predict(embeddings)

    # Update the model ratings in the database
    show_search_status(f"Updating model ratings of files...") 
    new_items = []
    update_items = []
    last_shown_time = 0
    for ind, full_path in enumerate(filtered_files_list):
      print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} files.")

      hash = files_list_hash_map[full_path]
      model_rating = model_ratings[ind].item()

      music_db_item = db_models.MusicLibrary.query.filter_by(hash=hash).first()
      if music_db_item:
        music_db_item.model_rating = model_rating
        music_db_item.model_hash = music_evaluator.hash
        update_items.append(music_db_item)
      else:
        file_data = {
            "hash": hash,
            "file_path": os.path.relpath(full_path, media_directory),
            "model_rating": model_rating,
            "model_hash": music_evaluator.hash
        }
        new_items.append(db_models.MusicLibrary(**file_data))

      current_time = time.time()
      if current_time - last_shown_time >= 1:
        show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} files.")
        last_shown_time = current_time   

    # Bulk update and insert
    if update_items:
        db_models.db.session.bulk_save_objects(update_items)
    if new_items:
        db_models.db.session.bulk_save_objects(new_items)

    # Commit the transaction
    db_models.db.session.commit()

  # necessary to allow web application access to music files
  @app.route('/music_files/<path:filename>')
  def serve_media(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_music_page_get_files')
  def get_files(input_data):
    nonlocal media_directory

    if media_directory is None:
      show_search_status("Music media folder is not set.")
      return

    start_time = time.time()

    path = input_data.get('path', '')
    pagination = input_data.get('pagination', 0)
    limit = input_data.get('limit', 100)
    text_query = input_data.get('text_query', None)
    
    files_data = []
    if path == "":
      current_path = media_directory
    else:
      current_path = os.path.abspath(os.path.join(media_directory, '..', path))

    
    folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
    print('folder_path', folder_path)

    show_search_status(f"Searching for music files in '{folder_path}'.")

    all_files = []
    
    # Walk with cache 1.5s for 66k files
    all_files = cached_file_list.get_all_files(current_path, cfg.music.media_formats)

    show_search_status(f"Gathering hashes for {len(all_files)} files.")
    
    # Initialize the last shown time
    last_shown_time = 0

    # Get the hash of each file
    all_hashes = []
    for ind, file_path in enumerate(all_files):
      current_time = time.time()
      if current_time - last_shown_time >= 1:
        show_search_status(f"Gathering hashes for {ind+1}/{len(all_files)} files.")
        last_shown_time = current_time
      
      all_hashes.append(cached_file_hash.get_file_hash(file_path))

    cached_file_hash.save_hash_cache()

    # Update the hashes in the database if there are instances without a hash
    show_search_status(f"Update hashes for imported files...")
    update_none_hashes_in_db(all_files, all_hashes)


    # Sort music files by text or file query
    show_search_status(f"Sorting music files by {text_query}")
    if text_query and len(text_query) > 0:
      # If there is an music file with the path of the query, sort files by similarity to that file
      if os.path.isfile(text_query) and text_query.lower().endswith(tuple(cfg.music.media_formats)):
        target_path = text_query
        show_search_status(f"Extracting embeddings")
        embeds_img = MusicSearch.process_audio(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        target_emb = MusicSearch.process_audio([target_path], callback=embedding_gathering_callback, media_folder=media_directory)

        show_search_status(f"Computing distances between embeddings")
        scores = torch.cdist(embeds_img, target_emb, p=2).cpu().detach().numpy() 

        # Create a list of indices sorted by their corresponding score
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]
      # Sort music by file size
      elif text_query.lower().strip() == "file size":
        show_search_status(f"Gathering file sizes for sorting...") # Initial status message
        all_files = sorted(all_files, key=os.path.getsize)
      # Sort music by length
      elif text_query.lower().strip() == "length":
        show_search_status(f"Gathering resolutions for sorting...") # Initial status message

        durations = {}
        progress_callback = SortingProgressCallback(show_search_status, operation_name="Gathering music duration ") # Create callback
        for ind, full_path in enumerate(all_files):
          file_path = os.path.relpath(full_path, media_directory)
          file_hash = all_hashes[ind]

          file_metadata = cached_metadata.get_metadata(full_path, file_hash)
          if 'duration' not in file_metadata:
            raise Exception(f"Duration not found for file: {file_path}")
          
          durations[full_path] = file_metadata['duration']
          progress_callback(ind + 1, len(all_files)) # Update progress

        # Check if there is none value in the durations and print the files with None duration
        none_durations = [file_path for file_path, duration in durations.items() if duration is None]
        if none_durations:
          print("Files with None duration:", none_durations)

        all_files = sorted(all_files, key=lambda x: durations[x])
      # Sort music by duplicates
      elif text_query.lower().strip() == "similarity":
        show_search_status(f"Extracting embeddings")
        embeds_img = MusicSearch.process_audio(all_files, callback=embedding_gathering_callback, media_folder=media_directory).cpu().detach().numpy() 

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
      # Sort music randomly
      elif text_query.lower().strip() == "random":
        np.random.shuffle(all_files)
      # Sort music by rating
      elif text_query.lower().strip() == "rating": 
        # Update the model ratings of all current files
        update_model_ratings(all_files)

        # Get all ratings from the database
        items = db_models.MusicLibrary.query.filter(db_models.MusicLibrary.hash.in_(all_hashes)).all()

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
      # Sort music with recommendation engine 
      elif text_query.lower().strip() == "recommendation":
        # Update the model ratings of all current files
        update_model_ratings(all_files)
        
        music_data = db_models.MusicLibrary.query.with_entities(
            db_models.MusicLibrary.hash,
            db_models.MusicLibrary.user_rating,
            db_models.MusicLibrary.model_rating,
            db_models.MusicLibrary.full_play_count,
            db_models.MusicLibrary.skip_count,
            db_models.MusicLibrary.last_played
        ).filter(db_models.MusicLibrary.hash.in_(all_hashes)).all()

        keys = ['hash', 'user_rating', 'model_rating', 'full_play_count', 'skip_count', 'last_played']
        music_data_dict = [dict(zip(keys, row)) for row in music_data]
        all_files, scores = sort_files_by_recommendation(all_files, music_data_dict)
        #for m, s in zip(all_files, scores):
        #    print(f"{m} => recommendation score: {s}")
      # Sort music by the text query
      else:
        show_search_status(f"Extracting embeddings")
        embeds_audio = MusicSearch.process_audio(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        embeds_text = MusicSearch.process_text(text_query)
        scores = MusicSearch.compare(embeds_audio, embeds_text)

        # Create a list of indices sorted by their corresponding score
        show_search_status(f"Sorting by relevance")
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]

    #all_files = sorted(all_files, key=os.path.basename)

    
    # Extracting metadata for relevant batch of music files
    show_search_status(f"Extracting metadata for relevant batch of files.")
    page_files = all_files[pagination:limit]
    page_hashes = [cached_file_hash.get_file_hash(file_path) for file_path in page_files]

    # Extract DB data for the relevant batch of music files
    music_db_items = db_models.MusicLibrary.query.filter(db_models.MusicLibrary.hash.in_(page_hashes)).all()
    music_db_items_map = {item.hash: item for item in music_db_items}

    # Check if there files without model rating
    no_model_rating_files = [os.path.join(media_directory, item.file_path) for item in music_db_items if item.model_rating is None]
    print('no_model_rating_files size', len(no_model_rating_files))
    
    if len(no_model_rating_files) > 0:
      # Update the model ratings of all current files
      update_model_ratings(no_model_rating_files)

    for ind, full_path in enumerate(page_files):
      file_path = os.path.relpath(full_path, media_directory)
      basename = os.path.basename(full_path)
      file_size = os.path.getsize(full_path)
      file_hash = page_hashes[ind]

      audiofile_data = cached_metadata.get_metadata(full_path, file_hash)

      
      #resolution = get_image_resolution(full_path)  # Returns a tuple (width, height)
    
      user_rating = None
      model_rating = None
      last_played = "Never"

      if file_hash in music_db_items_map:
        music_db_item = music_db_items_map[file_hash]
        user_rating = music_db_item.user_rating
        model_rating = music_db_item.model_rating

        # convert datetime to string
        if music_db_item.last_played:
          last_played_timestamp = music_db_item.last_played.timestamp() if music_db_item.last_played else None
          last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())
          
      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": file_path,
        "base_name": basename,
        "hash": file_hash,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "audiofile_data": audiofile_data,
        "file_size": convert_size(file_size),
        "length": convert_length(audiofile_data['duration']),
        "last_played": last_played
      }
      files_data.append(data)
    
    # Save all extracted metadata to the cache
    cached_metadata.save_metadata_cache()

    # Extract subfolders structure from the path into a dict
    folders = file_manager.get_folder_structure(media_directory, cfg.music.media_formats)

    # Return "No files in the directory" if the path not exist or empty.
    if not folders:
      show_search_status(f"No files in the directory.")
      socketio.emit('emit_music_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": 0, "folders": folders, "all_files_paths": []})
      return

    # Extract main folder name
    main_folder_name = os.path.basename(os.path.normpath(media_directory))
    folders['name'] = main_folder_name

    #print(folders)

    all_files_paths = [os.path.relpath(file_path, media_directory) for file_path in all_files]
    
    socketio.emit('emit_music_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders, "all_files_paths": all_files_paths})

    show_search_status(f'{len(all_files)} files processed in {time.time() - start_time:.4f} seconds.')

  @socketio.on('emit_music_page_get_song_details')
  def get_song_details(data):
    file_path = data.get('file_path', '')

    full_path = os.path.join(media_directory, file_path)
    file_hash = cached_file_hash.get_file_hash(full_path)
    audiofile_data = cached_metadata.get_metadata(full_path, file_hash)
    audiofile_data['hash'] = cached_file_hash.get_file_hash(full_path)
    db_item = db_models.MusicLibrary.query.filter_by(hash=audiofile_data['hash']).first()
    audiofile_data['user_rating'] = db_item.user_rating
    audiofile_data['model_rating'] = db_item.model_rating
    audiofile_data['file_path'] = file_path

    socketio.emit('emit_music_page_show_song_details', audiofile_data)

  @socketio.on('emit_music_page_set_song_play_rate')
  def request_new_song(data):
    cur_song_hash = None
    song_score_change = None
    
    if len(data) > 0:
      cur_song_hash = data['hash']
      skip_score_change = data['skip_score_change']

    if cur_song_hash is not None:
      print('Set song play rate:', cur_song_hash, skip_score_change)
      song = db_models.MusicLibrary.query.filter_by(hash=cur_song_hash).first()
      if skip_score_change ==  1:  song.full_play_count += 1
      if skip_score_change == -1:  song.skip_count += 1
      
      #song.skip_score += skip_score_change
      db_models.db.session.commit()

  @socketio.on('emit_music_page_set_song_rating')
  def set_song_rating(data):
    song_hash = data['hash'] 
    song_score = data['score']

    print('Set song rating:', song_hash, song_score)

    song = db_models.MusicLibrary.query.filter_by(hash=song_hash).first()
    song.user_rating = float(song_score)
    song.user_rating_date = datetime.datetime.now()
    db_models.db.session.commit()

  @socketio.on('emit_music_page_update_song_info')
  def update_song_info(data):
    # NOTE: It is very important to update the hash of the file in the database after the 
    # metadata has been updated to not lose user rating of the song and other connected data
    
    #print('update_song_info', data)
    #edit_lyrics(data['file_path'], data['lyrics'])
    pass

  @socketio.on('emit_music_page_song_start_playing')
  def song_start_playing(song_hash):
    song = db_models.MusicLibrary.query.filter_by(hash=song_hash).first()
    song.last_played = datetime.datetime.now()
    db_models.db.session.commit()

  @socketio.on('emit_music_page_get_path_to_media_folder')
  def get_path_to_media_folder():
    nonlocal media_directory
    socketio.emit('emit_music_page_show_path_to_media_folder', cfg.music.media_directory)

  @socketio.on('emit_music_page_update_path_to_media_folder')
  def update_path_to_media_folder(new_path):
    nonlocal media_directory
    cfg.music.media_directory = new_path

    # Update the configuration file
    with open(os.path.join(data_folder, 'Anagnorisis-app', 'config.yaml'), 'w') as file:
      OmegaConf.save(cfg, file)

    media_directory = os.path.join(data_folder, cfg.music.media_directory)
    socketio.emit('emit_music_page_show_path_to_media_folder', cfg.music.media_directory)

  @socketio.on('emit_music_page_open_file_in_folder')
  def open_file_in_folder(file_path):
    file_manager.open_file_in_folder(file_path)