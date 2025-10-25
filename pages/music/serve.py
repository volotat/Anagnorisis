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

from pages.socket_events import CommonSocketEvents
from pages.common_filters import CommonFilters

# EVENTS:
# Incoming (handled with @socketio.on):

# emit_music_page_get_files
# emit_music_page_get_song_details
# emit_music_page_set_song_play_rate
# emit_music_page_set_song_rating
# emit_music_page_update_song_info
# emit_music_page_song_start_playing
# emit_music_page_get_path_to_media_folder
# emit_music_page_update_path_to_media_folder
# emit_music_page_open_file_in_folder

# Outgoing (emitted with socketio.emit):

# emit_music_page_show_search_status
# emit_music_page_show_files
# emit_music_page_show_song_details
# emit_music_page_show_path_to_media_folder

def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
  if cfg.music.media_directory is None:
    print("Music media folder is not set.")
    media_directory = None
  else:
    # media_directory = os.path.join(data_folder, cfg.music.media_directory)
    media_directory = cfg.music.media_directory

  print('Music media_directory:', media_directory)

  music_search_engine = MusicSearch(cfg=cfg) 
  music_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)

  # cached_file_hash = music_search_engine.cached_file_hash
  # cached_metadata = music_search_engine.cached_metadata

  music_evaluator = MusicEvaluator(embedding_dim=music_search_engine.embedding_dim) #src.scoring_models.Evaluator(embedding_dim=768)

  print('Loading music evaluator model from', os.path.join(cfg.main.personal_models_path, 'music_evaluator.pt'))
  music_evaluator.load(os.path.join(cfg.main.personal_models_path, 'music_evaluator.pt'))

  common_socket_events = CommonSocketEvents(socketio)
  # def show_search_status(status):
  #   socketio.emit('emit_music_page_show_search_status', status)

  embedding_gathering_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

  music_file_manager = file_manager.FileManager(
    cfg=cfg,
    media_directory=media_directory,
    engine=music_search_engine,
    module_name="music",
    media_formats=cfg.music.media_formats,
    socketio=socketio,
    db_schema=db_models.MusicLibrary,
  )

  # def show_search_status(status):
  #   socketio.emit('emit_music_page_show_search_status', status)

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
    files_list_hash_map = {file_path: music_search_engine.get_file_hash(file_path) for file_path in files_list}
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
    embeddings = music_search_engine.process_audio(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
    model_ratings = music_evaluator.predict(embeddings)

    # Update the model ratings in the database
    common_socket_events.show_search_status(f"Updating model ratings of files...") 
    new_items = []
    update_items = []
    last_shown_time = 0
    for ind, full_path in enumerate(filtered_files_list):
      # print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} files.")

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
        common_socket_events.show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} files.")
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

  @socketio.on('emit_music_page_get_folders')  
  def get_folders(data):
    path = data.get('path', '')
    return music_file_manager.get_folders(path)
  
  def debug_data_types(data, path=""):
    """
    Recursively traverses a data structure and prints the type of each value.
    """
    if isinstance(data, dict):
        for key, value in data.items():
            debug_data_types(value, f"{path}.{key}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            debug_data_types(item, f"{path}[{i}]")
    else:
        # Print the path and type of the value
        print(f"Path: {path:<70} | Type: {type(data)}")

  @socketio.on('emit_music_page_get_files')
  def get_files(input_data):
    nonlocal media_directory

    # Define domain specific filters
    def filter_by_length(all_files, text_query):
      common_socket_events.show_search_status(f"Gathering resolutions for sorting...") # Initial status message

      all_hashes = [music_search_engine.get_file_hash(file_path) for file_path in all_files]

      durations = {}
      progress_callback = SortingProgressCallback(common_socket_events.show_search_status, operation_name="Gathering music duration ") # Create callback
      for ind, full_path in enumerate(all_files):
        file_path = os.path.relpath(full_path, media_directory)
        # file_hash = all_hashes[ind]

        file_metadata = music_search_engine.get_metadata(full_path)
        if 'duration' not in file_metadata:
          raise Exception(f"Duration not found for file: {file_path}")
        
        durations[full_path] = file_metadata['duration']
        progress_callback(ind + 1, len(all_files)) # Update progress

      # Check if there is none value in the durations and print the files with None duration
      none_durations = [file_path for file_path, duration in durations.items() if duration is None]
      if none_durations:
        print("Files with None duration:", none_durations)

      return durations
    
    def filter_by_recommendation(all_files, text_query):
      # Update the model ratings of all current files
      update_model_ratings(all_files)
      
      all_hashes = [music_search_engine.get_file_hash(file_path) for file_path in all_files]
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
      scores = sort_files_by_recommendation(all_files, music_data_dict)

      return scores
    
    # Create common filters instance
    common_filters = CommonFilters(
        engine=music_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.MusicLibrary,
        update_model_ratings_func=update_model_ratings
    )

    # Get parameters
    # path = input_data.get('path', '')
    # pagination = input_data.get('pagination', 0)
    # limit = input_data.get('limit', 100)
    # text_query = input_data.get('text_query', None)
    # seed = input_data.get('seed', None)
    # mode = input_data.get('mode', 'file-name') # can be 'file-name', 'semantic-content', 'semantic-metadata'
    
    # Define available filters
    filters = {
      "by_file": common_filters.filter_by_file, # special sorting case when file path used as query
      "by_text": common_filters.filter_by_text, # special sorting case when text used as query, i.e. all other cases wasn't triggered
      "file_size": common_filters.filter_by_file_size,
      "length": filter_by_length,
      "similarity": common_filters.filter_by_similarity, 
      "random": common_filters.filter_by_random, 
      "rating": common_filters.filter_by_rating, 
      "recommendation": filter_by_recommendation
    }

    # Define a method to gather domain specific file information
    def get_file_info(full_path, file_hash):
      db_item = db_models.MusicLibrary.query.filter_by(hash=file_hash).first()
          
      if db_item:
        # convert datetime to string
        last_played = "Never"
        if db_item.last_played:
            last_played_timestamp = db_item.last_played.timestamp()
            last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())

        audiofile_data = music_search_engine.get_metadata(full_path)
      else:
        raise Exception(f"File '{full_path}' with hash '{file_hash}' not found in the database.")

      return {
        "user_rating": db_item.user_rating,
        "model_rating": db_item.model_rating,
        "full_play_count": db_item.full_play_count,
        "skip_count": db_item.skip_count,
        "last_played": last_played,
        "audiofile_data": audiofile_data,
        "length": convert_length(audiofile_data['duration']),
      }

    # path, pagination, limit, text_query, seed, filters, get_file_info, update_model_ratings, mode
    input_params = input_data.copy()
    input_params.update({
      "filters": filters,
      "get_file_info": get_file_info,
      "update_model_ratings": update_model_ratings,
    })
    return music_file_manager.get_files(**input_params)

  @socketio.on('emit_music_page_get_song_details')
  def get_song_details(data):
    file_path = data.get('file_path', '')

    full_path = os.path.join(media_directory, file_path)
    audiofile_data = music_search_engine.get_metadata(full_path)
    audiofile_data['hash'] = music_search_engine.get_file_hash(full_path)
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