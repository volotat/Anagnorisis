import os
import time
import base64
from tinytag import TinyTag
from flask import send_from_directory

import math
import numpy as np
from omegaconf import OmegaConf

import pages.file_manager as file_manager
import pages.music.db_models as db_models

from pages.music.engine import MusicSearch, MusicEvaluator

from pages.utils import convert_size, convert_length


class EmbeddingGatheringCallback:
  def __init__(self, show_status_function=None):
    self.last_shown_time = 0
    self.start_time = time.time()
    self.show_status_function = show_status_function

  def __call__(self, num_extracted, num_total):
    current_time = time.time()
    if current_time - self.last_shown_time >= 1:
      # Calculate the percentage of processed files
      percent = (num_extracted / num_total) * 100

      # Show the status
      self.show_status_function(f"Extracted embeddings for {num_extracted}/{num_total} ({percent:.2f}%) files.")
      self.last_shown_time = current_time

def get_audiofile_data(file_path, url_path):
  """
  tag.album         # album as string
  tag.albumartist   # album artist as string
  tag.artist        # artist name as string
  tag.audio_offset  # number of bytes before audio data begins
  tag.bitdepth      # bit depth for lossless audio
  tag.bitrate       # bitrate in kBits/s
  tag.comment       # file comment as string
  tag.composer      # composer as string 
  tag.disc          # disc number
  tag.disc_total    # the total number of discs
  tag.duration      # duration of the song in seconds
  tag.filesize      # file size in bytes
  tag.genre         # genre as string
  tag.samplerate    # samples per second
  tag.title         # title of the song
  tag.track         # track number as string
  tag.track_total   # total number of tracks as string
  tag.year          # year or date as string
  """

  metadata = {
    'file_path': file_path,
    #'url_path': url_path,
    #'hash': calculate_audiodata_hash(file_path),
  }

  

  #audiofile = eyed3.load(file_path)

  tag = TinyTag.get(file_path, image=True)

  metadata['title'] = tag.title or "N/A"
  metadata['artist'] = tag.artist or "N/A"
  metadata['album'] = tag.album or "N/A"
  metadata['track_num'] = tag.track if tag.track else "N/A"
  metadata['genre'] = tag.genre if tag.genre else "N/A"
  metadata['date'] = str(tag.year) if tag.year else "N/A"

  metadata['duration'] = tag.duration #(seconds)
  metadata['bitrate'] = tag.bitrate #(kbps)

  metadata['lyrics'] = tag.extra.get('lyrics', "")

  img = tag.get_image()
  if img is not None:
    base64_image = base64.b64encode(img).decode('utf-8')

    #buffer = io.BytesIO()
    #img.save(buffer, format='PNG')
    #buffer.seek(0)
    
    #data_uri = base64.b64encode(buffer.read()).decode('ascii')
    metadata['image'] = f"data:image/png;base64,{base64_image}"
  else:
    metadata['image'] = None

    # Get all available tags and their values as a dictionary
    #tag_dict = audiofile.tag.frame_set

    # If there are multiple artists, they will be stored in a list
    #if audiofile.tag.artist:
    #    print("Artists:", ", ".join(audiofile.tag.artist))

    # If there are multiple genres, they will be stored in a list
    #if audiofile.tag.genre:
    #    print("Genres:", ", ".join(audiofile.tag.genre))

    # You can access other tag fields in a similar way

    # To print all tags and their values, you can iterate through them
    #for tag in audiofile.tag.frame_set:
    #    print(tag, ":", audiofile.tag.frame_set[tag][0])

    # If you want to access additional metadata, you can use audiofile.tag.file_info
    #print("Sample Width (bits):", audiofile.tag.file_info.sample_width)
    #print("Channel Mode:", audiofile.tag.file_info.mode)

    # To print the entire tag as a dictionary
    #print("Tag Dictionary:", audiofile.tag.frame_set)

    #for frame in audiofile.tag.frameiter(["TXXX"]):
    #  print(f"{frame.description}: {frame.text}")
  
  return metadata

def init_socket_events(socketio, app=None, cfg=None):
  media_directory = cfg.music.media_directory

  MusicSearch.initiate()
  cached_file_list = MusicSearch.cached_file_list
  cached_file_hash = MusicSearch.cached_file_hash

  music_evaluator = MusicEvaluator(embedding_dim=MusicSearch.embedding_dim) #src.scoring_models.Evaluator(embedding_dim=768)
  music_evaluator.load('./models/music_evaluator.pt')

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
        '''target_path = text_query
        show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        target_emb = ImageSearch.process_images([target_path], callback=embedding_gathering_callback, media_folder=media_directory)

        show_search_status(f"Computing distances between embeddings")
        scores = torch.cdist(embeds_img, target_emb, p=2).cpu().detach().numpy() 

        # Create a list of indices sorted by their corresponding score
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]'''
      # Sort music by file size
      elif text_query.lower().strip() == "file size":
        all_files = sorted(all_files, key=os.path.getsize)
      # Sort music by length
      elif text_query.lower().strip() == "length":
        '''if len(all_files) > 10000:
          raise Exception("Too many images to sort by resolution")
        # TODO: THIS IS VERY SLOW, NEED TO OPTIMIZE
        resolutions = {file_path: get_image_resolution(file_path) for file_path in all_files}
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])'''
      # Sort music by duplicates
      elif text_query.lower().strip() == "similarity":
        '''show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files, callback=embedding_gathering_callback, media_folder=media_directory).cpu().detach().numpy() 

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
        all_files = [file_path for _, _, _, file_path in sorted_files]'''
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
      # Sort music by the text query
      else:
        show_search_status(f"Extracting embeddings")
        embeds_img = MusicSearch.process_audio(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
        embeds_text = MusicSearch.process_text(text_query)
        scores = MusicSearch.compare(embeds_img, embeds_text)

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
      basename = os.path.basename(full_path)
      file_size = os.path.getsize(full_path)

      audiofile_data = get_audiofile_data(full_path, "")

      hash = page_hashes[ind]
      #resolution = get_image_resolution(full_path)  # Returns a tuple (width, height)
    
      user_rating = None
      model_rating = None

      if hash in music_db_items_map:
        music_db_item = music_db_items_map[hash]
        user_rating = music_db_item.user_rating
        model_rating = music_db_item.model_rating

      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": os.path.relpath(full_path, media_directory),
        "base_name": basename,
        "hash": hash,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "audiofile_data": audiofile_data,
        "file_size": convert_size(file_size),
        "length": convert_length(audiofile_data['duration']),
      }
      files_data.append(data)
    
    

    # Extract subfolders structure from the path into a dict
    folders = file_manager.get_folder_structure(media_directory, cfg.music.media_formats)

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
    audiofile_data = get_audiofile_data(full_path, "")
    audiofile_data['hash'] = cached_file_hash.get_file_hash(full_path)
    db_item = db_models.MusicLibrary.query.filter_by(hash=audiofile_data['hash']).first()
    audiofile_data['user_rating'] = db_item.user_rating
    audiofile_data['model_rating'] = db_item.model_rating
    audiofile_data['file_path'] = file_path

    socketio.emit('emit_music_page_show_song_details', audiofile_data)

  @socketio.on('emit_music_page_get_path_to_media_folder')
  def get_path_to_media_folder():
    nonlocal media_directory
    socketio.emit('emit_music_page_show_path_to_media_folder', media_directory)

  @socketio.on('emit_music_page_update_path_to_media_folder')
  def update_path_to_media_folder(new_path):
    nonlocal media_directory
    cfg.music.media_directory = new_path

    # Update the configuration file
    with open('config.yaml', 'w') as file:
      OmegaConf.save(cfg, file)

    media_directory = cfg.music.media_directory
    socketio.emit('emit_music_page_show_path_to_media_folder', media_directory)