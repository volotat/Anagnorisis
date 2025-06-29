
from flask import Flask, render_template, send_from_directory, Response, stream_with_context
import os
import sys
import glob
import subprocess
import hashlib
import numpy as np
from io import BytesIO
from PIL import Image
import math
from scipy.spatial import distance
import torch
import gc
import send2trash
import time
import datetime
from moviepy.editor import VideoFileClip

import threading
import uuid
import tempfile
import shutil

import pages.file_manager as file_manager
import pages.videos.db_models as db_models
from pages.videos.engine import VideoSearch, VideoEvaluator
from pages.recommendation_engine import sort_files_by_recommendation

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

def print_emb_extracting_status(num_extracted, num_total):
  if num_extracted % 100 == 0:
    print(f"Extracted embeddings for {num_extracted} out of {num_total} videos.")

def generate_preview(video_path, preview_path):
  """
  Generates a preview image for a given video file.

  Args:
  video_path (str): The path to the video file.
  preview_path (str): The path where the preview image will be saved.
  """
  try:
    # Load the video file
    clip = VideoFileClip(video_path)
    # Calculate the middle frame time
    middle_time = clip.duration / 2
    # Save the frame at the middle of the video as an image
    clip.save_frame(preview_path, t=middle_time)
  except Exception as e:
    print(f"Error generating preview for {video_path}: {e}")


def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
  if cfg.videos.media_directory is None:
      print("Videos media folder is not set.")
      media_directory = None
  else:
      media_directory = os.path.join(data_folder, cfg.videos.media_directory)

      # Check if media_directory exists, if not, print a warning and set to None
      if not os.path.isdir(media_directory):
          print(f"Warning: Videos media directory '{os.path.join(data_folder, cfg.videos.media_directory)}' does not exist. Setting media folder to None.")
          media_directory = None

  def show_search_status(status):
    socketio.emit('emit_videos_page_show_search_status', status)

  # necessary to allow web application access to music files
  @app.route('/video_files/<path:filename>')
  def serve_video_files(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)
  
  video_search_engine = VideoSearch(cfg=cfg)
  video_search_engine.initiate(models_folder=cfg.main.models_path, cache_folder=cfg.main.cache_path) # Needs actual models path
  
  # For now, cached_file_list and cached_file_hash can be from the base search engine
  cached_file_list = video_search_engine.cached_file_list
  cached_file_hash = video_search_engine.cached_file_hash

  @socketio.on('emit_videos_page_get_files')
  def get_files(input_data):
    nonlocal media_directory

    if media_directory is None:
      show_search_status("Video media folder is not set.")
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
        show_search_status("Access denied: Directory traversal attempt detected")
        # Default to the safe base directory if an invalid path is provided.
        current_path = safe_base_dir
    # --- End of Prevention ---

    # if path == "":
    #   current_path = media_directory
    # else:
    #   current_path = os.path.join(media_directory, '..', path)


    folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
    print('folder_path', folder_path)

    show_search_status(f"Searching for videos in '{folder_path}'.")

      
    # all_files = glob.glob(os.path.join(current_path, '**/*'), recursive=True)

    # # Filter the list to only include files of certain types
    # all_files = [f for f in all_files if f.lower().endswith(tuple(cfg.videos.media_formats))]

    all_files = cached_file_list.get_all_files(current_path, cfg.videos.media_formats)

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
      
      # Get file hash, use dummy if file not found
      try:
          all_hashes.append(cached_file_hash.get_file_hash(file_path))
      except FileNotFoundError:
          all_hashes.append("file_not_found_hash_" + str(uuid.uuid4())) # Unique dummy hash

    cached_file_hash.save_hash_cache()


    # Sort image by text or image query
    show_search_status(f"Sorting videos by {text_query}")
    if text_query and len(text_query) > 0:
      if text_query.lower().strip() == "random":
        np.random.shuffle(all_files)
      elif text_query.lower().strip() == "recommendation":
        # For now, we only need basic info from DB for sorting.
        # Model ratings update logic can be added later if needed.
        video_data = db_models.VideosLibrary.query.with_entities(
            db_models.VideosLibrary.hash,
            db_models.VideosLibrary.user_rating,
            db_models.VideosLibrary.model_rating,
            db_models.VideosLibrary.full_play_count,
            db_models.VideosLibrary.skip_count,
            db_models.VideosLibrary.last_played
        ).filter(db_models.VideosLibrary.hash.in_(all_hashes)).all()

        keys = ['hash', 'user_rating', 'model_rating', 'full_play_count', 'skip_count', 'last_played']
        video_data_dict = [dict(zip(keys, row)) for row in video_data]
        
        # Create a map from hash to db_data for quick lookup
        hash_to_db_data = {item['hash']: item for item in video_data_dict}
        
        # Create a list of dictionaries with all necessary fields for sort_files_by_recommendation
        full_video_data_for_sorting = []
        for file_path, file_hash in zip(all_files, all_hashes):
            full_video_data_for_sorting.append(hash_to_db_data.get(file_hash, {
                'hash': file_hash, 'user_rating': None, 'model_rating': None, 'full_play_count': 0, 'skip_count': 0, 'last_played': None
            }))
        all_files, scores = sort_files_by_recommendation(all_files, full_video_data_for_sorting)

    '''
    # Sort image by text or image query
    show_search_status(f"Sorting images by {text_query}")
    if text_query and len(text_query) > 0:
      # Sort images by file size
      if text_query.lower().strip() == "file size":
        all_files = sorted(all_files, key=os.path.getsize)
      # Sort images by resolution
      elif text_query.lower().strip() == "resolution":
        # TODO: THIS IS VERY SLOW, NEED TO OPTIMIZE
        resolutions = {file_path: Image.open(file_path).size for file_path in all_files}
        all_files = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])
      # Sort images by duplicates
      elif text_query.lower().strip() == "similarity":
        show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files).cpu().detach().numpy() 

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
      elif text_query.lower().strip() == "random":
        np.random.shuffle(all_files)
      else:
        show_search_status(f"Extracting embeddings")
        embeds_img = ImageSearch.process_images(all_files, callback=print_emb_extracting_status)
        embeds_text = ImageSearch.process_text(text_query)
        scores = ImageSearch.compare(embeds_img, embeds_text)

        # Create a list of indices sorted by their corresponding score
        show_search_status(f"Sorting by relevance")
        sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

        # Use the sorted indices to sort all_files
        all_files = [all_files[i] for i in sorted_indices]

    #all_files = sorted(all_files, key=os.path.basename)
    '''

    # Extracting metadata for relevant batch of videos
    show_search_status(f"Extracting metadata for relevant batch of videos")
    page_files = all_files[pagination:limit]
    page_files_hashes = [cached_file_hash.get_file_hash(f) for f in page_files]
    
    # Fetch DB data for the relevant batch of videos
    video_db_items = db_models.VideosLibrary.query.filter(db_models.VideosLibrary.hash.in_(page_files_hashes)).all()
    video_db_items_map = {item.hash: item for item in video_db_items}

    for ind, full_path in enumerate(page_files):
      basename = os.path.basename(full_path)
      preview_path = os.path.join(os.path.dirname(full_path), basename + ".preview.png")
      
      # #with open(full_path, "rb") as f:
      # #  bytes = f.read() # read entire file as bytes
      # #  file_size = len(bytes) # Get the file size in bytes
      # #  hash = hashlib.md5(bytes).hexdigest() # Compute the hash of the file

      #   #Use BytesIO to create a file-like object from bytes and get resolution with Pillow
      #   #image = Image.open(BytesIO(bytes))
      #   #resolution = image.size  # Returns a tuple (width, height)

      video_file_hash = page_files_hashes[ind]
      
      # Generate preview if it does not exist
      if not os.path.exists(preview_path):
        show_search_status(f"Generating preview for {basename}...")
        try:
          generate_preview(full_path, preview_path)
          show_search_status(f"Generated preview for {basename}.")
        except Exception as e:
          print(f"Error generating preview for {basename}: {e}")
          show_search_status(f"Failed to generate preview for {basename}. Using placeholder.")

      # data = {
      #   "type": "file",
      #   "full_path": full_path,
      #   "file_path": os.path.relpath(full_path, media_directory),
      #   "preview_path": os.path.relpath(preview_path, media_directory),
      #   "base_name": basename,
      #   "hash": "", #hash,
      #   "user_rating": "...",
      #   "model_rating": "...",
      #   "file_size": "...", #convert_size(file_size),
      #   "resolution": "...", #f"{resolution[0]}x{resolution[1]}",
      # }
      # files_data.append(data)

      file_size = os.path.getsize(full_path)

      user_rating = None
      model_rating = None
      last_played = "Never"

      from pages.utils import time_difference
      
      # Fetch data from DB for the current video
      video_db_item = video_db_items_map.get(video_file_hash)
      if video_db_item:
        user_rating = video_db_item.user_rating
        model_rating = video_db_item.model_rating
        if video_db_item.last_played:
            last_played_timestamp = video_db_item.last_played.timestamp()
            last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())
        
      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": os.path.relpath(full_path, media_directory), # Relative path for serving
        "preview_path": os.path.relpath(preview_path, media_directory), # Relative path for serving preview
        "base_name": basename,
        "hash": video_file_hash,
        "user_rating": user_rating,
        "model_rating": model_rating,
        "file_size": convert_size(file_size),
        "resolution": "N/A", # Placeholder, implement proper metadata extraction later (e.g., using ffprobe)
        "length": "N/A",     # Placeholder, implement proper metadata extraction later (e.g., using ffprobe)
        "last_played": last_played,
      }
      files_data.append(data)
    
                 
    # Extract subfolders structure from the path into a dict
    #folders = get_folder_structure(media_directory)
    folders = file_manager.get_folder_structure(media_directory, cfg.videos.media_formats)

    # Return "No files in the directory" if the path not exist or empty.
    if not folders:
      show_search_status(f"No files in the directory.")
      socketio.emit('emit_videos_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": 0, "folders": folders, "all_files_paths": []})
      return

    # Extract main folder name
    main_folder_name = os.path.basename(os.path.normpath(media_directory))
    folders['name'] = main_folder_name

    all_files_paths = [os.path.relpath(file_path, media_directory) for file_path in all_files]
    
    socketio.emit('emit_videos_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders, "all_files_paths": all_files_paths})

    show_search_status(f'{len(all_files)} files processed in {time.time() - start_time:.4f} seconds.')


    # # Extract main folder name
    # main_folder_name = os.path.basename(os.path.normpath(media_directory))
    # folders = {main_folder_name: folders}
    
    # socketio.emit('emit_videos_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders})

    # show_search_status(f'{len(all_files)} videos processed in {time.time() - start_time:.4f} seconds.')

  @socketio.on('emit_videos_page_open_file_in_folder')
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

  @socketio.on('emit_videos_page_send_file_to_trash')
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

  @socketio.on('emit_videos_page_video_start_playing')
  def video_start_playing(video_hash):
    """
    Updates the last_played timestamp for a video in the database.
    """
    print(f"Updating last_played for video: {video_hash}")
    video = db_models.VideosLibrary.query.filter_by(hash=video_hash).first()
    if video:
        video.last_played = datetime.datetime.now()
        # Also, increment full_play_count if needed, or handle skip here based on player events.
        # For now, just last_played as requested.
        db_models.db.session.commit()
    else:
        # Create a minimal record in the database if video not found
        print(f"Video with hash {video_hash} not found in DB, creating minimal record.")
        new_video = db_models.VideosLibrary(
            hash=video_hash,
            last_played=datetime.datetime.now()
        )
        db_models.db.session.add(new_video)
        db_models.db.session.commit()


  # ----------------------------------------
  # HSL Streaming test
  # ----------------------------------------

  # This section requires a lot of refactoring and rethinking, for now it just works, somehow
  

  # Store active transcoding processes
  active_transcodings = {}

  @app.route('/stream/<stream_id>/master.m3u8')
  def stream_video(stream_id):
    """Stream a video with the given stream_id"""
    if stream_id not in active_transcodings:
        return "Stream not found", 404
        
    # If stream is already set up, just serve the playlist
    if 'temp_dir' in active_transcodings[stream_id]:
        master_playlist = os.path.join(active_transcodings[stream_id]['temp_dir'], "master.m3u8")
        if os.path.exists(master_playlist):
            return send_from_directory(active_transcodings[stream_id]['temp_dir'], "master.m3u8")
    
    # Otherwise, set up the stream
    video_path = active_transcodings[stream_id]['path']
    print(f"Starting stream for {stream_id} from {video_path}")
    
    # Create a temp directory for HLS files
    temp_dir = tempfile.mkdtemp()
    active_transcodings[stream_id]['temp_dir'] = temp_dir
    
    # Create master playlist path
    master_playlist = os.path.join(temp_dir, "master.m3u8")
    
    # Get video duration and other metadata
    try:
        metadata_cmd = [
            'ffprobe', 
            '-v', 'error', 
            '-show_entries', 'format=duration', 
            '-of', 'json', 
            video_path
        ]
        metadata_output = subprocess.check_output(metadata_cmd).decode('utf-8').strip()
        import json
        metadata = json.loads(metadata_output)
        duration = float(metadata['format']['duration'])
        print(f"Video duration: {duration} seconds")
        active_transcodings[stream_id]['duration'] = duration
    except Exception as e:
        print(f"Error getting video metadata: {e}")
        duration = 0
    
    # Build FFmpeg command for HLS with optimized settings
    cmd = [
        'ffmpeg',
        '-loglevel', 'error',
        '-i', video_path,
        '-c:v', 'libx264',       # Video codec
        '-preset', 'ultrafast',  # Faster encoding
        '-tune', 'zerolatency',  # Reduce latency
        '-c:a', 'aac',           # Audio codec
        '-ar', '44100',          # Audio sample rate
        '-ac', '2',              # Stereo audio
        '-b:a', '128k',          # Audio bitrate
        '-f', 'hls',             # HLS format
        '-hls_time', '2',        # 2-second segments
        '-hls_list_size', '0',   # Keep all segments
        '-hls_flags', 'independent_segments+split_by_time+append_list', # append_list might be redundant with vod but often harmless
        '-hls_segment_type', 'mpegts',
        '-hls_playlist_type', 'event',
        '-force_key_frames', 'expr:gte(t,n_forced*2)', # Keyframe every 2s
        '-g', '48',              # GOP size
        '-start_number', '0',    # Start segment numbering at 0
        '-hls_segment_filename', os.path.join(temp_dir, f'segment_%03d.ts'),
        master_playlist
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.DEVNULL, 
        stderr=subprocess.PIPE
    )
    
    # Store the process
    active_transcodings[stream_id]['process'] = process
    
    # Wait for the master playlist to be created (with a timeout)
    start_time = time.time()
    while not os.path.exists(master_playlist) and time.time() - start_time < 10:
        time.sleep(0.1)
        # Check if process died
        if process.poll() is not None:
            stderr_output = ""
            if process.stderr: # Check if stderr is available
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                except Exception as e:
                    print(f"Error reading ffmpeg stderr: {e}")
            # You might want to log stderr_output here if it contains useful error info
            # print(f"FFmpeg stderr: {stderr_output}") 
            return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500
    
    if not os.path.exists(master_playlist):
        return "Timeout waiting for playlist to be created", 500
    
    # Wait for initial buffer segments to be created before serving
    segments_ready = 0
    segment_pattern = os.path.join(temp_dir, "segment_*.ts")
    while segments_ready < 3 and time.time() - start_time < 10:  # Wait for 3 segments or 10s max
        segments_ready = len(glob.glob(segment_pattern))
        if segments_ready >= 3:
            break
        time.sleep(0.2)
        if process.poll() is not None:
            stderr_output = ""
            if process.stderr: # Check if stderr is available
                try:
                    stderr_output = process.stderr.read().decode('utf-8', errors='ignore') # Added errors='ignore'
                except Exception as e:
                    print(f"Error reading ffmpeg stderr: {e}")
            # print(f"FFmpeg stderr: {stderr_output}")
            return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500


    print(f"Starting playback with {segments_ready} segments ready")
    
    # Serve the m3u8 file
    return send_from_directory(temp_dir, "master.m3u8")

  @app.route('/stream/<stream_id>/<path:filename>')
  def stream_segment(stream_id, filename):
      """Serve HLS segment files"""
      if stream_id not in active_transcodings or 'temp_dir' not in active_transcodings[stream_id]:
          return "Stream not found", 404
      
      temp_dir = active_transcodings[stream_id]['temp_dir']
      return send_from_directory(temp_dir, filename)

  @socketio.on('emit_videos_page_start_streaming')
  def start_streaming(file_path):
      """Start streaming a video file"""
      # Make file_path absolute if it's not already
      if not os.path.isabs(file_path):
          file_path = os.path.join(media_directory, file_path)
      
      # Verify file exists
      if not os.path.isfile(file_path):
          print(f"Error: File not found at {file_path}")
          return {"error": "File not found"}
          
      # Generate unique stream ID
      stream_id = str(uuid.uuid4())
      
      # Store info about the stream
      active_transcodings[stream_id] = {
          'path': file_path,
          'process': None,
          'start_time': time.time()
      }
      
      # Return stream URL to client
      return {
          'stream_id': stream_id,
          'stream_url': f'/stream/{stream_id}/master.m3u8'
      }
  
  @socketio.on('emit_videos_page_stop_streaming')
  def stop_streaming(stream_id):
      """Stop streaming a video file and clean up resources"""
      if stream_id in active_transcodings:
          print(f"Stopping stream {stream_id}")
          
          # Clean up the resources
          cleanup_transcoding(stream_id)
          
          return {"status": "success", "message": "Stream stopped and resources cleaned up"}
      else:
          print(f"Stream {stream_id} not found or already stopped")
          return {"status": "error", "message": "Stream not found"}

  def cleanup_transcoding(stream_id):
    """Clean up temporary files and processes"""
    if stream_id in active_transcodings:
        print(f"Cleaning up transcoding for stream {stream_id}")
        
        # Terminate the process
        if 'process' in active_transcodings[stream_id] and active_transcodings[stream_id]['process']:
            try:
                active_transcodings[stream_id]['process'].terminate()
                print(f"Process for stream {stream_id} terminated")
            except Exception as e:
                print(f"Error terminating process: {e}")
        
        # Close error log file
        if 'error_log' in active_transcodings[stream_id]:
            try:
                active_transcodings[stream_id]['error_log'].close()
            except:
                pass
        
        # Remove temp directory
        if 'temp_dir' in active_transcodings[stream_id]:
            try:
                shutil.rmtree(active_transcodings[stream_id]['temp_dir'])
                print(f"Removed temp directory for stream {stream_id}")
            except Exception as e:
                print(f"Error removing temp directory: {e}")
        
        # Remove from active transcodings
        del active_transcodings[stream_id]
        print(f"Stream {stream_id} completely cleaned up")
  
  # Cleanup old transcodings periodically
  def cleanup_old_transcodings():
      """Periodically check for and clean up old transcoding processes"""
      while True:  # Continuous loop
          try:
              # Print some debug info
              print(f"Cleanup thread running, active streams: {len(active_transcodings)}")
              
              current_time = time.time()
              for stream_id in list(active_transcodings.keys()):
                  # Check if the stream has been active for more than 1 hour
                  if current_time - active_transcodings[stream_id]['start_time'] > 3600:  # 1 hour timeout
                      print(f"Cleaning up expired stream: {stream_id}")
                      cleanup_transcoding(stream_id)
                      
              # Sleep for a specified interval before checking again
              time.sleep(300)  # Check every 5 minutes
          except Exception as e:
              print(f"Error in cleanup thread: {e}")
              time.sleep(60)  # If there's an error, wait a bit before retrying
              
  # Start cleanup thread
  cleanup_thread = threading.Thread(target=cleanup_old_transcodings)
  cleanup_thread.daemon = True
  cleanup_thread.start()