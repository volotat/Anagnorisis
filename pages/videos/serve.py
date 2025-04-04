
from flask import Flask, render_template, send_from_directory
import os
import sys
import glob
import subprocess
import hashlib
import numpy as np
from io import BytesIO
from PIL import Image
from pages.images.engine import ImageSearch
import math
from scipy.spatial import distance
import torch
import gc
import send2trash
import time
from moviepy.editor import VideoFileClip

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
  media_directory = os.path.join(data_folder, cfg.videos.media_directory)

  def show_search_status(status):
    socketio.emit('emit_videos_page_show_search_status', status)

  # necessary to allow web application access to music files
  @app.route('/video_files/<path:filename>')
  def serve_video_files(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_videos_page_get_files')
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

      
    all_files = glob.glob(os.path.join(current_path, '**/*'), recursive=True)

    # Filter the list to only include files of certain types
    all_files = [f for f in all_files if f.lower().endswith(tuple(cfg.videos.media_formats))]

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
    
    for full_path in page_files:
      basename = os.path.basename(full_path)
      preview_path = os.path.join(os.path.dirname(full_path), basename + ".preview.png")
      
      #with open(full_path, "rb") as f:
      #  bytes = f.read() # read entire file as bytes
      #  file_size = len(bytes) # Get the file size in bytes
      #  hash = hashlib.md5(bytes).hexdigest() # Compute the hash of the file

        #Use BytesIO to create a file-like object from bytes and get resolution with Pillow
        #image = Image.open(BytesIO(bytes))
        #resolution = image.size  # Returns a tuple (width, height)
    
      # Generate preview if it does not exist
      if not os.path.exists(preview_path):
        generate_preview(full_path, preview_path)

      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": os.path.relpath(full_path, media_directory),
        "preview_path": os.path.relpath(preview_path, media_directory),
        "base_name": basename,
        "hash": "", #hash,
        "user_rating": "...",
        "model_rating": "...",
        "file_size": "...", #convert_size(file_size),
        "resolution": "...", #f"{resolution[0]}x{resolution[1]}",
      }
      files_data.append(data)
    
                 
    # Extract subfolders structure from the path into a dict
    folders = get_folder_structure(media_directory)

    # Extract main folder name
    main_folder_name = os.path.basename(os.path.normpath(media_directory))
    folders = {main_folder_name: folders}
    
    socketio.emit('emit_videos_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders})

    show_search_status(f'{len(all_files)} videos processed in {time.time() - start_time:.4f} seconds.')

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