
from flask import Flask, render_template, send_from_directory
import os
import glob
from PIL import Image
from pages.images.engine import ImageSearch

def init_socket_events(socketio, app=None, cfg=None):
  media_directory = cfg.images.media_directory

  # necessary to allow web application access to music files
  @app.route('/images_files/<path:filename>')
  def serve_images_files(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_images_page_get_files')
  def get_files(input_data):
    nonlocal media_directory

    path = input_data.get('path', '')
    pagination = input_data.get('pagination', 0)
    limit = input_data.get('limit', 100)
    text_query = input_data.get('text_query', None)

    files_data = []
    all_files = glob.glob(os.path.join(media_directory, path, '**/*'), recursive=True)

    # Filter the list to only include files of certain types
    all_files = [f for f in all_files if f.lower().endswith(tuple(cfg.images.media_formats))]

    # Sort image by text or image query
    if text_query and len(text_query) > 0:
      embeds_img = ImageSearch.process_images(all_files)
      embeds_text = ImageSearch.process_text(text_query)
      scores = ImageSearch.compare(embeds_img, embeds_text)

      # Create a list of indices sorted by their corresponding score
      sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

      # Use the sorted indices to sort all_files
      all_files = [all_files[i] for i in sorted_indices]

    #all_files = sorted(all_files, key=os.path.basename)
    page_files = all_files[pagination:limit]

    for full_path in page_files:
      basename = os.path.basename(full_path)

      data = {
        "type": "file",
        "full_path": full_path,
        "file_path": os.path.relpath(full_path, media_directory),
        "base_name": basename
      }
      files_data.append(data)

    socketio.emit('emit_images_page_show_files', {"files_data": files_data, "folder_path": path, "total_files": len(all_files)})

    '''files_data = []
    for file_path in os.listdir(os.path.join(media_directory, path)):
      full_path = os.path.join(media_directory, path, file_path)
      basename = os.path.basename(file_path)

      file_type = "undefined"
      if os.path.isdir(full_path): file_type = "folder"
      if os.path.isfile(full_path): file_type = "file"

      if file_type == "folder" or basename.lower().endswith(tuple(cfg.images.media_formats)):
        data = {
          "type": file_type,
          "full_path": full_path,
          "file_path": os.path.join(path, file_path),
          "base_name": basename
        }
        files_data.append(data)
      
      #if file_path.lower().endswith(tuple(cfg.music.media_formats)):
        

    socketio.emit('emit_images_page_show_files', {"files_data":files_data, "folder_path": path}) '''
