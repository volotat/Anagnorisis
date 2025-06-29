import os
import time

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

import pages.utils
import pages.file_manager as file_manager # or pages.utils if you put get_folder_structure there
from pages.socket_events import CommonSocketEvents

import numpy as np

from pages.text.engine import TextSearch #, TextEvaluator
from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback

from omegaconf import OmegaConf


def get_text_preview(file_path):
    preview_text = '' # Default no preview
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Get the first few lines as preview, or adjust as needed
            preview_text = '\n'.join(f.readlines()[:25])  # First 3 lines
            if len(preview_text) > 400: # Limit preview length
                preview_text = preview_text[:400] + '...'
    except Exception as e:
        print(f"Error reading preview from {file_path}: {e}")
        preview_text = 'Error loading preview!'

    return preview_text

def resolve_media_path(media_directory, path):
    """Return the absolute path for the given media directory and relative path."""
    if path == "":
        return media_directory
    return os.path.abspath(os.path.join(media_directory, '..', path))

def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
    if cfg.text.media_directory is None:
        print("Text media folder is not set.")
        media_directory = None
    else:
        media_directory = os.path.join(data_folder, cfg.text.media_directory)

    text_search_engine = TextSearch(cfg=cfg)
    text_search_engine.initiate(models_folder=cfg.main.models_path, cache_folder=cfg.main.cache_path)
    cached_file_list = text_search_engine.cached_file_list
    cached_file_hash = text_search_engine.cached_file_hash
    cached_metadata = text_search_engine.cached_metadata

    #TextSearch.initiate(cfg, models_folder=cfg.main.models_path, cache_folder=cfg.main.cache_path)
    #cached_file_list = TextSearch.cached_file_list
    #cached_file_hash = TextSearch.cached_file_hash
    #cached_metadata = TextSearch.cached_metadata

    common_socket_events = CommonSocketEvents(socketio)

    embedding_gathering_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

    @socketio.on('emit_text_page_get_folders')  
    def get_folders(data):
        path = data.get('path', '')
        current_path = resolve_media_path(media_directory, path)
        folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
        print('folder_path', folder_path)

        # Extract subfolders structure from the path into a dict
        folders = file_manager.get_folder_structure(media_directory, cfg.text.media_formats)

        # Extract main folder name
        main_folder_name = os.path.basename(os.path.normpath(media_directory))
        folders['name'] = main_folder_name

        socketio.emit('emit_text_page_show_folders', {"folders": folders, "folder_path": folder_path})

    @socketio.on('emit_text_page_get_files')
    def get_files(data):
        nonlocal media_directory

        if media_directory is None:
            common_socket_events.show_search_status("Text media folder is not set.")
            return

        start_time = time.time()

        path = data.get('path', '')
        pagination = data.get('pagination', 0)
        limit = data.get('limit', 100)
        text_query = data.get('text_query', None)
        
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
            common_socket_events.show_search_status("Access denied: Directory traversal attempt detected")
            # Default to the safe base directory if an invalid path is provided.
            current_path = safe_base_dir
        # --- End of Prevention ---

        folder_path = os.path.relpath(current_path, os.path.join(media_directory, '..')) + os.path.sep
        print('folder_path', folder_path)

        common_socket_events.show_search_status(f"Searching for text files in '{folder_path}'.")

        
        all_files = []
        all_files = cached_file_list.get_all_files(current_path, cfg.text.media_formats)
        
        print(f"Found {len(all_files)} files in {current_path}")
        common_socket_events.show_search_status(f"Gathering hashes for {len(all_files)} files.")
        
        # Initialize the last shown time
        last_shown_time = 0

        # Get the hash of each file
        all_hashes = []
        for ind, file_path in enumerate(all_files):
            current_time = time.time()
            if current_time - last_shown_time >= 1:
                common_socket_events.show_search_status(f"Gathering hashes for {ind+1}/{len(all_files)} files.")
                last_shown_time = current_time
            
            all_hashes.append(cached_file_hash.get_file_hash(file_path))

        cached_file_hash.save_hash_cache()


        # Sort files by text or file query
        common_socket_events.show_search_status(f"Sorting music files by {text_query}")
        if text_query and len(text_query) > 0:
            # If there is a file path in the query, sort files by similarity to that file
            if os.path.isfile(text_query) and text_query.lower().endswith(tuple(cfg.music.media_formats)):
                pass
            # Sort music by file size
            elif text_query.lower().strip() == "file size":
                common_socket_events.show_search_status(f"Gathering file sizes for sorting...") # Initial status message
                all_files = sorted(all_files, key=os.path.getsize)
            # Sort music by length
            elif text_query.lower().strip() == "length":
                common_socket_events.show_search_status(f"Gathering resolutions for sorting...") # Initial status message
                pass
            # Sort music by duplicates
            elif text_query.lower().strip() == "similarity":
                pass
            # Sort music randomly
            elif text_query.lower().strip() == "random":
                np.random.shuffle(all_files)
            # Sort music by rating
            elif text_query.lower().strip() == "rating": 
                pass
            # Sort music with recommendation engine 
            elif text_query.lower().strip() == "recommendation":
                pass 
            # Sort files by the text query
            else:
                common_socket_events.show_search_status(f"Extracting embeddings")
                embeds_files = text_search_engine.process_files(all_files, callback=embedding_gathering_callback, media_folder=media_directory)
                embeds_text = text_search_engine.process_text(text_query)
                scores = text_search_engine.compare(embeds_files, embeds_text)

                # Create a list of indices sorted by their corresponding score
                common_socket_events.show_search_status(f"Sorting by relevance")
                sorted_indices = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)

                # Use the sorted indices to sort all_files
                all_files = [all_files[i] for i in sorted_indices]
        
        # Truncate the list of hashes to the current pagination
        page_files = all_files[pagination:limit]
        page_hashes = [cached_file_hash.get_file_hash(file_path) for file_path in page_files]

        # Extracting metadata for relevant batch of files
        common_socket_events.show_search_status(f"Extracting metadata for relevant batch of files.")
        

        # Extract DB data for the relevant batch of files
        # text_db_items = db_models.TextLibrary.query.filter(db_models.TextLibrary.hash.in_(page_hashes)).all()
        # text_db_items_map = {item.hash: item for item in text_db_items}

        # Check if there files without model rating
        # no_model_rating_files = [os.path.join(media_directory, item.file_path) for item in text_db_items if item.model_rating is None]
        # print('no_model_rating_files size', len(no_model_rating_files))
        
        # if len(no_model_rating_files) > 0:
        #    # Update the model ratings of all current files
        #    update_model_ratings(no_model_rating_files)

        for ind, full_path in enumerate(page_files):
            file_path = os.path.relpath(full_path, media_directory)
            basename = os.path.basename(full_path)
            file_size = os.path.getsize(full_path)
            file_hash = page_hashes[ind]
        
            user_rating = None
            model_rating = None

            preview_text = get_text_preview(full_path)

            # if file_hash in text_db_items_map:
            #     text_db_item = text_db_items_map[file_hash]
            #     user_rating = text_db_item.user_rating
            #     model_rating = text_db_item.model_rating

            data = {
                "type": "file",
                "full_path": full_path,
                "file_path": file_path,
                "base_name": basename,
                "hash": file_hash,
                "user_rating": user_rating,
                "model_rating": model_rating,
                "file_size": pages.utils.convert_size(file_size),
                "preview_text": preview_text
            }
            files_data.append(data)

        all_files_paths = [os.path.relpath(file_path, media_directory) for file_path in all_files]
        
        socketio.emit('emit_text_page_show_files', {"files_data": files_data, "total_files": len(all_files), "all_files_paths": all_files_paths})

        common_socket_events.show_search_status(f'{len(all_files)} files processed in {time.time() - start_time:.4f} seconds.')

    @socketio.on('emit_text_page_get_file_content')
    def get_file_content(data):
        file_path = data.get('file_path')
        full_path = os.path.join(media_directory, file_path)
        try:
            with open(full_path, 'r', encoding='utf-8') as f: 
                content = f.read()
            socketio.emit('emit_text_page_show_file_content', {"content": content, "file_path": file_path})
        except Exception as e:
            print(f"Error reading file: {full_path}: {e}")
            socketio.emit('emit_text_page_show_file_content', {"content": "Error loading file.", "file_path": file_path}) # Error to frontend


    @socketio.on('emit_text_page_save_file_content')
    def save_file_content(data):
        file_path = data.get('file_path')
        text_content = data.get('text_content')
        full_path = os.path.join(media_directory, file_path)
        try:
            with open(full_path, 'w', encoding='utf-8') as f: 
                f.write(text_content)
            print(f"File saved: {full_path}") # Log success
            # Optionally emit success event to frontend
        except Exception as e:
            print(f"Error saving file: {full_path}: {e}") # Log error
            # Optionally emit error event to frontend

    @socketio.on('emit_text_page_get_path_to_media_folder')
    def get_path_to_media_folder():
        nonlocal media_directory
        socketio.emit('emit_text_page_show_path_to_media_folder', cfg.text.media_directory)

    @socketio.on('emit_text_page_update_path_to_media_folder')
    def update_path_to_media_folder(new_path):
        nonlocal media_directory
        cfg.text.media_directory = new_path

        # Update the configuration file
        with open(os.path.join(data_folder, 'Anagnorisis-app', 'config.yaml'), 'w') as file:
            OmegaConf.save(cfg, file)

        media_directory = os.path.join(data_folder, cfg.text.media_directory)
        socketio.emit('emit_text_page_show_path_to_media_folder', cfg.text.media_directory)