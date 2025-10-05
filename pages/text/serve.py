import os
import time

from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO

import pages.utils
import pages.file_manager as file_manager # or pages.utils if you put get_folder_structure there
import pages.text.db_models as db_models
from pages.socket_events import CommonSocketEvents

import numpy as np

from pages.text.engine import TextSearch, TextEvaluator
from pages.utils import SortingProgressCallback, EmbeddingGatheringCallback
from pages.common_filters import CommonFilters

from omegaconf import OmegaConf

# EVENTS:

# Incoming (handled with @socketio.on):

# emit_text_page_get_folders
# emit_text_page_get_files
# emit_text_page_get_file_content
# emit_text_page_save_file_content
# emit_text_page_get_path_to_media_folder
# emit_text_page_update_path_to_media_folder

# Outgoing (explicit socketio.emit calls):

# emit_text_page_show_folders
# emit_text_page_show_files
# emit_text_page_show_file_content
# emit_text_page_show_path_to_media_folder

# Outgoing (indirect via CommonSocketEvents):

# show_search_status (emits a “search status” event from CommonSocketEvents)

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
        # media_directory = os.path.join(data_folder, cfg.text.media_directory)
        media_directory = cfg.text.media_directory

    text_search_engine = TextSearch(cfg=cfg)
    text_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)
    cached_file_list = text_search_engine.cached_file_list
    cached_file_hash = text_search_engine.cached_file_hash
    cached_metadata = text_search_engine.cached_metadata

    text_evaluator = TextEvaluator(embedding_dim=text_search_engine.embedding_dim)

    #TextSearch.initiate(cfg, models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)
    #cached_file_list = TextSearch.cached_file_list
    #cached_file_hash = TextSearch.cached_file_hash
    #cached_metadata = TextSearch.cached_metadata

    common_socket_events = CommonSocketEvents(socketio)

    embedding_gathering_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

    text_file_manager = file_manager.FileManager(
        media_directory=media_directory,
        engine=text_search_engine,
        module_name="text",
        media_formats=cfg.text.media_formats,
        socketio=socketio,
        db_schema=db_models.TextLibrary,
    )

    def update_model_ratings(files_list):
        print('update_model_ratings')

        # filter out files that already have a rating in the DB
        files_list_hash_map = {file_path: cached_file_hash.get_file_hash(file_path) for file_path in files_list}
        hash_list = list(files_list_hash_map.values())

        # Fetch rated files from the database in a single query
        rated_files_db_items = db_models.TextLibrary.query.filter(
            db_models.TextLibrary.hash.in_(hash_list),
            db_models.TextLibrary.model_rating.isnot(None),
            db_models.TextLibrary.model_hash.is_(text_evaluator.hash)
        ).all()

        # Create a list of hashes for rated files
        rated_files_hashes = {item.hash for item in rated_files_db_items}

        # Filter out files that already have a rating in the database
        filtered_files_list = [file_path for file_path, file_hash in files_list_hash_map.items() if file_hash not in rated_files_hashes]
        if not filtered_files_list: return
        

        # Rate all files in case they are not rated or model was updated
        embeddings = text_search_engine.process_files(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
        # model_ratings = text_evaluator.predict(embeddings)

        # Update the model ratings in the database
        common_socket_events.show_search_status(f"Updating model ratings of files...") 
        new_items = []
        update_items = []
        last_shown_time = 0
        for ind, full_path in enumerate(filtered_files_list):
            # print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} files.")

            hash = files_list_hash_map[full_path]

            # print('model_ratings[ind]', model_ratings[ind])
            model_rating = None #model_ratings[ind].mean().item()

            music_db_item = db_models.TextLibrary.query.filter_by(hash=hash).first()
            if music_db_item:
                music_db_item.model_rating = model_rating
                music_db_item.model_hash = text_evaluator.hash
                update_items.append(music_db_item)
            else:
                file_data = {
                        "hash": hash,
                        "file_path": os.path.relpath(full_path, media_directory),
                        "model_rating": model_rating,
                        "model_hash": text_evaluator.hash
                }
                new_items.append(db_models.TextLibrary(**file_data))

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

    @socketio.on('emit_text_page_get_folders')  
    def get_folders(data):
        path = data.get('path', '')
        return text_file_manager.get_folders(path)

    @socketio.on('emit_text_page_get_files')
    def get_files(data):
        # Create common filters instance
        common_filters = CommonFilters(
            engine=text_search_engine,
            common_socket_events=common_socket_events,
            media_directory=media_directory,
            db_schema=db_models.TextLibrary,
            update_model_ratings_func=update_model_ratings
        )

        # Get parameters
        path = data.get('path', '')
        pagination = data.get('pagination', 0)
        limit = data.get('limit', 100)
        text_query = data.get('text_query', None)
        seed = data.get('seed', None)

        # Define available filters
        filters = {
            # "by_file": filter_by_file, # special sorting case when file path used as query
            "by_text": common_filters.filter_by_text, # special sorting case when text used as query, i.e. all other cases wasn't triggered
            # "file_size": filter_by_file_size,
            # "length": filter_by_length,
            # "similarity": filter_by_similarity, 
            # "random": filter_by_random, 
            # "rating": filter_by_rating, 
            # "recommendation": filter_by_recommendation
        }

        # Define a method to gather domain specific file information
        def get_file_info(full_path, file_hash):
            db_item = db_models.TextLibrary.query.filter_by(hash=file_hash).first()
                    
            if db_item:
                file_data = text_search_engine.cached_metadata.get_metadata(full_path, file_hash)     
            else:
                raise Exception(f"File '{full_path}' with hash '{file_hash}' not found in the database.")
            
            return {
                    "user_rating": db_item.user_rating,
                    "model_rating": db_item.model_rating,
                    "preview_text": get_text_preview(full_path),    
                    "file_data": file_data,
                }

        return text_file_manager.get_files(path, pagination, limit, text_query, seed, filters, get_file_info, update_model_ratings)

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