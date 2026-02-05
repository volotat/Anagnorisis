
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

from src.metadata_search import MetadataSearch

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
    common_socket_events = CommonSocketEvents(socketio, module_name="images")

    common_socket_events.show_loading_status('Checking media directory configuration...')

    if cfg.images.media_directory is None:
        print("Images media folder is not set.")
        media_directory = None
    else:
        # media_directory = os.path.join(data_folder, cfg.images.media_directory)
        media_directory = cfg.images.media_directory
    
    # TODO: Remove this hack from the codebase and create a proper file-manager system

    # Check if running in Docker
    is_docker = os.environ.get('RUNNING_IN_DOCKER', 'false').lower() == 'true'
    
    # Define Docker volume mappings (host -> container)
    docker_volume_mappings = {
        '/mnt/media/images': os.environ.get('IMAGES_MODULE_DATA_PATH', './media/Images'),
        '/mnt/media/music': os.environ.get('MUSIC_MODULE_DATA_PATH', './media/Music'),
        '/mnt/media/text': os.environ.get('TEXT_MODULE_DATA_PATH', './media/Text'),
        '/mnt/media/videos': os.environ.get('VIDEOS_MODULE_DATA_PATH', './media/Videos'),
        '/mnt/project_config': os.environ.get('PROJECT_CONFIG_FOLDER_PATH', './project_config')
    }
    
    def convert_host_path_to_container(host_path):
        """Convert a host path to its equivalent Docker container path."""
        if not is_docker:
            return host_path  # No conversion needed if not in Docker
        
        # Normalize the host path
        host_path = os.path.abspath(host_path)
        
        # Check each volume mapping
        for container_path, host_volume in docker_volume_mappings.items():
            host_volume = os.path.abspath(host_volume)
            
            # Check if the host path is within this volume
            if host_path.startswith(host_volume + os.sep) or host_path == host_volume:
                # Convert to container path
                relative_path = os.path.relpath(host_path, host_volume)
                if relative_path == '.':
                    return container_path
                return os.path.join(container_path, relative_path)
        
        # Path is not in any shared volume
        raise ValueError(f"Path '{host_path}' is not accessible from Docker container. Only paths within shared volumes are allowed.")
    
    def convert_container_path_to_host(container_path):
        """Convert a container path to its equivalent host path (for informational purposes)."""
        if not is_docker:
            return container_path
        
        container_path = os.path.abspath(container_path)
        
        for container_volume, host_path in docker_volume_mappings.items():
            if container_path.startswith(container_volume + os.sep) or container_path == container_volume:
                relative_path = os.path.relpath(container_path, container_volume)
                if relative_path == '.':
                    return os.path.abspath(host_path)
                return os.path.join(os.path.abspath(host_path), relative_path)
        
        return container_path
    
    common_socket_events.show_loading_status('Initializing image search engine...')
    images_search_engine = ImageSearch(cfg=cfg)
    images_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)

    # cached_file_hash = images_search_engine.cached_file_hash
    # cached_metadata = images_search_engine.cached_metadata

    common_socket_events.show_loading_status('Initializing image evaluator...')
    image_evaluator = ImageEvaluator() #src.scoring_models.Evaluator(embedding_dim=768)

    common_socket_events.show_loading_status('Loading image evaluator model...')
    image_evaluator.load(os.path.join(cfg.main.personal_models_path, 'image_evaluator.pt'))

    # def show_search_status(status):
    #     common_socket_events.show_search_status(status)
    #     # socketio.emit('emit_images_page_show_search_status', status)

    def gather_resolutions(all_files, all_hashes, media_directory):
        resolutions = {}
        progress_callback = SortingProgressCallback(common_socket_events.show_search_status, operation_name="Gathering images resolution") # Create callback

        for ind, full_path in enumerate(all_files):
            file_path = os.path.relpath(full_path, media_directory)
            # file_hash = all_hashes[ind]

            image_metadata = images_search_engine.get_metadata(full_path)
            if 'resolution' not in image_metadata:
                raise Exception(f"Resolution not found for image: {file_path}")
            
            resolutions[full_path] = image_metadata['resolution']
            progress_callback(ind + 1, len(all_files)) # Update progress

        return resolutions

    embedding_gathering_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

    common_socket_events.show_loading_status('Setting up file manager...')
    images_file_manager = file_manager.FileManager(
        cfg=cfg,
        media_directory=media_directory,
        engine=images_search_engine,
        module_name="images",
        media_formats=cfg.images.media_formats,
        socketio=socketio,
        db_schema=db_models.ImagesLibrary,
    )

    # Create metadata search engine
    common_socket_events.show_loading_status('Initializing metadata search...')
    metadata_search_engine = MetadataSearch(engine=images_search_engine)

    def update_model_ratings(files_list):
        print(f"Updating model ratings for {len(files_list)} images...")

        # filter out files that already have a rating in the DB
        #files_list_hash_map = {file_path: images_search_engine.get_file_hash(file_path) for file_path in files_list}

        files_list_hash_map = {}
        for ind, file_path in enumerate(files_list):
            common_socket_events.show_search_status(f"Computing files hashes {ind+1}/{len(files_list)}")
            file_hash = images_search_engine.get_file_hash(file_path)
            files_list_hash_map[file_path] = file_hash

        hash_list = list(files_list_hash_map.values())

        # # Fetch rated images from the database in a single query
        # rated_images_db_items = db_models.ImagesLibrary.query.filter(
        #     db_models.ImagesLibrary.hash.in_(hash_list),
        #     db_models.ImagesLibrary.model_rating.isnot(None),
        #     db_models.ImagesLibrary.model_hash.is_(image_evaluator.hash)
        # ).all()

        # # Create a list of hashes for rated images
        # rated_images_hashes = {item.hash for item in rated_images_db_items}

        # Fetch ALL images from the database in a single query (not just rated ones)
        all_existing_db_items = db_models.ImagesLibrary.query.filter(
            db_models.ImagesLibrary.hash.in_(hash_list)
        ).all()

        # Create separate sets for existing hashes and those with current model ratings
        existing_hashes = {item.hash for item in all_existing_db_items}
        rated_images_hashes = {item.hash for item in all_existing_db_items 
                            if item.model_rating is not None and item.model_hash == image_evaluator.hash}

        # Filter out files that already have a rating in the database
        filtered_files_list = [file_path for file_path, file_hash in files_list_hash_map.items() if file_hash not in rated_images_hashes]
        if not filtered_files_list: return
        
        # Rate all images in case they are not rated or model was updated
        common_socket_events.show_search_status(f"Computing embeddings for {len(filtered_files_list)} files...")
        embeddings = images_search_engine.process_images(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
        model_ratings = image_evaluator.predict(embeddings)

        # Update the model ratings in the database
        common_socket_events.show_search_status(f"Updating model ratings of images...") 
        new_items = []
        update_items = []
        processed_hashes = set()  # Track hashes we've already processed in this batch

        for ind, full_path in enumerate(filtered_files_list):
            # print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} images.")

            hash = files_list_hash_map[full_path]

            # Skip if we've already processed this hash in this batch
            if hash in processed_hashes:
                continue
            processed_hashes.add(hash)

            model_rating = model_ratings[ind].item()

            # Check if this hash already exists in DB (avoid UNIQUE constraint error)
            if hash in existing_hashes:
                # Update existing record
                image_db_item = db_models.ImagesLibrary.query.filter_by(hash=hash).first()
                if image_db_item:
                    image_db_item.model_rating = model_rating
                    image_db_item.model_hash = image_evaluator.hash
                    update_items.append(image_db_item)
            else:
                image_data = {
                    "hash": hash,
                    "hash_algorithm": images_search_engine.get_hash_algorithm(),
                    "file_path": os.path.relpath(full_path, media_directory),
                    "model_rating": model_rating,
                    "model_hash": image_evaluator.hash
                }
                new_items.append(db_models.ImagesLibrary(**image_data))
                # Add to existing_hashes so subsequent duplicates in this batch are caught
                existing_hashes.add(hash)

            common_socket_events.show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} images.")    

        # Bulk update and insert
        if update_items:
                db_models.db.session.bulk_save_objects(update_items)
        if new_items:
                db_models.db.session.bulk_save_objects(new_items)

        # Commit the transaction
        db_models.db.session.commit()

    common_socket_events.show_loading_status('Setting up filters and routes...')
    # Create common filters instance
    common_filters = CommonFilters(
        engine=images_search_engine,
        metadata_engine=metadata_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.ImagesLibrary,
        update_model_ratings_func=update_model_ratings
    )
    

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
            common_socket_events.show_search_status(f"Gathering resolutions for sorting...") # Initial status message
            all_hashes = [images_search_engine.get_file_hash(file_path) for file_path in all_files]
            resolutions = gather_resolutions(all_files, all_hashes, media_directory)
            # all_files_sorted = sorted(all_files, key=lambda x: resolutions[x][0] * resolutions[x][1])
            scores = [resolutions[x][0] * resolutions[x][1] for x in all_files]
            return scores

        def filter_by_proportion(all_files, text_query):
            common_socket_events.show_search_status(f"Gathering resolutions for sorting...") # Initial status message
            all_hashes = [images_search_engine.get_file_hash(file_path) for file_path in all_files]
            resolutions = gather_resolutions(all_files, all_hashes, media_directory)
            #all_files_sorted = sorted(all_files, key=lambda x: resolutions[x][0] / resolutions[x][1])
            scores = [resolutions[x][0] / resolutions[x][1] for x in all_files]
            return scores

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
            resolution = image_metadata.get('resolution')    # Returns a tuple (width, height)
        
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
                "file_data": file_data,
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
        """
        Move selected files to a target folder with automatic path conversion.
        
        This function handles moving files between folders with the following features:
        - Automatic conversion from host paths to Docker container paths
        - Validation that paths are within allowed Docker volumes
        - Automatic creation of target folder if it doesn't exist
        - Conflict resolution by adding numeric suffixes to duplicate filenames
        - Automatic moving of associated .meta files alongside the main files
        - Database update for moved files
        - Progress reporting and comprehensive error handling
        
        Args:
            data (dict): Contains 'files' (list of file paths) and 'target_folder' (destination path)
            
        Emits:
            - emit_images_page_show_error: On validation or critical errors
            - emit_images_page_move_complete: On successful completion with results
            - emit_show_search_status: Progress updates during move operation
        """
        try:
            files = data['files']
            target_folder = data['target_folder']
            
            if not files:
                socketio.emit('emit_images_page_show_error', {'message': 'No files selected to move.'})
                return
            
            # Convert host path to container path if needed
            try:
                target_folder_container = convert_host_path_to_container(target_folder)
            except ValueError as e:
                socketio.emit('emit_images_page_show_error', {'message': str(e)})
                return
            
            # Validate target folder exists
            if not os.path.exists(target_folder_container):
                try:
                    os.makedirs(target_folder_container, exist_ok=True)
                    print(f"Created target folder: {target_folder_container}")
                except Exception as e:
                    socketio.emit('emit_images_page_show_error', {
                        'message': f"Failed to create target folder: {str(e)}"
                    })
                    return
            
            if not os.path.isdir(target_folder_container):
                socketio.emit('emit_images_page_show_error', {
                    'message': f"Target path '{target_folder}' is not a directory."
                })
                return
            
            moved_count = 0
            errors = []
            
            for idx, file_path in enumerate(files):
                try:
                    # Show progress
                    common_socket_events.show_search_status(f"Moving file {idx + 1}/{len(files)}: {os.path.basename(file_path)}")
                    
                    if not os.path.exists(file_path):
                        errors.append(f"File not found: {file_path}")
                        continue
                    
                    if not os.path.isfile(file_path):
                        errors.append(f"Not a file: {file_path}")
                        continue
                    
                    base_name = os.path.basename(file_path)
                    target_path = os.path.join(target_folder_container, base_name)
                    
                    # Check if the target path already exists, add a suffix if necessary
                    if os.path.exists(target_path):
                        # If it is the same file, skip
                        if os.path.samefile(file_path, target_path):
                            print(f"Skipping {file_path} - already at destination")
                            continue
                        
                        # Generate unique filename
                        file_name, file_extension = os.path.splitext(base_name)
                        counter = 1
                        new_file_name = f"{file_name}_{counter}{file_extension}"
                        target_path = os.path.join(target_folder_container, new_file_name)
                        
                        while os.path.exists(target_path):
                            counter += 1
                            new_file_name = f"{file_name}_{counter}{file_extension}"
                            target_path = os.path.join(target_folder_container, new_file_name)
                        
                        print(f"Renamed to avoid conflict: {new_file_name}")
                    
                    # Move the file
                    import shutil
                    shutil.move(file_path, target_path)
                    moved_count += 1
                    print(f"Moved: {file_path} -> {target_path}")
                    
                    # Move associated .meta file if it exists
                    meta_file_path = file_path + ".meta"
                    if os.path.exists(meta_file_path) and os.path.isfile(meta_file_path):
                        try:
                            meta_target_path = target_path + ".meta"
                            
                            # Handle conflict for meta file if needed
                            if os.path.exists(meta_target_path):
                                # If the target meta file is identical to source, just remove source
                                if os.path.samefile(meta_file_path, meta_target_path):
                                    pass  # Already at destination
                                else:
                                    # Use same naming scheme as the main file got
                                    meta_base_name = os.path.basename(target_path) + ".meta"
                                    meta_target_path = os.path.join(os.path.dirname(target_path), meta_base_name)
                            
                            # Move the meta file
                            if not os.path.samefile(meta_file_path, meta_target_path):
                                shutil.move(meta_file_path, meta_target_path)
                                print(f"Moved associated .meta file: {meta_file_path} -> {meta_target_path}")
                        except Exception as meta_error:
                            # Don't fail the whole operation if meta file move fails
                            print(f"Warning: Failed to move .meta file for {file_path}: {meta_error}")
                    
                    # Update database entry if exists
                    file_hash = images_search_engine.get_file_hash(target_path)
                    db_item = db_models.ImagesLibrary.query.filter_by(hash=file_hash).first()
                    if db_item:
                        db_item.file_path = os.path.relpath(target_path, media_directory)
                        db_models.db.session.commit()
                    
                except Exception as e:
                    errors.append(f"{os.path.basename(file_path)}: {str(e)}")
                    print(f"Error moving {file_path}: {e}")
            
            # Send results back to client
            result_message = f"Successfully moved {moved_count} file(s)"
            if errors:
                result_message += f". {len(errors)} error(s) occurred."
            
            socketio.emit('emit_images_page_move_complete', {
                'success': True,
                'moved_count': moved_count,
                'total_count': len(files),
                'errors': errors,
                'message': result_message
            })
            
            common_socket_events.show_search_status(result_message)
            
        except Exception as e:
            print(f"Critical error in move_files: {e}")
            import traceback
            traceback.print_exc()
            socketio.emit('emit_images_page_show_error', {
                'message': f"Critical error while moving files: {str(e)}"
            })

    @socketio.on('emit_images_page_open_file_in_folder')
    def open_file_in_folder(file_path):
        file_manager.open_file_in_folder(file_path)

    # Only register delete handlers if explicitly allowed
    allow_file_deletion = os.environ.get('ALLOW_FILE_DELETION', 'false').lower() == 'true'
    
    if allow_file_deletion:
        @socketio.on('emit_images_page_send_file_to_trash')
        def send_file_to_trash(file_path):
            if os.path.isfile(file_path):
                send2trash.send2trash(file_path)
                print(f"File '{file_path}' sent to trash.")
                '''if sys.platform == "win32":    # Windows
                    # Move the file to the recycle bin
                    subprocess.run(["cmd", "/c", "del", "/q", "/f", file_path], check=True)
                elif sys.platform == "darwin":    # macOS
                    # Move the file to the trash
                    subprocess.run(["trash", file_path], check=True)
                else:    # Linux and other Unix-like OS
                    # Move the file to the trash
                    subprocess.run(["gio", "trash", file_path], check=True)'''
            else:
                print(f"Error: File '{file_path}' does not exist.")    

        @socketio.on('emit_images_page_send_files_to_trash')
        def send_files_to_trash(files):
            for file_path in files:
                send_file_to_trash(file_path)
    else:
        print("Images module: File deletion handlers disabled (ALLOW_FILE_DELETION=false)")

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
                "hash_algorithm": images_search_engine.get_hash_algorithm(),
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
        
        socketio.emit('emit_images_page_show_image_metadata_content', {"content": content, "file_path": file_path})

    @socketio.on('emit_images_page_get_external_metadata_file_content')
    def get_external_metadata_file_content(file_path):
        """
        Reads the content of the external .meta file associated with a file.
        If the file does not exist, returns an empty string.
        """
        nonlocal media_directory
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"

        content = ""
        try:
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            print(f"Read external metadata for {file_path}")
        except Exception as e:
            print(f"Error reading external metadata for {file_path}: {e}")

        return {"content": content, "file_path": file_path}

    @socketio.on('emit_images_page_save_external_metadata_file_content')
    def save_external_metadata_file_content(data):
        """
        Saves the provided content to the .meta file associated with a file.
        """
        nonlocal media_directory
        file_path = data['file_path']
        metadata_content = data['metadata_content']
        
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"

        try:
            # Ensure the directory exists before writing the file
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            print(f"Saved metadata for {file_path}")
        except Exception as e:
            print(f"Error saving metadata for {file_path}: {e}")

    @socketio.on('emit_images_page_get_full_metadata_description')
    def get_full_metadata_description(file_path):
        """
        Generates a full metadata description for a single file.
        """
        nonlocal media_directory
        full_path = os.path.join(media_directory, file_path)
        content = metadata_search_engine.generate_full_description(full_path, media_directory)
        return {"content": content, "file_path": file_path}

    common_socket_events.show_loading_status('Images module ready!')

    
