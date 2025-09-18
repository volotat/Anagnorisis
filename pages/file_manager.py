import os
import pickle
import hashlib
import datetime

###########################################
# File Hash Caching

class CachedFileHash:
    def __init__(self, cache_file_path):
        self.cache_file_path = cache_file_path
        self.file_hash_cache = {}
        self.load_hash_cache()

    def load_hash_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as cache_file:
                self.file_hash_cache = pickle.load(cache_file)
            
            # Remove entries older than three months
            three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
            self.file_hash_cache = {k: v for k, v in self.file_hash_cache.items() if v[2] > three_months_ago}

    def save_hash_cache(self):
        # Save the updated cache to the file
        with open(self.cache_file_path, 'wb') as cache_file:
            pickle.dump(self.file_hash_cache, cache_file)

    def get_file_hash(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get the last modified time of the file
        last_modified_time = os.path.getmtime(file_path)
        
        # Check if the file is in the cache and if the last modified time matches
        if file_path in self.file_hash_cache:
            cached_last_modified_time, cached_hash, timestamp = self.file_hash_cache[file_path]
            if cached_last_modified_time == last_modified_time:
                return cached_hash
        
        # If not in cache or file has been modified, calculate the hash
        with open(file_path, "rb") as f:
            bytes = f.read()  # Read the entire file as bytes
            file_hash = hashlib.md5(bytes).hexdigest()
        
        # Update the cache
        self.file_hash_cache[file_path] = (last_modified_time, file_hash, datetime.datetime.now())

        return file_hash

###########################################
# File List Caching

class CachedFileList:
    def __init__(self, cache_file_path):
        self.cache_file_path = cache_file_path
        self.file_list_cache = {}
        self.cache_changed = False # Flag to track changes
        self.load_file_list_cache()

    def load_file_list_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as cache_file:
                self.file_list_cache = pickle.load(cache_file)

            # Remove entries older than three months
            three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
            self.file_list_cache = {k: v for k, v in self.file_list_cache.items() if v[2] > three_months_ago}

    def save_file_list_cache(self):
        with open(self.cache_file_path, 'wb') as cache_file:
            pickle.dump(self.file_list_cache, cache_file)

    def get_files_in_folder(self, folder_path, media_formats):
        # Get the last modified time of the folder
        folder_last_modified_time = os.path.getmtime(folder_path)
        
        # Check if the folder is in the cache and if the last modified time matches
        if folder_path in self.file_list_cache:
            cached_last_modified_time, cached_file_list, timestamp = self.file_list_cache[folder_path]
            if cached_last_modified_time == folder_last_modified_time:
                return cached_file_list
        
        # If not in cache or folder has been modified, list the files in the folder
        file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(tuple(media_formats))]
        
        # Update the cache with the current timestamp and file list
        self.file_list_cache[folder_path] = (folder_last_modified_time, file_list, datetime.datetime.now())
        self.cache_changed = True  # Set the flag to indicate changes
        
        return file_list

    def get_all_files(self, current_path, media_formats):
        # Load the cache from the file if it exists and file_list_cache is empty
        if not self.file_list_cache:
            self.load_file_list_cache()

        all_files = []
        for root, dirs, files in os.walk(current_path):
            all_files.extend(self.get_files_in_folder(root, media_formats))

        # Save the updated cache to the file
        if self.cache_changed:
            self.save_file_list_cache()
        
        return all_files

###########################################
# Cached Metadata (Hash-Dependent Version)

class CachedMetadata:
    def __init__(self, cache_file_path, metadata_func):
        self.cache_file_path = cache_file_path
        self.metadata_cache = {}
        self.load_metadata_cache()
        self.metadata_func = metadata_func

    def load_metadata_cache(self):
        if os.path.exists(self.cache_file_path):
            with open(self.cache_file_path, 'rb') as cache_file:
                self.metadata_cache = pickle.load(cache_file)

            # Remove entries older than three months
            three_months_ago = datetime.datetime.now() - datetime.timedelta(days=90)
            self.metadata_cache = {k: v for k, v in self.metadata_cache.items() if v[2] > three_months_ago}

    def save_metadata_cache(self):
        with open(self.cache_file_path, 'wb') as cache_file:
            pickle.dump(self.metadata_cache, cache_file)

    def get_metadata(self, file_path, file_hash):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Get the last modified time of the file (still used for time-based invalidation)
        last_modified_time = os.path.getmtime(file_path)

        cache_key = file_hash # Use file_hash as the cache key

        # Check if metadata for file_hash is in the cache and if the last modified time is still valid
        if cache_key in self.metadata_cache:
            cached_last_modified_time, cached_metadata, timestamp = self.metadata_cache[cache_key]
            if cached_last_modified_time == last_modified_time: # Keep time-based invalidation
                return cached_metadata

        # If not in cache or file has been modified, extract metadata using metadata_func
        metadata = self.metadata_func(file_path) # Method expected to return a dictionary of metadata attributes
        metadata['file_path'] = file_path

        # Update the cache using file_hash as the key
        self.metadata_cache[cache_key] = (last_modified_time, metadata, datetime.datetime.now())
        return metadata


def get_folder_structure(folder_path, media_extensions=None):
    # Check if directory exists and return None if not
    if not os.path.isdir(folder_path):
        return None
  
    def count_files(folder):
        return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in media_extensions)

    def build_structure(path):
        folder_dict = {
        'name': os.path.basename(path),
        'num_files': count_files(path),
        'total_files': 0,
        'subfolders': {}
        }
        folder_dict['total_files'] = folder_dict['num_files']
        
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                subfolder_structure = build_structure(subfolder_path)
                folder_dict['subfolders'][subfolder] = subfolder_structure
                folder_dict['total_files'] += subfolder_structure['total_files']
        
        return folder_dict
    return build_structure(folder_path)

import subprocess
import sys

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




# ----------------------------------------------------------------------------------

from pathlib import Path

class PathTraversalError(Exception):
    pass

def resolve_subpath(base_dir: str, user_path: str | None) -> Path:
    """
    Safely resolve user_path inside base_dir. Raises PathTraversalError if escape attempt.
    Empty / None user_path returns base_dir.
    """
    base = Path(base_dir).resolve()
    candidate = base if not user_path else (base / user_path)
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base)  # raises ValueError if outside
    except Exception:
        raise PathTraversalError(f"Invalid path: {user_path}")
    return resolved

# filters = {
#    "by_file": by_file_sort_function,
#    "by_text": by_text_sort_function,
#    "custom": custom_sort_function,
#    ...
# }

from pages.socket_events import CommonSocketEvents
import time
import numpy as np
from pages.utils import convert_size


class FileManager:
    def __init__(self, media_directory, engine=None, module_name="FileManager", media_formats=None, socketio=None, db_schema=None):
        assert media_directory is not None, "Media directory must be specified"
        assert engine is not None, "Engine must be specified"
        assert socketio is not None, "SocketIO instance must be specified"
        assert media_formats is not None and len(media_formats) > 0, "Media formats must be specified"
        assert db_schema is not None, "Database schema must be specified"

        self.media_directory = media_directory
        self.module_name = module_name
        self.media_formats = media_formats
        self.common_socket_events = CommonSocketEvents(socketio)
        self.db_schema = db_schema

        self.cached_file_list = engine.cached_file_list
        self.cached_file_hash = engine.cached_file_hash
        self.cached_metadata = engine.cached_metadata

    def show_status(self, message):
        self.common_socket_events.show_search_status(message)

    def resolve_media_path(self, path):
        """Return the absolute path for the given media directory and relative path."""
        if path == "":
            return self.media_directory
        return os.path.abspath(os.path.join(self.media_directory, '..', path))

    def get_folders(self, path = ""):
        current_path = self.resolve_media_path(path)
        folder_path = os.path.relpath(current_path, os.path.join(self.media_directory, '..')) + os.path.sep
        print('folder_path', folder_path)

        # Extract subfolders structure from the path into a dict
        folders = get_folder_structure(self.media_directory, self.media_formats)

        # Extract main folder name
        main_folder_name = os.path.basename(os.path.normpath(self.media_directory))
        folders['name'] = main_folder_name

        return {"folders": folders, "folder_path": folder_path}

    def get_files(self, path = "", pagination = 0, limit = 100, text_query = None, seed = None, filters: dict = {}, get_file_info = None, update_model_ratings = None):

        if self.media_directory is None:
            self.show_status(f"{self.module_name} media folder is not set.")
            return

        start_time = time.time()

        if seed is not None:
            np.random.seed(int(seed))

        # If path is not specified, use the media directory as default
        #if path == "": 
        #    path = os.path.realpath(self.media_directory)

        # Directory Traversal Prevention ---
        resolve_subpath(self.media_directory, path)
        
        current_path = self.resolve_media_path(path)

        
        folder_path = os.path.relpath(current_path, os.path.join(self.media_directory, '..')) + os.path.sep
        print('get_files', 'folder_path', folder_path)

        self.show_status(f"Searching for music files in '{folder_path}'.")

        all_files = []
        
        # Walk with cache 1.5s for 66k files
        all_files = self.cached_file_list.get_all_files(current_path, self.media_formats)

        self.show_status(f"Gathering hashes for {len(all_files)} files.")
        
        # Initialize the last shown time
        last_shown_time = 0

        # Get the hash of each file
        all_hashes = []
        for ind, file_path in enumerate(all_files):
            current_time = time.time()
            if current_time - last_shown_time >= 1:
                self.show_status(f"Gathering hashes for {ind+1}/{len(all_files)} files.")
                last_shown_time = current_time
        
            all_hashes.append(self.cached_file_hash.get_file_hash(file_path))

        self.cached_file_hash.save_hash_cache()

        # Update the hashes in the database if there are instances without a hash
        # self.show_status(f"Update hashes for imported files...")
        # update_none_hashes_in_db(all_files, all_hashes)


        # Sort music files by text or file query
        self.show_status(f"Sorting music files by: \"{text_query}\"")
        if text_query and len(text_query) > 0:
            filter_name = text_query.lower().strip()

            # If there is a file path used as a query, this file exists and within specified formats, sort files by similarity to that file
            if os.path.isfile(text_query) and text_query.lower().endswith(tuple(cfg.music.media_formats)):
                if filters["by_file"] is not None:
                    all_files = filters["by_file"](all_files, text_query) # Set the filter for other components
            
            # Custom sorts
            elif (filter_name not in ["by_file", "by_text"]) and (filter_name in filters):
                if filters[filter_name] is not None:
                    all_files = filters[filter_name](all_files, text_query)

            # In any other case sort by text query
            else:
                if filters["by_text"] is not None:
                    all_files = filters["by_text"](all_files, text_query)

        #all_files = sorted(all_files, key=os.path.basename)

        
        # Extracting metadata for relevant batch of music files
        self.show_status(f"Extracting metadata for relevant batch of files.")
        page_files = all_files[pagination:limit]
        print(f'page_files {pagination}:{limit}', page_files)

        page_hashes = [self.cached_file_hash.get_file_hash(file_path) for file_path in page_files]

        # Extract DB data for the relevant batch of music files
        db_items = self.db_schema.query.filter(self.db_schema.hash.in_(page_hashes)).all()
        db_items_map = {item.hash: item for item in db_items}


        # Check if there files without model rating
        no_model_rating_files = [os.path.join(self.media_directory, item.file_path) for item in db_items if item.model_rating is None and item.file_path is not None]
        print('no_model_rating_files size', len(no_model_rating_files))

        # Add files that are not in the database yet
        new_files_list = [file_path for file_path in page_files if self.cached_file_hash.get_file_hash(file_path) not in db_items_map]
        no_model_rating_files.extend(new_files_list)

        # Remove from list files that do not exist anymore
        no_model_rating_files = [file_path for file_path in no_model_rating_files if os.path.exists(file_path)]

        if len(no_model_rating_files) > 0:
            # Update the model ratings of all current files
            update_model_ratings(no_model_rating_files)


        files_data = []

        for ind, full_path in enumerate(page_files):
            file_path = os.path.relpath(full_path, self.media_directory)
            basename = os.path.basename(full_path)
            file_size = os.path.getsize(full_path)
            file_hash = page_hashes[ind]

            #audiofile_data = self.cached_metadata.get_metadata(full_path, file_hash)

            
            #resolution = get_image_resolution(full_path)  # Returns a tuple (width, height)
            
            user_rating = None
            model_rating = None
            last_played = "Never"

            # if file_hash in db_items_map:
            #     db_item = db_items_map[file_hash]
            #     user_rating = db_item.user_rating
            #     model_rating = db_item.model_rating

            #     # convert datetime to string
            #     if db_item.last_played:
            #         last_played_timestamp = db_item.last_played.timestamp() if db_item.last_played else None
            #         last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())
                
            data = {
                "type": "file",
                "full_path": full_path,
                "file_path": file_path,
                "base_name": basename,
                "hash": file_hash,
                "file_size": convert_size(file_size),
                "file_info": get_file_info(full_path, file_hash),

                # "user_rating": user_rating,
                # "model_rating": model_rating,
                # "audiofile_data": audiofile_data,
                # "length": convert_length(audiofile_data['duration']),
                # "last_played": last_played
            }
            files_data.append(data)
        
        # Save all extracted metadata to the cache
        self.cached_metadata.save_metadata_cache()

        # Extract subfolders structure from the path into a dict
        # folders = get_folder_structure(self.media_directory, self.media_formats)

        # # Return "No files in the directory" if the path not exist or empty.
        # if not folders:
        #     self.show_status(f"No files in the directory.")
        #     self.socketio.emit('emit_music_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": 0, "folders": folders, "all_files_paths": []})
        #     return

        # # Extract main folder name
        # main_folder_name = os.path.basename(os.path.normpath(self.media_directory))
        # folders['name'] = main_folder_name

        #print(folders)

        all_files_paths = [os.path.relpath(file_path, self.media_directory) for file_path in all_files]
        
        #socketio.emit('emit_music_page_show_files', {"files_data": files_data, "folder_path": folder_path, "total_files": len(all_files), "folders": folders, "all_files_paths": all_files_paths})

        self.show_status(f'{len(all_files)} files processed in {time.time() - start_time:.4f} seconds.')

        return {
            "files_data": files_data, 
            "folder_path": folder_path, 
            "total_files": len(all_files), 
            "all_files_paths": all_files_paths
            }

