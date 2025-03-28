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


def get_folder_structure(folder_path, image_extensions=None):
    # Check if directory exists and return None if not
    if not os.path.isdir(folder_path):
        return None
  
    def count_files(folder):
        return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in image_extensions)

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