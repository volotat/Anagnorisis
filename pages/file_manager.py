import os
import pickle
import hashlib
import datetime
import shlex
import json
import torch

from src.db_models import db

def parse_terminal_command(input_string: str) -> tuple[str | None, dict]:
    """
    Parses a terminal-like command string into a command and a dictionary of arguments.

    Args:
        input_string: The string to parse, e.g., "recommendation -t 0.2 --mode strict".

    Returns:
        A tuple containing:
        - The command name (str) if parsing is successful, otherwise None.
        - A dictionary of parsed arguments.
    """
    try:
        # Use shlex.split to handle quoted arguments correctly
        parts = shlex.split(input_string)
    except ValueError:
        # shlex fails on unclosed quotes, treat as a non-command
        return None, {}

    if not parts:
        return None, {}

    command = parts[0]
    # A simple check: if the first "word" contains internal spaces, it's likely not a command.
    # shlex.split handles quotes, so this is a safe check.
    if ' ' in command:
        return None, {}

    # Check if any subsequent part looks like an argument flag.
    # This is our heuristic to decide if it's a "command-like" string.
    is_command_like = any(p.startswith('-') for p in parts[1:])
    
    # If it's just a single word (like "recommendation"), treat it as a command.
    if len(parts) == 1 and not command.startswith('-'):
         is_command_like = True

    if not is_command_like:
        return None, {}

    args = {}
    i = 1
    while i < len(parts):
        part = parts[i]
        if part.startswith('-'):
            # Check if there is a next part and it's not another flag
            if i + 1 < len(parts) and not parts[i+1].startswith('-'):
                # It's a key-value pair, e.g., "-t 0.2"
                args[part] = parts[i+1]
                i += 2
            else:
                # It's a flag without a value, e.g., "--verbose"
                args[part] = True
                i += 1
        else:
            # This part is not a flag, so we can't process it in key-value style.
            # You could assign it to a default key or ignore it.
            # For now, we'll ignore it to keep the parsing strict.
            i += 1
            
    return command, args

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
from pages.utils import convert_size, weighted_shuffle
from src.caching import TwoLevelCache 
import threading
from typing import Dict

_TLC_SINGLETONS: Dict[str, "TwoLevelCache"] = {}
_TLC_LOCK = threading.Lock()

def get_two_level_cache(cache_dir: str, **kwargs) -> "TwoLevelCache":
    """
    Return a shared TwoLevelCache instance for this cache_dir within the process.
    First caller’s kwargs win; later calls ignore differing kwargs.
    """
    abs_dir = os.path.abspath(cache_dir)
    with _TLC_LOCK:
        inst = _TLC_SINGLETONS.get(abs_dir)
        if inst is not None:
            return inst
        inst = TwoLevelCache(cache_dir=abs_dir, **kwargs)
        _TLC_SINGLETONS[abs_dir] = inst
        return inst

class FileManager:
    def __init__(self, cfg, media_directory, engine=None, module_name="FileManager", media_formats=None, socketio=None, db_schema=None):
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
        self.engine = engine

        # self.cached_file_list = engine.cached_file_list
        # self.cached_file_hash = engine.cached_file_hash
        # self.cached_metadata = engine.cached_metadata

        # Folder tree cache (persisted across many instances of FileManager as a singleton object)
        cache_folder = os.path.join(cfg.main.cache_path, "file_manager")
        self._fast_cache = get_two_level_cache(cache_dir=cache_folder, name="file_manager")

    def show_status(self, message):
        self.common_socket_events.show_search_status(message)

    def resolve_media_path(self, path):
        """Return the absolute path for the given media directory and relative path."""
        if path == "":
            return self.media_directory
        return os.path.abspath(os.path.join(self.media_directory, '..', path))

    def _build_folder_tree_cached(self, path: str, media_extensions: set[str]) -> dict:
        """
        Build folder tree with per-directory caching keyed by (dir_path, dir_mtime, contents_hash).
        The contents_hash ensures cache invalidation when subfolders are moved in/out.
        """
        name = os.path.basename(path)
        
        # Scan directory to get current contents
        try:
            entries = list(os.scandir(path))
        except FileNotFoundError:
            return {"name": name, "num_files": 0, "total_files": 0, "subfolders": {}}
        except Exception:
            return {"name": name, "num_files": 0, "total_files": 0, "subfolders": {}}
        
        # Cache is temporarily disabled due to issues with subfolders file counts not updating correctly
        # TODO: Fix this one day 

        # # Build a signature of the directory contents (sorted child names)
        # # This ensures that moving subfolders/files in/out invalidates the cache
        # child_names = sorted(e.name for e in entries)
        # contents_sig = hashlib.md5(",".join(child_names).encode()).hexdigest()[:8]
        
        # # Try to get cached result for this directory
        # # Note: We rely on contents_sig instead of mtime for cache invalidation
        # # because mtime updates can be delayed on some filesystems (especially Docker volumes)
        # ext_sig = ",".join(sorted(media_extensions)) if media_extensions else "-"
        # cache_key = f"FOLDER_TREE::{path}|{contents_sig}|{ext_sig}"
        
        # cached_result = self._fast_cache.get(cache_key)
        # if cached_result is not None:
        #     return cached_result
        
        # Build the folder structure from the entries we already scanned
        num_files = 0
        subfolders = {}

        for entry in entries:
            try:
                if entry.is_file(follow_symlinks=False):
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in media_extensions:
                        num_files += 1
                elif entry.is_dir(follow_symlinks=False):
                    # Recursively get subfolder structure (which will use its own cache)
                    child = self._build_folder_tree_cached(entry.path, media_extensions)
                    subfolders[entry.name] = child
            except Exception:
                # Skip entries that vanish mid-scan or are inaccessible
                continue

        total_files = num_files + sum(sf["total_files"] for sf in subfolders.values())
        result = {"name": name, "num_files": num_files, "total_files": total_files, "subfolders": subfolders}
        
        # Cache the result for this directory
        # self._fast_cache.set(cache_key, result)
        
        return result


    def get_folders(self, path = ""):
        current_path = self.resolve_media_path(path)
        folder_path = os.path.relpath(current_path, os.path.join(self.media_directory, '..')) + os.path.sep
        print('folder_path', folder_path)

        # Extract subfolders structure from the path into a dict
        #folders = get_folder_structure(self.media_directory, self.media_formats)

        # Build from cache-aware walker
        media_exts = set(self.media_formats)
        folders = self._build_folder_tree_cached(self.media_directory, media_exts)

        # Extract main folder name
        main_folder_name = os.path.basename(os.path.normpath(self.media_directory))
        folders['name'] = main_folder_name

        return {"folders": folders, "folder_path": folder_path}
    
    def _get_hashes_with_progress(self, files: list[str]) -> list[str]:
        """
        Gathers hashes for all files
        """
        self.show_status(f"Gathering hashes for {len(files)} files.")
        
        hashes = []
        last_shown_time = 0
        total_files = len(files)

        for ind, file_path in enumerate(files):
            current_time = time.time()
            
            # Show progress every second
            if current_time - last_shown_time >= 1:
                percent = (ind + 1) / total_files * 100
                self.show_status(f"Gathering hashes: {ind+1}/{total_files} ({percent:.2f}%)")
                last_shown_time = current_time
                
            hashes.append(self.engine.get_file_hash(file_path))

        return hashes

    def _list_dir_cached(self, path: str, media_exts: set[str]) -> tuple[list[str], list[str]]:
        """
        Return (files_in_dir, subdirs) for path using TwoLevelCache keyed by dir mtime.
        """
        try:
            st = os.stat(path, follow_symlinks=False)
        except FileNotFoundError:
            return [], []
        except Exception:
            return [], []

        ext_sig = ",".join(sorted(media_exts)) if media_exts else "-"
        key = f"MEDIAFILES_OF:{path}|{st.st_mtime_ns}|{ext_sig}"

        cached = self._fast_cache.get(key)
        if cached is not None:
            return cached

        files: list[str] = []
        subdirs: list[str] = []
        try:
            with os.scandir(path) as it:
                for e in it:
                    try:
                        if e.is_file(follow_symlinks=False):
                            ext = os.path.splitext(e.name)[1].lower()
                            if not media_exts or ext in media_exts:
                                files.append(os.path.join(path, e.name))
                        elif e.is_dir(follow_symlinks=False):
                            subdirs.append(e.path)
                    except Exception:
                        # Skip entries that change/disappear mid-scan
                        continue
        except Exception:
            return [], []

        self._fast_cache.set(key, (files, subdirs))
        return files, subdirs

    def _walk_files_cached(self, root: str, media_exts: set[str]) -> list[str]:
        files, subdirs = self._list_dir_cached(root, media_exts)
        all_files = list(files)
        for d in subdirs:
            all_files.extend(self._walk_files_cached(d, media_exts))
        return all_files

    def get_files(self, path = "", pagination = 0, limit = 100, text_query = None, seed = None, filters: dict = {}, get_file_info = None, update_model_ratings = None, mode = 'file-name', order = 'most-relevant', temperature = 0, evaluator_hash = None):

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

        self.show_status(f"Searching for files in '{folder_path}'.")

        all_files = []
        
        # Walk with cache 1.5s for 66k files
        all_files = self._walk_files_cached(current_path, self.media_formats)

        # self.show_status(f"Gathering hashes for {len(all_files)} files.")
        # all_hashes = self._get_all_hashes_with_progress(all_files)

        # Sort files by text or file query
        self.show_status(f"Sorting files by: \"{text_query}\"")

        scores = np.zeros(len(all_files), dtype=np.float32)
        if text_query and len(text_query) > 0:
            # use first word as filter name
            filter_name = text_query


            # If there is a file path used as a query, this file exists and within specified formats, sort files by similarity to that file
            if os.path.isfile(text_query) and text_query.lower().endswith(tuple(self.media_formats)):
                if filters["by_file"] is not None:
                    scores = filters["by_file"](all_files, text_query) # Set the filter for other components
            
            # Custom sorts
            elif filter_name and (filter_name not in ["by_file", "by_text"]) and (filter_name in filters):
                if filters[filter_name] is not None:
                    scores = filters[filter_name](all_files, text_query) # Set the filter for other components

            # In any other case sort by text query
            else:
                if filters["by_text"] is not None:
                    scores = filters["by_text"](all_files, text_query, mode=mode)
                else:
                    raise ValueError("No way to filter files. No 'by_text' filter provided.")

        if scores is None or len(scores) == 0:
            raise ValueError("No scores calculated for files. Cannot sort.")

        #if type(scores) not in [list, torch.Tensor, np.ndarray]:
        #    raise ValueError("Scores should be a flat list or array of floats.")

        if type(scores) is torch.Tensor:
            scores = scores.cpu().numpy().tolist()

        if type(scores) is np.ndarray:
            scores = scores.tolist()

        # print(f"Scores calculated for {len(scores)} files.")

        # print(f"Sorting {len(all_files)} files with temperature={temperature}, order={order}...")
        indices = weighted_shuffle(scores, temperature=temperature) 

        if order == 'most-relevant':
            pass  # already sorted in descending order
        elif order == 'least-relevant':
            indices = indices[::-1]  # reverse for ascending order
        else:
            raise ValueError("order must be 'most-relevant' or 'least-relevant'")

        sorted_files = [all_files[i] for i in indices]
        sorted_scores = [scores[i] for i in indices]

        # Select files for the current page
        page_files = sorted_files[pagination:pagination+limit]
        page_files_scores = sorted_scores[pagination:pagination+limit]
        
        # print(f'page_files {pagination}:{limit}', page_files)

        self.show_status(f"Gathering hashes for {len(page_files)} files.")
        page_hashes = self._get_hashes_with_progress(page_files) #[self.engine.get_file_hash(file_path) for file_path in page_files]

        # Extract DB data for the relevant batch of files
        self.show_status(f"Extracting database info for relevant files.")
        db_items = self.db_schema.query.filter(self.db_schema.hash.in_(page_hashes)).all()
        db_items_map = {item.hash: item for item in db_items}

        # Keep DB path in sync if file was moved/renamed for all files in the page
        self.show_status(f"Syncing database paths for relevant files.")
        db_updated = False
        for ind, full_path in enumerate(page_files):
            file_hash = page_hashes[ind]
            if file_hash in db_items_map:
                db_item = db_items_map[file_hash]
                rel_path = os.path.relpath(full_path, self.media_directory)
                if db_item.file_path != rel_path:
                    db_item.file_path = rel_path
                    db_updated = True

        if db_updated:
            db.session.commit()

        # Check if there files without model rating or with stale model hash
        no_model_rating_files = [os.path.join(self.media_directory, item.file_path) for item in db_items if item.file_path is not None and (
            item.model_rating is None or (evaluator_hash is not None and item.model_hash != evaluator_hash)
        )]
        # print('no_model_rating_files size', len(no_model_rating_files))

        # Add files that are not in the database yet
        new_files_list = [file_path for file_path in page_files if self.engine.get_file_hash(file_path) not in db_items_map]
        no_model_rating_files.extend(new_files_list)

        # Remove from list files that do not exist anymore
        no_model_rating_files = [file_path for file_path in no_model_rating_files if os.path.exists(file_path)]

        self.show_status(f"Upgrade ratings of {len(no_model_rating_files)} files.")
        if len(no_model_rating_files) > 0:
            # Update the model ratings of all current files
            update_model_ratings(no_model_rating_files)

        files_data = []
        
        # self.show_status(f"Extracting metadata for {len(page_files)} files.")

        for ind, full_path in enumerate(page_files):
            self.show_status(f"Extracting metadata for {ind+1}/{len(page_files)} files.")

            # page_hashes = self._get_hashes_with_progress(page_files)
            page_hashes = [self.engine.get_file_hash(file_path) for file_path in page_files]

            file_path = os.path.relpath(full_path, self.media_directory)
            basename = os.path.basename(full_path)
            file_size = os.path.getsize(full_path)
            file_hash = page_hashes[ind]

            #audiofile_data = self.get_metadata(full_path)

            
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
                "has_meta": os.path.exists(full_path + ".meta"),
                "search_score": page_files_scores[ind]

                # "user_rating": user_rating,
                # "model_rating": model_rating,
                # "audiofile_data": audiofile_data,
                # "length": convert_length(audiofile_data['duration']),
                # "last_played": last_played
            }
            files_data.append(data)
        
        # Save all extracted metadata to the cache
        # self.cached_metadata.save_metadata_cache()

        sorted_files_paths = [os.path.relpath(file_path, self.media_directory) for file_path in sorted_files]

        self.show_status(f'{len(sorted_files)} files processed in {time.time() - start_time:.4f} seconds.')

        print(f'get_files returning {len(files_data)} files data.')

        # Check if all the data is serializable and if not print the non-serializable data
        for file_data in files_data:
            try:
                json.dumps(file_data, indent=2)
            except (TypeError, OverflowError) as e:
                raise ValueError(f'Non-serializable data found in file: {file_data["full_path"]}, error: {e}')

        return {
            "files_data": files_data, 
            "folder_path": folder_path, 
            "total_files": len(sorted_files), 
            "all_files_paths": sorted_files_paths
            }

