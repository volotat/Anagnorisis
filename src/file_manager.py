import os
import json
import pickle
import hashlib
import datetime
import concurrent.futures
import torch

import anagnorisis_core.storage.virtual_file_system as vfs
import fs
import fs.opener

from src.db_models import db
import logging

# --- LOGGING CONFIGURATION FOR THIS MODULE ---
# Choose verbosity: 
#   - logging.DEBUG    (Show all logs, including path resolution and cache hits)
#   - logging.INFO     (Show standard initialization and milestones)
#   - logging.WARNING  (Hide standard info, show only warnings/errors)
#   - logging.CRITICAL (Virtually disable all logging)
LOG_LEVEL = logging.INFO 

# Create a scoped logger specifically for this module
logger = logging.getLogger("FileManager")
logger.setLevel(LOG_LEVEL)

# Prevent double-logging if your main Flask app has a root logger
logger.propagate = False

# Configure the console handler specifically for this logger
if not logger.handlers:
    console_handler = logging.StreamHandler()
    
    # Set a simple formatter for console output
    formatter = logging.Formatter('%(levelname)-5s [%(name)s] %(message)s')
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
# ---------------------------------------------


# Path handling — including the traversal guard — lives in the core, so the
# CLI and the annotator are protected by the same code as the web routes.
from anagnorisis_core.storage.file_paths import (  # noqa: F401  (re-exported)
    PathTraversalError, get_folder_structure, open_file_in_folder, resolve_subpath,
)


# ----------------------------------------------------------------------------------

from pathlib import Path

from src.socket_events import CommonSocketEvents
import time
import numpy as np
from src.utils import convert_size
from anagnorisis_core.search.ranking import weighted_shuffle
from anagnorisis_core.storage.caching import get_two_level_cache
from anagnorisis_core.storage.file_walker import get_file_walker
import src.db_models as db_models
from src.db_models import FilesLibrary

import fs


class FileManager:
    def __init__(self, app, cfg, media_directory, engine=None, module_name="FileManager", media_formats=None, socketio=None, db_schema=None):
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

        # Traversal, the directory cache and server availability are shared
        # process-wide — see src/file_walker.py.
        self._walker = get_file_walker(app.user_cfg.servers, cfg.main.cache_path)
        self.servers = self._walker.servers

        # Folder tree cache (persisted across many instances of FileManager as a singleton object)
        cache_folder = os.path.join(cfg.main.cache_path, "file_manager")
        self._fast_cache = get_two_level_cache(cache_dir=cache_folder, name="file_manager")

    def show_status(self, message, force=False):
        self.common_socket_events.show_search_status(message, force=force)

    def resolve_media_path(self, path):
        """Return the absolute path for the given media directory and relative path."""
        if path == "":
            return self.media_directory
        return path

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
                if entry.name.startswith('.'):
                    continue
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


    def sync_file_paths(self) -> int:
        """Walk the media directory and correct any stale file_path values in the
        DB by comparing hashes.  Returns the number of rows updated.

        Useful after files are moved or renamed so the DB stays consistent.
        Uses the engine's hash cache, so repeated calls are fast.
        """
        if self.media_directory is None:
            return 0

        all_files = self._walk_files_cached(self.media_directory, set(self.media_formats))
        if not all_files:
            return 0

        all_hashes = [self.engine.get_file_hash(f) for f in all_files]
        hash_to_rel = {
            h: os.path.relpath(p, self.media_directory)
            for h, p in zip(all_hashes, all_files)
            if h is not None
        }

        BATCH = 500
        disk_hashes = list(hash_to_rel.keys())
        db_items = []
        for i in range(0, len(disk_hashes), BATCH):
            db_items.extend(self.db_schema.query.filter(
                self.db_schema.hash.in_(disk_hashes[i:i + BATCH])
            ).all())

        updated = 0
        for item in db_items:
            new_rel = hash_to_rel.get(item.hash)
            if new_rel and item.file_path != new_rel:
                item.file_path = new_rel
                updated += 1

        if updated:
            db.session.commit()

        return updated

    def get_unrated_files(self, evaluator_hash: str | None = None) -> list[str]:
        """Walk all files and return full paths of files that need rating.

        A file needs rating if its hash has no DB row, or its DB row has
        model_rating IS NULL, or its model_hash doesn't match evaluator_hash
        (i.e. the model was updated).

        Returns:
            List of full file paths that are unrated or stale-rated.
        """
        logger.info(f"[FileManager] Checking for unrated files with evaluator hash: {evaluator_hash}")

        # self.sync_file_paths()

        all_files = self._walk_files_cached("/", set(self.media_formats))
        if not all_files:
            return []
        
        logger.info(f"[FileManager] Found {len(all_files)} files in media directory.")

        # all_hashes = [self.engine.get_file_hash(f) for f in all_files]

        # # First occurrence wins for duplicate hashes (e.g. identical files)
        # hash_to_full: dict[str, str] = {}
        # for h, f in zip(all_hashes, all_files):
        #     if h is not None and h not in hash_to_full:
        #         hash_to_full[h] = f

        # disk_hashes = list(hash_to_full.keys())
        # if not disk_hashes:
        #     return []

        # Find which files are already rated with the current model
        rated_paths: list[str] = []
        stale_paths: list[str] = []
        BATCH = 500
        for i in range(0, len(all_files), BATCH):
            batch = all_files[i:i + BATCH]

            # Up-to-date: rated with the current model hash — exclude entirely
            q_current = FilesLibrary.query.with_entities(FilesLibrary.file_path).filter(
                FilesLibrary.file_path.in_(batch),
                FilesLibrary.model_rating.isnot(None),
            )
            if evaluator_hash is not None:
                q_current = q_current.filter(FilesLibrary.model_hash == evaluator_hash)
            rated_paths.extend(row.file_path for row in q_current.all())

            # Stale: rated but with a different (outdated) model hash
            if evaluator_hash is not None:
                q_stale = FilesLibrary.query.with_entities(FilesLibrary.file_path).filter(
                    FilesLibrary.file_path.in_(batch),
                    FilesLibrary.model_rating.isnot(None),
                    FilesLibrary.model_hash != evaluator_hash,
                )
                stale_paths.extend(row.file_path for row in q_stale.all())

        # Create list of new files that are not in the DB at all 
        new_files = list(set(all_files) - set(rated_paths) - set(stale_paths))

        # Unrated files (no rating at all) come first, stale-rated files second
        logger.info(f"[FileManager] Found {len(rated_paths)} rated files, {len(stale_paths)} stale-rated files, and {len(new_files)} new files.")
        return new_files + stale_paths

    # def list_all_files(self) -> list[str]:
    #     """Return all media file paths currently on disk, local files only.

    #     OmniDescriptor's auto-description pipeline downloads the entire file,
    #     so this walker is restricted to `osfs:///` mounts. Remote servers are
    #     intentionally excluded — their descriptions are populated only by the
    #     MemorySystem path when the user explicitly rates a remote file.
    #     """
    #     if self.media_directory is None:
    #         return []
    #     return self._walk_files_cached("osfs:///mnt/media/", set(self.media_formats))

    # def get_folders(self, path = ""):
    #     # current_path = self.resolve_media_path(path)
    #     # folder_path = os.path.relpath(current_path, os.path.join(self.media_directory, '..')) + os.path.sep
    #     folder_path = self.resolve_media_path(path)

    #     # Extract subfolders structure from the path into a dict
    #     #folders = get_folder_structure(self.media_directory, self.media_formats)

    #     # Build from cache-aware walker
    #     media_exts = set(self.media_formats)
    #     folders = self._build_folder_tree_cached(self.media_directory, media_exts)

    #     # Extract main folder name
    #     main_folder_name = os.path.basename(os.path.normpath(self.media_directory))
    #     folders['name'] = main_folder_name

    #     return {"folders": folders, "folder_path": folder_path}

    def get_folders(self, path = ""):
        # Determine the active folder path
        active_path = self.resolve_media_path(path)
        media_exts = set(self.media_formats)

        # def _get_file_counts(dir_path: str):
        #     """Returns (num_files, total_files) using the cached filesystem scanners."""
        #     try:
        #         files, subdirs = self._list_dir_cached(dir_path, media_exts)
        #         num_files = len(files)
        #         # Only perform deep recursive walking for local osfs to prevent network delays
        #         if dir_path.startswith("osfs://"):
        #             total_files = num_files + sum(len(self._walk_files_cached(sd, media_exts)) for sd in subdirs)
        #         else:
        #             total_files = num_files
        #         return num_files, total_files
        #     except Exception:
        #         return 0, 0

        def _is_ancestor_or_self(parent_path: str, child_path: str) -> bool:
            """Checks if parent_path is an ancestor of or equal to child_path."""
            p = parent_path.rstrip('/') + '/'
            c = child_path.rstrip('/') + '/'
            return c.startswith(p)

        def _build_tree_node(display_name: str, full_path: str, node_type="folder") -> dict:
            """Recursively builds the tree nodes, expanding only along the active path."""
            formatted_full_path = full_path.rstrip('/') + '/'
            
            try:
                base_url, path_in_fs = vfs.resolve_base_and_path_from_url(formatted_full_path)
            except Exception:
                base_url, path_in_fs = "", "/"

            # Standardize directory paths with a trailing slash
            if path_in_fs and not path_in_fs.endswith('/'):
                path_in_fs += '/'

            # num_files, total_files = _get_file_counts(full_path)

            node = {
                "display_name": display_name,
                "full_path": formatted_full_path,
                "base_url": base_url,
                "path_in_fs": path_in_fs,
                "type": node_type,
                # "num_files": num_files,
                # "total_files": total_files,
                "subfolders": []
            }

            # Only expand and fetch subfolders if this directory lies on the active path
            if _is_ancestor_or_self(full_path, active_path):
                _, subdirs = self._list_dir_cached(full_path, media_exts)
                for sd in subdirs:
                    sd_name = os.path.basename(sd.rstrip('/'))
                    child_node = _build_tree_node(sd_name, sd, "folder")
                    node["subfolders"].append(child_node)

            return node

        # Assemble the root structure
        root_node = {
            "display_name": "All Files",
            "full_path": "/",
            "type": "root",
            "subfolders": []
        }

        # Populate the first level (servers) dynamically using allowed roots
        for server in self.servers:
            display_name = server.name
            server_node = _build_tree_node(display_name, server.url, "server")

            server_node["is_available"] = self._walker.availability(server.url)

            root_node["subfolders"].append(server_node)

        return {"folders": root_node, "folder_path": active_path}
    
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

    def _list_dir_cached(self, path: str, media_exts: set[str], active_fs=None) -> tuple[list[str], list[str]]:
        return self._walker.list_dir(path, media_exts, active_fs=active_fs)

    def _walk_files_cached(self, path: str, media_exts: set[str]) -> list[str]:
        return self._walker.walk(path, media_exts, status_callback=self.show_status)

    def get_files(self, path = "", pagination = 0, limit = 100, text_query = None, seed = None, filters: dict = {}, get_file_info = None, mode = 'file-name', order = 'most-relevant', temperature = 0):

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
        #resolve_subpath(self.media_directory, path)
        logger.debug(f"get_files path: {path}" + (" (Empty)" if path == "" else ""))
        
        current_path = self.resolve_media_path(path)
        logger.debug(f"get_files current_path: {current_path}")

        folder_path = current_path
        # folder_path = os.path.relpath(current_path, os.path.join(self.media_directory, '..')) + os.path.sep
        # logger.debug(f"get_files folder_path: {folder_path}")

        self.show_status(f"Searching for files in '{folder_path}'.")

        all_files = self._walk_files_cached(current_path, self.media_formats)

        if len(all_files) == 0:
            self.show_status(f"No files found in '{folder_path}'.")
            return {
                "files_data": [],
                "folder_path": folder_path,
                "total_files": 0,
                "all_files_paths": []
            }

        # self.show_status(f"Gathering hashes for {len(all_files)} files.")
        # all_hashes = self._get_all_hashes_with_progress(all_files)

        # Sort files by text or file query
        self.show_status(f"Sorting files by: \"{text_query}\"")

        scores = np.zeros(len(all_files), dtype=np.float32)
        if text_query and len(text_query) > 0:
            # use first word as filter name
            filter_name = text_query

            # If there is a file path used as a query, this file exists and within specified formats, sort files by similarity to that file
            if vfs.is_file_exists(text_query) and text_query.lower().endswith(tuple(self.media_formats)):
                if filters["by_file"] is not None:
                    scores = filters["by_file"](all_files, text_query)
            
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

        

        # logger.info(f"Scores calculated for {len(scores)} files.")

        # logger.info(f"Sorting {len(all_files)} files with temperature={temperature}, order={order}...")
        def is_valid_pair(i):
            score = scores[i]
            file_path = all_files[i]
            if score is None or file_path is None:
                return False
            # NaN scores mean the file couldn't be embedded (e.g. empty file with
            # no cache entry under generate_embs_if_not_in_cache=False). They are
            # NOT valid search results and would also break JSON serialization
            # downstream (NaN is non-standard JSON and crashes JSON.parse() in JS).
            # Filter them out exactly like None.
            if isinstance(score, float) and score != score:   # NaN != NaN → True
                return False
            return True

        # Discard unscored files BEFORE sorting, not after. Ranking them and then
        # dropping them wastes the work, and any unscored file left in the input
        # used to drag the scored ones out of order along with it.
        valid = [i for i in range(len(all_files)) if is_valid_pair(i)]
        unindexed_count = len(all_files) - len(valid)

        indices = weighted_shuffle([scores[i] for i in valid], temperature=temperature)

        if order == 'most-relevant':
            pass  # already sorted in descending order
        elif order == 'least-relevant':
            indices = indices[::-1]  # reverse for ascending order
        else:
            raise ValueError("order must be 'most-relevant' or 'least-relevant'")

        sorted_files = [all_files[valid[j]] for j in indices]
        sorted_scores = [scores[valid[j]] for j in indices]

        # Select files for the current page
        page_files = sorted_files[pagination:pagination+limit]
        page_files_scores = sorted_scores[pagination:pagination+limit]
        
        # logger.info(f'page_files {pagination}:{limit}', page_files)

        # self.show_status(f"Gathering hashes for {len(page_files)} files.")
        # page_hashes = self._get_hashes_with_progress(page_files) #[self.engine.get_file_hash(file_path) for file_path in page_files]

        # Extract DB data for the relevant batch of files
        # self.show_status(f"Extracting database info for relevant files.")
        # db_items = self.db_schema.query.filter(self.db_schema.hash.in_(page_hashes)).all()
        # db_items_map = {item.hash: item for item in db_items}

        # # Keep DB path in sync if file was moved/renamed for all files in the page
        # self.show_status(f"Syncing database paths for relevant files.")
        # db_updated = False
        # for ind, full_path in enumerate(page_files):
        #     file_hash = page_hashes[ind]
        #     if file_hash in db_items_map:
        #         db_item = db_items_map[file_hash]
        #         if db_item.file_path != full_path:
        #             db_item.file_path = full_path
        #             db_updated = True

        # if db_updated:
        #     db.session.commit()

        files_data = []
        
        # self.show_status(f"Extracting metadata for {len(page_files)} files.")

        # Parse and open the target filesystem
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(current_path)
        logger.debug(f"Resolved base_url: {base_url}, path_in_fs: {path_in_fs}")

        # Maintain a temporary connection pool to keep execution fast and handle different servers
        opened_fses = {}
        try:
            for ind, full_path in enumerate(page_files):
                self.show_status(f"Extracting metadata for {ind+1}/{len(page_files)} files.")

                file_base_url, path_in_fs = vfs.resolve_base_and_path_from_url(full_path)
                logger.debug(f"Processing file: {full_path}, base_url: {file_base_url}, path_in_fs: {path_in_fs}")

                # Open or reuse the filesystem connection
                if file_base_url not in opened_fses:
                    opened_fses[file_base_url] = fs.open_fs(file_base_url)
                
                my_fs = opened_fses[file_base_url]

                # Get Size and modification timestamp (ns)
                info = my_fs.getinfo(path_in_fs, namespaces=['details'])
                file_size = info.size
                logger.debug(f"File size for {full_path}: {file_size} bytes")

                basename = os.path.basename(path_in_fs)

                # file_hash = page_hashes[ind]

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

                # Clean NaN values from the scores, to avoid serialization issues
                score = page_files_scores[ind]
                if isinstance(score, float) and score != score:
                    score = None
                    
                data = {
                    "type": "file",
                    "full_path": full_path,
                    "file_path": full_path,
                    "base_url": base_url,
                    "path_in_fs": path_in_fs,
                    "base_name": basename,
                    # "hash": file_hash,
                    "file_size": convert_size(file_size),
                    "file_info": get_file_info(full_path),
                    "has_meta": my_fs.exists(path_in_fs + '.meta'),
                    "search_score": score
                }
                files_data.append(data)
        finally:
            # Safely close all opened connection pools
            for opened_fs in opened_fses.values():
                try:
                    opened_fs.close()
                except Exception:
                    pass

        # Save all extracted metadata to the cache
        # self.cached_metadata.save_metadata_cache()

        # elapsed = time.time() - start_time
        # if unindexed_count > 0:
        #     self.show_status(
        #         f'Showed {len(sorted_files)} of {len(all_files)} files in {elapsed:.2f}s. '
        #         f'{unindexed_count} file(s) still unindexed.'
        #     )
        # else:
        #     self.show_status(
        #         f'Processed {len(sorted_files)} files in {elapsed:.2f}s.'
        #     )

        # logger.info(f'get_files returning {len(files_data)} files data.')

        # Check if all the data is serializable and if not print the non-serializable data
        for file_data in files_data:
            try:
                json.dumps(file_data, indent=2)
            except (TypeError, OverflowError) as e:
                raise ValueError(f'Non-serializable data found in file: {file_data["full_path"]}, error: {e}')
            

        # Show final status
        elapsed = time.time() - start_time
        final_summary = (
            f'Showed {len(sorted_files)} of {len(all_files)} files in {elapsed:.2f}s. '
            f'{unindexed_count} file(s) still unindexed.'
            if unindexed_count > 0
            else f'Processed {len(sorted_files)} files in {elapsed:.2f}.'
        )
        self.show_status(final_summary, force=True)   # force=True bypasses throttle

        logger.info(f'get_files returning {len(files_data)} files data.')

        return {
            "files_data": files_data, 
            "folder_path": folder_path, 
            "total_files": len(sorted_files), 
            "all_files_paths": sorted_files
        }

