import os
from pathlib import Path
from modules.text.engine import TextSearch
from src.socket_events import CommonSocketEvents
from src.file_manager import FileManager 
from src.common_filters import CommonFilters
from src.metadata_search import MetadataSearch

import modules.text.db_models as db_models
from src.universal_evaluator import UniversalEvaluator

from src.scheduler import Scheduler
from omegaconf import OmegaConf
import src.db_models as main_db_models
import fs
import src.virtual_file_system as vfs
# from src.module_helpers import make_scheduled_rating_check, make_scheduled_description_check

from src.utils import SortingProgressCallback, EmbeddingGatheringCallback

# -------------------------------------------------------------------------
# MODULE-SPECIFIC HELPER FUNCTIONS
# -------------------------------------------------------------------------

def get_text_preview(file_path):
    base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)

    preview_text = '' # Default no preview
    try:
        with fs.open_fs(base_url) as my_fs:
            # Read as binary and decode manually
            with my_fs.open(path_in_fs, 'rb') as f:
                content_bytes = f.read()
                content = content_bytes.decode('utf-8', errors='ignore')
                preview_text = '\n'.join(content.splitlines()[:25])  # First 25 lines
                if len(preview_text) > 400: # Limit preview length
                    preview_text = preview_text[:400] + '...'
    except Exception as e:
        print(f"[TextModuleServer] Error reading preview from {file_path}: {e}")
        preview_text = 'Error loading preview!'

    return preview_text

class TextModuleServer:
    """
    Class-Based Module Controller for the 'Text' extension.
    """
    
    def __init__(self, app, socketio, cfg, app_root_folder):
        # 1. Store core dependencies
        self.app = app
        self.socketio = socketio
        self.cfg = cfg
        self.app_root_folder = app_root_folder
        self.cse = CommonSocketEvents(socketio, module_name="text")
        self.media_directory = cfg.text.media_directory

        if self.media_directory is None:
            raise ValueError("[TextModuleServer] Text media folder is not set.")

    def initialize(self):
        """Main lifecycle hook to boot up the module."""

        self.cse.show_loading_status('Initializing text search engine...')
        self.text_search_engine = TextSearch(cfg=self.cfg)

        self.cse.show_loading_status('Loading embedding models...')
        self.text_search_engine.initiate(models_folder=self.cfg.main.embedding_models_path, cache_folder=self.cfg.main.cache_path)

        self.cse.show_loading_status('Initializing universal evaluator for text module...')
        self.text_evaluator = UniversalEvaluator()

        self.cse.show_loading_status('Loading universal evaluator model...')
        self.text_evaluator.load(os.path.join(self.cfg.main.personal_models_path, 'universal_evaluator.pt'))

        self.cse.show_loading_status('Setting up file manager...')
        self.file_manager = FileManager(
            app=self.app,
            cfg=self.cfg,
            media_directory=self.media_directory,
            engine=self.text_search_engine,
            module_name="text",
            media_formats=self.cfg.text.media_formats,
            socketio=self.socketio,
            db_schema=main_db_models.FilesLibrary,
        )

        # Create metadata search engine
        self.cse.show_loading_status('Initializing metadata search...')
        self.metadata_search_engine = MetadataSearch(engine=self.text_search_engine)

        # Create common filters instance
        self.cse.show_loading_status('Setting up filters...')
        self.common_filters = CommonFilters(
            engine=self.text_search_engine,
            metadata_engine=self.metadata_search_engine,
            common_socket_events=self.cse,
            media_directory=self.media_directory,
            db_schema=main_db_models.FilesLibrary,
        )

        # _check_and_submit_description = make_scheduled_description_check(
        #     app, 'Text', self.file_manager, metadata_search_engine, cfg, 'text'
        # )

        # Bind all Socket.IO events to their respective class methods
        self.cse.show_loading_status('Registering socket events...')
        self._register_socket_events()
        self._register_schedulers()
        
        self.cse.show_loading_status('Text module ready!')

    def _register_socket_events(self):
        """Maps Socket.IO event strings directly to class methods."""
        # Note: on_event is much cleaner than using nested @socketio.on decorators
        self.socketio.on_event('emit_text_page_get_folders', self.handle_get_folders)
        self.socketio.on_event('emit_text_page_get_files', self.handle_get_files)
        self.socketio.on_event('emit_text_page_get_file_content', self.handle_open_file)
        self.socketio.on_event('emit_text_page_save_file_content', self.handle_save_file)

    def _register_schedulers(self):
        """Registers background tasks for the module."""
       
        app = self.app
        cfg = self.cfg

        rating_update_interval = OmegaConf.select(cfg, 'text.rating_update_interval_minutes', default=None)

        def _check_if_model_rating_needed():
            unrated_files = self.file_manager.get_unrated_files(self.text_evaluator.hash)
            return self.text_evaluator.hash is not None and len(unrated_files) > 0

        Scheduler(app, interval_minutes=rating_update_interval, fn=self.update_model_ratings_schedule,
                name='Text: rate unrated files',
                check_fn=_check_if_model_rating_needed, start_immediately=True)

        # desc_interval = OmegaConf.select(cfg, 'text.description_update_interval_minutes', default=None)
        # Scheduler(app, interval_minutes=desc_interval, fn=_check_and_submit_description,
        #         name='Text: describe undescribed files')
    
        # TODO: Create scheduler that takes data (user ratings needs to be preserved)from the old DB (TextLibrary) and for each rated file create appropriate entry in project_config/memory/ with all available metadata about the file.

    def update_model_ratings_schedule(self):
        # TODO:
        # Instead of updating all the files in the text media directory, we want to build a list of files we encounter (both locally and remotely) and
        # store them in some list with the priority score, and the more time we encounter a file, the more priority it gets to be rated and described. 
        # This way we can focus on the most relevant files first.

        print('[TextModuleServer] update_model_ratings_schedule')

        if self.text_evaluator.hash is None:
            return
        candidates = self.file_manager.get_unrated_files(self.text_evaluator.hash)
        total = len(candidates)
        batch_size = self.cfg.text.rating_update_batch_size
        batch_size = min(batch_size, total) if batch_size else total
        count_str = f"{batch_size} of {total}" if batch_size < total else f"{total}"
        print(f"[TextModuleServer] Rating {count_str} files...")

        def task(ctx):    
            files_list = candidates[:batch_size]
            ctx.update(0.0, f'Rating {len(files_list)} of {total} files...')
            self.update_model_ratings(files_list, ctx=ctx)

        return self.app.task_manager.submit(f'[TextModuleServer] rate unrated files: ({count_str})', task)

    def update_model_ratings(self, files_list, ctx=None):
        print('[TextModuleServer] update_model_ratings')

        _progress = [0.0]

        def _status(msg):
            if ctx is not None:
                ctx.update(_progress[0], msg)
            else:
                self.cse.show_search_status(msg)

        def _check_if_paused():
            if ctx is not None:
                ctx.check()

        _embedding_callback = EmbeddingGatheringCallback(_status)

        # Choose embedding strategy from config (mirrors universal training)
        text_embed_method = OmegaConf.select(self.cfg, "evaluator.text_embedding_method", default="full_text")
        if text_embed_method == "full_text":
            # Full chunked content via TextSearch cache — consistent with universal training
            embeddings = self.text_search_engine.process_files(
                files_list, callback=_embedding_callback, media_folder=self.media_directory
            )
        else:
            # Metadata/summary description path — one description per file
            embeddings = []
            for ind, file_path in enumerate(files_list):
                _check_if_paused()
                _progress[0] = (ind + 1) / len(files_list) * 0.7
                description = self.metadata_search_engine.generate_full_description(
                    file_path,
                    media_folder=self.media_directory,
                    generate_desc_if_not_in_cache=False,
                )
                self.metadata_search_engine.omni_descriptor.unload()
                if description and len(description.strip()) >= 10:
                    chunk_embs = self.metadata_search_engine.text_embedder.embed_text(description)
                    embeddings.append(chunk_embs if chunk_embs is not None else [])
                else:
                    embeddings.append([])

        model_ratings = self.text_evaluator.predict(embeddings)

        # Update the model ratings in the database
        _progress[0] = 0.7
        _status(f"Updating model ratings of files...")
        new_items = []
        update_items = []
        last_shown_time = 0
        for ind, full_path in enumerate(files_list):
            _check_if_paused()
            model_rating = model_ratings[ind].item()

            db_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            if db_item:
                db_item.model_rating = model_rating
                db_item.model_hash = self.text_evaluator.hash
                update_items.append(db_item)
            else:
                file_data = {
                    # "hash": hash,
                    # "hash_algorithm": self.text_search_engine.get_hash_algorithm(),
                    "file_path": full_path,
                    "model_rating": model_rating,
                    "model_hash": self.text_evaluator.hash
                }
                new_items.append(main_db_models.FilesLibrary(**file_data))

            _progress[0] = 0.7 + (ind + 1) / len(files_list) * 0.3
            _status(f"Updated model ratings for {ind+1}/{len(files_list)} files.")

        # Bulk update and insert
        if update_items:
            main_db_models.db.session.bulk_save_objects(update_items)
        if new_items:
            main_db_models.db.session.bulk_save_objects(new_items)

        # Commit the transaction
        db_models.db.session.commit()

    # -------------------------------------------------------------------------
    # SOCKET EVENT HANDLERS
    # -------------------------------------------------------------------------

    def handle_get_folders(self, data):
        """Returns a list of folders in the specified path."""
        path = data.get('path', '')
        return self.file_manager.get_folders(path)

    def handle_get_files(self, data):
        """Scans the requested path and returns module-related files."""
        # Define available filters
        filters = {
            # "by_file": filter_by_file, # special sorting case when file path used as query
            "by_text": self.common_filters.filter_by_text, # special sorting case when text used as query, i.e. all other cases wasn't triggered
            "file_size": self.common_filters.filter_by_file_size,
            # "length": self.common_filters.filter_by_length,
            # "similarity": self.common_filters.filter_by_similarity, 
            # "random": self.common_filters.filter_by_random, 
            "rating": self.common_filters.filter_by_rating,
            # "recommendation": filter_by_recommendation
        }

        # Define a method to gather domain specific file information
        def get_file_info(full_path):
            db_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
                    
            user_rating = None
            model_rating = None
            rating_is_stale = None
            file_data = None

            if db_item:
                user_rating = db_item.user_rating
                model_rating = db_item.model_rating
                rating_is_stale = (
                    db_item.model_rating is not None
                    and self.text_evaluator.hash is not None
                    and db_item.model_hash != self.text_evaluator.hash
                )
                file_data = self.text_search_engine.get_metadata(full_path)  
            else:
                print(f"[TextModule:Serve] File '{full_path}' not found in the database.")

            return {
                    "user_rating": user_rating,
                    "model_rating": model_rating,
                    "rating_is_stale": rating_is_stale,
                    "preview_text": get_text_preview(full_path),    
                    "file_data": file_data,
                }

        input_params = data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
        })
        return self.file_manager.get_files(**input_params)

    def handle_open_file(self, data):
        """Reads the content of a text file and sends it back to the frontend."""

        file_path = data.get('file_path')
        full_path = file_path

        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
        try:
            with fs.open_fs(base_url) as my_fs:
                # Read as binary and decode manually
                with my_fs.open(path_in_fs, 'rb') as f:
                    content_bytes = f.read()
                    content = content_bytes.decode('utf-8', errors='ignore')
            self.socketio.emit('emit_text_page_show_file_content', {"content": content, "file_path": file_path})
        except Exception as e:
            print(f"[TextModuleServer] Error reading file: {full_path}: {e}")
            self.socketio.emit('emit_text_page_show_file_content', {"content": "Error loading file.", "file_path": file_path}) # Error to frontend

    def handle_save_file(self, data):
        """Saves the content of a text file sent from the frontend."""
        # TODO: Check if the file is writable and locally accessible before attempting to write.
        # Otherwise, emit an error event back to the frontend.

        file_path = data.get('file_path')
        text_content = data.get('text_content')
        full_path = file_path

        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
        try:
            with fs.open_fs(base_url) as my_fs:
                # Open as binary 'wb' to write encoded text bytes safely
                with my_fs.open(path_in_fs, 'wb') as f:
                    f.write(text_content.encode('utf-8'))
            print(f"[TextModuleServer] File saved: {full_path}") # Log success
            # Optionally emit success event to frontend
        except Exception as e:
            print(f"[TextModuleServer] Error saving file: {full_path}: {e}") # Log error
            # Optionally emit error event to frontend


# -------------------------------------------------------------------------
# MODULE FACTORY ENTRY POINT
# -------------------------------------------------------------------------
def register_module(app, socketio, cfg, data_folder):
    """
    Standardized entry point called by the ExtensionManager.
    It instantiates the controller and boots it up.
    """
    module_server = TextModuleServer(app, socketio, cfg, data_folder)
    module_server.initialize()
    return module_server