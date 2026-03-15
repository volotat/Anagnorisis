"""
serve.py — Module entry point (REQUIRED)

This file is the heart of every Anagnorisis module. The main application
discovers it automatically and calls `init_socket_events(...)` during startup.

Responsibilities:
  1. Create a CommonSocketEvents instance for progress/status reporting.
  2. Instantiate and initialise the search engine (engine.py).
  3. Optionally load an evaluator model for AI-based scoring.
  4. Set up FileManager for browsing / hashing / paginating files.
  5. Register Flask routes (e.g. serving raw media files to the browser).
  6. Register SocketIO event handlers for every action the frontend needs.

Naming conventions for socket events:
  Incoming  →  emit_{MODULE}_page_{action}   (e.g. emit_example_page_get_files)
  Outgoing  →  emit_{MODULE}_page_{action}   (e.g. emit_example_page_show_files)
  The generic status channel is handled by CommonSocketEvents automatically.
"""

import os
import datetime

from flask import send_from_directory

from omegaconf import OmegaConf

import src.file_manager as file_manager

# Import your module's own submodules
from modules._module_template.engine import ExampleSearch
import modules._module_template.db_models as db_models

# Shared framework utilities
from src.socket_events import CommonSocketEvents
from src.common_filters import CommonFilters
from src.metadata_search import MetadataSearch
from src.utils import convert_size, SortingProgressCallback, EmbeddingGatheringCallback


# ---------------------------------------------------------------------------
# Incoming socket events handled in this module:
#
#   emit_example_page_get_files
#   emit_example_page_get_folders
#   emit_example_page_set_rating
#   emit_example_page_get_path_to_media_folder
#   emit_example_page_update_path_to_media_folder
#
# Outgoing socket events emitted by this module:
#
#   (status events are emitted automatically by CommonSocketEvents)
#   emit_example_page_show_files          — sent as return value from get_files
#   emit_example_page_show_path_to_media_folder
# ---------------------------------------------------------------------------


def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
    """
    Called once by the main application during startup.

    Parameters
    ----------
    socketio : flask_socketio.SocketIO
        The shared SocketIO instance.
    app : flask.Flask
        The Flask application (use to register routes).
    cfg : omegaconf.DictConfig
        The merged application configuration (config.yaml).
    data_folder : str
        Legacy parameter — prefer reading paths from ``cfg`` directly.
    """
    # --- 1. Status helper ------------------------------------------------
    common_socket_events = CommonSocketEvents(socketio, module_name="example")

    # --- 2. Read media directory from config -----------------------------
    common_socket_events.show_loading_status('Checking media directory configuration...')

    # Each module reads its own config section (must match module name in config.yaml)
    media_directory = cfg.get("example", {}).get("media_directory", None)
    if media_directory is None:
        print("Example module: media folder is not set.")

    # --- 3. Initialise search engine -------------------------------------
    common_socket_events.show_loading_status('Initializing search engine...')
    search_engine = ExampleSearch(cfg=cfg)
    search_engine.initiate(
        models_folder=cfg.main.embedding_models_path,
        cache_folder=cfg.main.cache_path,
    )

    # --- 4. (Optional) Load the universal evaluator for AI scoring --------
    # Anagnorisis uses a single universal evaluator for all modules.
    # Load it here if you want model-predicted ratings in your file listings:
    #
    # from modules.train.universal_train import UniversalEvaluator
    # evaluator = UniversalEvaluator()
    # evaluator.load(os.path.join(cfg.main.personal_models_path, 'universal_evaluator.pt'))

    # --- 5. Embedding callback (for progress reporting) ------------------
    embedding_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

    # --- 6. File manager -------------------------------------------------
    common_socket_events.show_loading_status('Setting up file manager...')
    example_file_manager = file_manager.FileManager(
        cfg=cfg,
        media_directory=media_directory,
        engine=search_engine,
        module_name="example",
        media_formats=cfg.get("example", {}).get("media_formats", []),
        socketio=socketio,
        db_schema=db_models.ExampleLibrary,
    )

    # --- 7. Metadata search engine (optional) ----------------------------
    common_socket_events.show_loading_status('Initializing metadata search...')
    metadata_search_engine = MetadataSearch(engine=search_engine)

    # --- 8. Common filters -----------------------------------------------
    common_socket_events.show_loading_status('Setting up filters and routes...')

    def update_model_ratings(files_list):
        """
        Re-compute AI model ratings for files that don't have one yet. 
        Called automatically by CommonFilters when the user sorts by 'rating'.
        Adapt this to your evaluator model.
        """
        pass  # TODO: implement when you add an evaluator model

    common_filters = CommonFilters(
        engine=search_engine,
        metadata_engine=metadata_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.ExampleLibrary,
        update_model_ratings_func=update_model_ratings,
    )

    # --- 9. Flask route to serve raw media files -------------------------
    @app.route('/example_files/<path:filename>')
    def serve_example_files(filename):
        nonlocal media_directory
        return send_from_directory(media_directory, filename)

    # --- 10. SocketIO event handlers -------------------------------------

    @socketio.on('emit_example_page_get_folders')
    def get_folders(data):
        path = data.get('path', '')
        return example_file_manager.get_folders(path)

    @socketio.on('emit_example_page_get_files')
    def get_files(input_data):
        """
        Main handler that the frontend calls to fetch a page of files.
        The ``filters`` dict maps filter names → callables; CommonFilters
        already provides the most common ones.
        """
        filters = {
            "by_file": common_filters.filter_by_file,
            "by_text": common_filters.filter_by_text,
            "file_size": common_filters.filter_by_file_size,
            "similarity": common_filters.filter_by_similarity,
            "random": common_filters.filter_by_random,
            "rating": common_filters.filter_by_rating,
            # Add module-specific filters here, e.g.:
            # "duration": filter_by_duration,
        }

        def get_file_info(full_path, file_hash):
            """
            Return a dict of module-specific metadata for a single file.
            This is sent to the frontend for display in the file grid.
            """
            file_path = os.path.relpath(full_path, media_directory)
            file_size = os.path.getsize(full_path)

            user_rating = None
            model_rating = None

            db_item = db_models.ExampleLibrary.query.filter_by(hash=file_hash).first()
            if db_item:
                user_rating = db_item.user_rating
                model_rating = db_item.model_rating

            return {
                "file_path": file_path,
                "base_name": os.path.basename(full_path),
                "user_rating": user_rating,
                "model_rating": model_rating,
                "file_size": convert_size(file_size),
            }

        input_params = input_data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
            "update_model_ratings": update_model_ratings,
            "evaluator_hash": None,  # Set to evaluator.hash when you have one
        })
        return example_file_manager.get_files(**input_params)

    @socketio.on('emit_example_page_set_rating')
    def set_rating(data):
        """Save a user rating for a file."""
        file_hash = data.get('hash')
        rating = data.get('rating')

        db_item = db_models.ExampleLibrary.query.filter_by(hash=file_hash).first()
        if db_item is None:
            db_item = db_models.ExampleLibrary(
                hash=file_hash,
                hash_algorithm=search_engine.get_hash_algorithm(),
                file_path=data.get('file_path'),
            )
            db_models.db.session.add(db_item)

        db_item.user_rating = rating
        db_item.user_rating_date = datetime.datetime.utcnow()
        db_models.db.session.commit()

    common_socket_events.show_loading_status('Initialization complete')
