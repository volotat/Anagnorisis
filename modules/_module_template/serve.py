"""
serve.py — Module entry point (REQUIRED)

This file is the heart of every Anagnorisis module. The main application
discovers it automatically and calls `init_socket_events(...)` during startup.

Responsibilities:
  1. Create a CommonSocketEvents instance for progress/status reporting.
  2. Scope the shared content-search engine to this module's media type.
  3. Load the universal evaluator for AI-based scoring.
  4. Set up FileManager for browsing / hashing / paginating files.
  5. Set up metadata search and CommonFilters.
  6. Register Flask routes (e.g. serving raw media files to the browser).
  7. Register SocketIO event handlers for every action the frontend needs.
  8. Define and schedule background tasks (rating, description generation).

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
from anagnorisis_core.search.content_search import get_content_search
import modules._module_template.db_models as db_models

# Shared framework utilities
from src.socket_events import CommonSocketEvents
from anagnorisis_core.search.common_filters import CommonFilters
from anagnorisis_core.search.metadata_search import get_metadata_search
from src.scheduler import Scheduler
from src.utils import convert_size, SortingProgressCallback, EmbeddingGatheringCallback


# ---------------------------------------------------------------------------
# Incoming socket events handled in this module:
#
#   emit_example_page_get_files
#   emit_example_page_get_folders
#   emit_example_page_set_rating
#   emit_example_page_get_path_to_media_folder
#   emit_example_page_update_path_to_media_folder
#   emit_example_page_get_external_metadata_file_content
#   emit_example_page_save_external_metadata_file_content
#   emit_example_page_get_full_metadata_description
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
    # There is no per-module engine any more. One shared multimodal embedder
    # backs every media type; get_content_search scopes it to the media type
    # this module declares in config.defaults.yaml.
    common_socket_events.show_loading_status('Initializing search engine...')
    search_engine = get_content_search(cfg, 'images')
    search_engine.initiate(
        models_folder=cfg.main.embedding_models_path,
        cache_folder=cfg.main.cache_path,
    )

    # --- 4. (Optional) Load the universal evaluator for AI scoring --------
    # Anagnorisis uses a single universal evaluator for all modules.
    # Uncomment these lines to enable model-predicted ratings in file listings:
    #
    from anagnorisis_core.ratings.training import UniversalEvaluator
    evaluator = UniversalEvaluator()
    evaluator.load(os.path.join(cfg.main.personal_models_path, 'universal_evaluator.pt'))

    # --- 5. Embedding callback (for progress reporting) ------------------
    embedding_callback = EmbeddingGatheringCallback(common_socket_events.show_search_status)

    # --- 6. File manager -------------------------------------------------
    common_socket_events.show_loading_status('Setting up file manager...')
    example_file_manager = file_manager.FileManager(
        app=app,
        cfg=cfg,
        media_directory=media_directory,
        engine=search_engine,
        module_name="example",
        media_formats=cfg.get("example", {}).get("media_formats", []),
        socketio=socketio,
        db_schema=db_models.ExampleLibrary,
    )

    # --- 7. Metadata search ----------------------------------------------
    # One process-wide, module-independent instance. It resolves each file's
    # media type from its extension, so this module needs no metadata config of
    # its own — declare `media_types:` in config.defaults.yaml and the file
    # extensions, tag vocabulary, internal-metadata reader and background
    # description/embedding passes all follow from media_types/.
    common_socket_events.show_loading_status('Initializing metadata search...')
    metadata_search_engine = get_metadata_search(cfg)

    # --- 8. Common filters -----------------------------------------------
    common_socket_events.show_loading_status('Setting up filters and routes...')

    def update_model_ratings(files_list):
        """
        Re-compute AI model ratings for files that don't have one yet. 
        Called automatically by CommonFilters when the user sorts by 'rating'
        and by the scheduled background rating task.

        This implementation follows the same pattern used by images/music/videos/text modules:
          1. For each file, check if a DB entry exists with a current model_hash.
          2. If not, generate a text description via metadata search.
          3. Embed the description and predict a rating with the evaluator.
          4. Save to DB (create or update the row).
        """
        if not evaluator.is_loaded():
            return

        with app.app_context():
            update_items = []
            new_items = []

            for file_path, file_hash in files_list:
                try:
                    description = metadata_search_engine.generate_full_description(
                        file_path, generate_desc_if_not_in_cache=False,
                    )
                    if not description:
                        continue

                    embedding = metadata_search_engine.embedder.embed_text(description)
                    if embedding is None or len(embedding) == 0:
                        continue

                    predicted_score = evaluator.predict(embedding)

                    db_item = db_models.ExampleLibrary.query.filter_by(hash=file_hash).first()
                    if db_item is None:
                        db_item = db_models.ExampleLibrary(
                            hash=file_hash,
                            hash_algorithm=search_engine.get_hash_algorithm(),
                            file_path=file_path,
                        )
                        new_items.append(db_item)
                    else:
                        update_items.append(db_item)

                    db_item.model_rating = predicted_score
                    db_item.model_hash = evaluator.hash
                except Exception as e:
                    print(f"[Example] Error rating {file_path}: {e}")

            # Deduplicate new items by hash before bulk insert
            seen = set()
            deduped = []
            for item in new_items:
                if item.hash not in seen:
                    seen.add(item.hash)
                    deduped.append(item)

            if update_items:
                db_models.db.session.bulk_save_objects(update_items)
            if deduped:
                db_models.db.session.bulk_save_objects(deduped)
            db_models.db.session.commit()

    # --- 8b. Scheduled background tasks -----------------------------------

    def _check_and_submit_rating():
        """Scheduled: find unrated files on disk and submit a rating task if needed."""
        candidates = example_file_manager.get_unrated_files(evaluator.hash)
        total = len(candidates)
        base_name = 'Example: rate unrated files'
        batch_size = OmegaConf.select(cfg, 'example.rating_update_batch_size', default=None)
        batch_size = min(batch_size, total) if batch_size else total
        label = f"{batch_size} of {total}" if batch_size < total else f"{total}"

        def task(ctx):
            files_list = candidates[:batch_size]
            ctx.update(0.0, f'Rating {len(files_list)} of {total} files...')
            update_model_ratings(files_list)

        return app.task_manager.submit(f'{base_name} ({label})', task)

    common_filters = CommonFilters(
        engine=search_engine,
        metadata_engine=metadata_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.ExampleLibrary,
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
            file_path = full_path
            file_size = os.path.getsize(full_path)

            user_rating = None
            model_rating = None

            db_item = db_models.ExampleLibrary.query.filter_by(hash=file_hash).first()
            if db_item:
                user_rating = db_item.user_rating
                model_rating = db_item.model_rating

            rating_is_stale = (
                model_rating is not None
                and db_item is not None
                and evaluator.is_loaded()
                and db_item.model_hash != evaluator.hash
            )
            return {
                "file_path": file_path,
                "base_name": os.path.basename(full_path),
                "user_rating": user_rating,
                "model_rating": model_rating,
                "rating_is_stale": rating_is_stale,
                "file_size": convert_size(file_size),
            }

        input_params = input_data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
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

    # --- .meta sidecar file handlers -------------------------------------

    @socketio.on('emit_example_page_get_external_metadata_file_content')
    def get_external_metadata_file_content(file_path):
        """Read the content of the .meta sidecar file for a given file."""
        nonlocal media_directory
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"
        content = ""
        try:
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
        except Exception as e:
            print(f"Error reading external metadata for {file_path}: {e}")
        return {"content": content, "file_path": file_path}

    @socketio.on('emit_example_page_save_external_metadata_file_content')
    def save_external_metadata_file_content(data):
        """Save content to the .meta sidecar file."""
        nonlocal media_directory
        file_path = data['file_path']
        metadata_content = data['metadata_content']
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"
        try:
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
        except Exception as e:
            print(f"Error saving metadata for {file_path}: {e}")

    @socketio.on('emit_example_page_get_full_metadata_description')
    def get_full_metadata_description(file_path):
        """Generate a full AI-powered metadata description for a single file."""
        nonlocal media_directory
        full_path = os.path.join(media_directory, file_path)
        content = metadata_search_engine.generate_full_description(full_path, generate_desc_if_not_in_cache=False)
        return {"content": content, "file_path": file_path}

    # --- Finish initialisation & schedule background tasks ----------------

    common_socket_events.show_loading_status('Example module ready!')

    rating_interval = OmegaConf.select(cfg, 'example.rating_update_interval_minutes', default=None)
    Scheduler(app, interval_minutes=rating_interval, fn=_check_and_submit_rating,
              name='Example: rate unrated files',
              check_fn=lambda: evaluator.hash is not None and len(example_file_manager.get_unrated_files(evaluator.hash)) > 0)

    # Descriptions and metadata embeddings need no scheduler here: the app-wide
    # MetadataIndexer (src/metadata_indexer.py) covers every media type on every
    # configured server in a single pass.
