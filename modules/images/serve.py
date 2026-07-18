import os
import shutil
import datetime
import traceback

import numpy as np
import send2trash
from omegaconf import OmegaConf

from modules.images.engine import ImageSearch
from src.socket_events import CommonSocketEvents
from src.file_manager import FileManager
from src.common_filters import CommonFilters
from src.metadata_search import MetadataSearch

import modules.images.db_models as db_models
from src.universal_evaluator import UniversalEvaluator
from src.embedding_proxy import EmbeddingProxyGenerator

import src.db_models as main_db_models
import src.virtual_file_system as vfs

from src.utils import convert_size, SortingProgressCallback, EmbeddingGatheringCallback
from src.scheduler import Scheduler
import src.module_helpers

# -------------------------------------------------------------------------
# EVENTS
# -------------------------------------------------------------------------
# Incoming (handled via socketio.on_event):
#   emit_images_page_get_folders
#   emit_images_page_get_files
#   emit_images_page_move_files
#   emit_images_page_open_file_in_folder
#   emit_images_page_send_file_to_trash       (only if ALLOW_FILE_DELETION)
#   emit_images_page_send_files_to_trash      (only if ALLOW_FILE_DELETION)
#   emit_images_page_get_path_to_media_folder
#   emit_images_page_update_path_to_media_folder
#   emit_images_page_get_image_metadata_file_content
#
# Ratings are handled by the UNIFIED emit_set_file_rating event registered in
# src/app_factory/event_manager.py (writes to FilesLibrary + memory).
#
# Outgoing (emitted via socketio.emit):
#   emit_images_page_show_path_to_media_folder
#   emit_images_page_show_image_metadata_content
#   emit_images_page_move_complete
#   emit_images_page_show_error
#   (status events are emitted automatically by CommonSocketEvents)
# -------------------------------------------------------------------------


class ImageModuleServer:
    """
    Class-Based Module Controller for the 'images' extension.

    Architecture mirrors modules/music/serve.py and modules/text/serve.py:
      * User / model ratings live in the shared FilesLibrary table (keyed by
        file_path) so they work with FileManager.get_unrated_files() and
        CommonFilters.filter_by_rating().
      * The legacy ImagesLibrary table is retained only as a one-time migration
        source for old ratings keyed by hash.
      * All file I/O is VFS / remote-server aware.
    """

    def __init__(self, app, socketio, cfg, app_root_folder):
        # 1. Store core dependencies
        self.app = app
        self.socketio = socketio
        self.cfg = cfg
        self.app_root_folder = app_root_folder
        self.cse = CommonSocketEvents(socketio, module_name="images")
        self.media_directory = cfg.images.media_directory

        if self.media_directory is None:
            raise ValueError("[ImageModuleServer] Images media folder is not set.")

    def initialize(self):
        """Main lifecycle hook to boot up the module."""

        self.cse.show_loading_status('Initializing image search engine...')
        self.images_search_engine = ImageSearch(cfg=self.cfg)

        self.cse.show_loading_status('Loading image embedding models...')
        self.images_search_engine.initiate(
            models_folder=self.cfg.main.embedding_models_path,
            cache_folder=self.cfg.main.cache_path,
        )

        self.cse.show_loading_status('Initializing universal evaluator for images module...')
        self.images_evaluator = UniversalEvaluator()

        self.cse.show_loading_status('Loading universal evaluator model...')
        self.images_evaluator.load(
            os.path.join(self.cfg.main.personal_models_path, 'universal_evaluator.pt')
        )

        self.cse.show_loading_status('Setting up file manager...')
        self.file_manager = FileManager(
            app=self.app,
            cfg=self.cfg,
            media_directory=self.media_directory,
            engine=self.images_search_engine,
            module_name="images",
            media_formats=self.cfg.images.media_formats,
            socketio=self.socketio,
            db_schema=main_db_models.FilesLibrary,
        )

        # Create metadata search engine
        self.cse.show_loading_status('Initializing metadata search...')
        self.metadata_search_engine = MetadataSearch(engine=self.images_search_engine)

        # Set up embedding proxy so that files without an OmniDescriptor description
        # still receive a meaningful text representation (SigLIP tags + fingerprint).
        self.cse.show_loading_status('Initializing embedding proxy...')
        _tag_list = list(OmegaConf.select(self.cfg, 'images.embedding_tags', default=[]) or [])
        _tag_threshold_raw = OmegaConf.select(self.cfg, 'images.embedding_tags_threshold', default=0.20)
        _tag_threshold = float(_tag_threshold_raw) if _tag_threshold_raw is not None else None
        self.images_proxy_gen = EmbeddingProxyGenerator(
            engine=self.images_search_engine,
            tag_list=_tag_list,
            threshold=_tag_threshold,
            cache_path=self.cfg.main.cache_path,
            model_name=getattr(self.cfg.image_embedder, 'model_name', 'SigLIP'),
        )
        self.metadata_search_engine.embedding_proxy = self.images_proxy_gen

        # Create common filters instance
        self.cse.show_loading_status('Setting up filters...')
        self.common_filters = CommonFilters(
            engine=self.images_search_engine,
            metadata_engine=self.metadata_search_engine,
            common_socket_events=self.cse,
            media_directory=self.media_directory,
            db_schema=main_db_models.FilesLibrary,
        )

        # Bind all Socket.IO events to their respective class methods
        self.cse.show_loading_status('Registering socket events...')
        self._register_socket_events()
        self._register_schedulers()
        self._register_background_tasks()

        self.cse.show_loading_status('Images module ready!')

    def _register_socket_events(self):
        """Maps Socket.IO event strings directly to class methods."""
        self.socketio.on_event('emit_images_page_get_folders', self.handle_get_folders)
        self.socketio.on_event('emit_images_page_get_files', self.handle_get_files)
        self.socketio.on_event('emit_images_page_move_files', self.handle_move_files)
        self.socketio.on_event('emit_images_page_open_file_in_folder', self.handle_open_file_in_folder)
        self.socketio.on_event('emit_images_page_get_path_to_media_folder', self.handle_get_path_to_media_folder)
        self.socketio.on_event('emit_images_page_update_path_to_media_folder', self.handle_update_path_to_media_folder)
        self.socketio.on_event('emit_images_page_get_image_metadata_file_content', self.handle_get_image_metadata)

        # File deletion handlers (only if explicitly allowed)
        if os.environ.get('ALLOW_FILE_DELETION', 'false').lower() == 'true':
            self.socketio.on_event('emit_images_page_send_file_to_trash', self.handle_send_file_to_trash)
            self.socketio.on_event('emit_images_page_send_files_to_trash', self.handle_send_files_to_trash)
        else:
            print("Images module: File deletion handlers disabled (ALLOW_FILE_DELETION=false)")

        # .meta sidecar handlers + full description handler (shared helper)
        src.module_helpers.register_meta_handlers(
            self.socketio, 'images', lambda: self.media_directory, self.metadata_search_engine
        )

    def _register_schedulers(self):
        """Registers background schedulers for the module."""
        app = self.app
        cfg = self.cfg

        # Proactively compute embeddings for any files not yet in the cache.
        _check_and_submit_embedding = src.module_helpers.make_scheduled_embedding_check(
            app, 'Images', self.file_manager, self.images_search_engine, cfg, 'images'
        )
        embedding_update_interval = OmegaConf.select(
            cfg, 'images.embedding_update_interval_minutes', default=10
        )
        Scheduler(
            app,
            interval_minutes=embedding_update_interval,
            fn=_check_and_submit_embedding,
            name='Images: compute missing embeddings',
        )

        # Proactively rate unrated files using the shared factory (writes to FilesLibrary)
        _check_and_submit_rating = src.module_helpers.make_scheduled_rating_check(
            app, 'Images', self.file_manager, self.images_evaluator, cfg, 'images',
            self.update_model_ratings,
        )
        rating_update_interval = OmegaConf.select(cfg, 'images.rating_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=rating_update_interval,
            fn=_check_and_submit_rating,
            name='Images: rate unrated files',
            check_fn=lambda: self.images_evaluator.hash is not None
            and len(self.file_manager.get_unrated_files(self.images_evaluator.hash)) > 0,
        )

        # Proactively describe undescribed files using the shared factory
        _check_and_submit_description = src.module_helpers.make_scheduled_description_check(
            app, 'Images', self.file_manager, self.metadata_search_engine, cfg, 'images'
        )
        desc_interval = OmegaConf.select(cfg, 'images.description_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=desc_interval,
            fn=_check_and_submit_description,
            name='Images: describe undescribed files',
        )



    def _register_background_tasks(self):
        """One-time migration task.

        Previously user/model ratings for images were stored in ImagesLibrary
        (keyed by hash). They now belong in the shared FilesLibrary table (keyed
        by file_path) so they are consistent with the rest of the framework.
        This task normalizes old relative file_paths to VFS URLs and then copies
        any rating that exists in ImagesLibrary but is missing (or unrated) in
        FilesLibrary. It is a no-op once everything has been migrated.
        """

        def _normalize_path(raw_path):
            if not raw_path:
                return raw_path
            if '://' in raw_path:
                return raw_path
            return vfs.join_fs_url(self.media_directory, raw_path)

        def _normalize_old_paths(ctx):
            try:
                candidates = db_models.ImagesLibrary.query.filter(
                    db_models.ImagesLibrary.file_path.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[ImageModuleServer] DB query failed: {exc}")
                return 0

            to_fix = [row for row in candidates if '://' not in (row.file_path or '')]
            total = len(to_fix)
            if total == 0:
                return 0

            print(f"[ImageModuleServer] Normalizing {total} legacy file_path values to VFS URLs.")
            for i, row in enumerate(to_fix):
                ctx.check()
                ctx.update((i + 1) / total, f'Normalizing path {i + 1}/{total}')
                row.file_path = _normalize_path(row.file_path)

            db_models.db.session.commit()
            print(f"[ImageModuleServer] Normalized {total} paths.")
            return total

        def _check_if_migration_needed():
            try:
                rel_count = db_models.ImagesLibrary.query.filter(
                    db_models.ImagesLibrary.file_path.isnot(None),
                    ~db_models.ImagesLibrary.file_path.like('%://%')
                ).count()
            except Exception as exc:
                print(f"[ImageModuleServer] DB query failed: {exc}")
                rel_count = 0
            if rel_count > 0:
                return True

            try:
                old_entries = db_models.ImagesLibrary.query.filter(
                    db_models.ImagesLibrary.user_rating.isnot(None)
                    | db_models.ImagesLibrary.model_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[ImageModuleServer] DB query failed: {exc}")
                return False

            for old_entry in old_entries:
                if not old_entry.file_path:
                    continue
                new_entry = main_db_models.FilesLibrary.query.filter_by(
                    file_path=old_entry.file_path
                ).first()
                needs_user = old_entry.user_rating is not None and (
                    not new_entry or new_entry.user_rating is None
                )
                needs_model = old_entry.model_rating is not None and (
                    not new_entry or new_entry.model_rating is None
                )
                if needs_user or needs_model:
                    return True
            return False

        def _copy_ratings_from_old_table(ctx):
            # Pass 1: normalize ImagesLibrary paths first (so lookups below match).
            _normalize_old_paths(ctx)

            # Pass 2: copy ratings into FilesLibrary, keyed by the normalized path.
            try:
                old_entries = db_models.ImagesLibrary.query.filter(
                    db_models.ImagesLibrary.user_rating.isnot(None)
                    | db_models.ImagesLibrary.model_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[ImageModuleServer] DB query failed: {exc}")
                return

            total = len(old_entries)
            if total == 0:
                print("[ImageModuleServer] No rated images found in old table.")
                return

            print(f"[ImageModuleServer] {total} rated images found in old table.")
            for i, old_entry in enumerate(old_entries):
                ctx.check()
                ctx.update((i + 1) / total, f'Copying rating for {i + 1}/{total}: {old_entry.file_path}')

                if not old_entry.file_path:
                    continue

                new_entry = main_db_models.FilesLibrary.query.filter_by(
                    file_path=old_entry.file_path
                ).first()
                if new_entry:
                    if new_entry.user_rating is None:
                        new_entry.user_rating = old_entry.user_rating
                        new_entry.user_rating_date = old_entry.user_rating_date
                    if new_entry.model_rating is None:
                        new_entry.model_rating = old_entry.model_rating
                        new_entry.model_hash = old_entry.model_hash
                else:
                    new_entry = main_db_models.FilesLibrary(
                        file_path=old_entry.file_path,
                        hash=old_entry.hash,
                        hash_algorithm=old_entry.hash_algorithm,
                        user_rating=old_entry.user_rating,
                        user_rating_date=old_entry.user_rating_date,
                        model_rating=old_entry.model_rating,
                        model_hash=old_entry.model_hash,
                    )
                    main_db_models.db.session.add(new_entry)

            main_db_models.db.session.commit()
            print("[ImageModuleServer] Ratings copied successfully.")

        if _check_if_migration_needed():
            self.app.task_manager.submit(
                'ImageModuleServer: Copy ratings from old table', _copy_ratings_from_old_table
            )

    # ------------------------------------------------------------------
    # MODEL RATING PIPELINE
    # ------------------------------------------------------------------

    def update_model_ratings(self, files_list, ctx=None):
        """
        Re-compute AI model ratings for the given files and persist them into
        FilesLibrary (keyed by file_path).

        Images-specific pipeline (kept from the legacy module):
          1. Compute SigLIP embeddings (fast — disk-cached).
          2. Pre-populate the embedding-proxy cache (SigLIP tags + fingerprint).
          3. Build a text description (incl. proxy section) + Jina embedding.
          4. Predict with the universal evaluator and persist.
        """
        print('[ImageModuleServer] update_model_ratings')

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

        if self.images_evaluator.hash is None:
            print('[ImageModuleServer] Universal evaluator not trained yet. Skipping model rating update.')
            return

        files_list_hash_map = {}
        for ind, file_path in enumerate(files_list):
            _check_if_paused()
            _progress[0] = (ind + 1) / len(files_list) * 0.1
            _status(f"Computing files hashes {ind + 1}/{len(files_list)}")
            file_hash = self.images_search_engine.get_file_hash(file_path)
            files_list_hash_map[file_path] = file_hash

        # Step 1: Compute SigLIP embeddings (fast — results are disk-cached)
        _progress[0] = 0.1
        _status(f"Computing SigLIP embeddings for {len(files_list)} files...")
        embeddings = self.images_search_engine.process_images(
            files_list, callback=_embedding_callback, media_folder=self.media_directory
        )  # [N, D] tensor

        # Step 2: Pre-populate proxy cache before the Jina embedding phase.
        _progress[0] = 0.3
        _status("Preparing embedding proxies...")
        for i, fp in enumerate(files_list):
            _check_if_paused()
            try:
                self.images_proxy_gen.compute_proxy_section(
                    files_list_hash_map[fp], embeddings[i].cpu().numpy()
                )
            except Exception as e:
                print(f"[ImageModuleServer] Proxy generation failed for {fp}: {e}")
            _progress[0] = 0.3 + (i + 1) / len(files_list) * 0.1

        # Step 3: Build text descriptions (includes proxy section) + Jina embed
        _progress[0] = 0.4
        _status(f"Computing metadata embeddings for {len(files_list)} files...")
        all_embeddings = []
        embedding_dim = self.metadata_search_engine.text_embedder.embedding_dim or 1024
        for ind, full_path in enumerate(files_list):
            _check_if_paused()
            _progress[0] = 0.4 + (ind + 1) / len(files_list) * 0.3
            _status(f"Computing metadata embeddings {ind + 1}/{len(files_list)}")
            try:
                description = self.metadata_search_engine.generate_full_description(
                    full_path,
                    media_folder=self.media_directory,
                    generate_desc_if_not_in_cache=False,
                )
                chunk_embeddings = self.metadata_search_engine.text_embedder.embed_text(description)
                if chunk_embeddings is not None and len(chunk_embeddings) > 0:
                    all_embeddings.append(np.array(chunk_embeddings, dtype=np.float32))
                else:
                    all_embeddings.append(np.zeros((1, embedding_dim), dtype=np.float32))
            except Exception as e:
                print(f"[ImageModuleServer] Embedding failed for {full_path}: {e}")
                all_embeddings.append(np.zeros((1, embedding_dim), dtype=np.float32))

        # Step 4: Predict with the universal evaluator and persist into FilesLibrary
        model_ratings = self.images_evaluator.predict(all_embeddings)

        _progress[0] = 0.7
        _status("Updating model ratings of images...")
        new_items = []
        update_items = []
        for ind, full_path in enumerate(files_list):
            _check_if_paused()
            model_rating = float(model_ratings[ind])

            db_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            if db_item:
                db_item.model_rating = model_rating
                db_item.model_hash = self.images_evaluator.hash
                update_items.append(db_item)
            else:
                file_data = {
                    "hash": files_list_hash_map[full_path],
                    "hash_algorithm": self.images_search_engine.get_hash_algorithm(),
                    "file_path": full_path,
                    "model_rating": model_rating,
                    "model_hash": self.images_evaluator.hash,
                }
                new_items.append(main_db_models.FilesLibrary(**file_data))

            _progress[0] = 0.7 + (ind + 1) / len(files_list) * 0.3
            _status(f"Updated model ratings for {ind + 1}/{len(files_list)} images.")

        if update_items:
            main_db_models.db.session.bulk_save_objects(update_items)
        if new_items:
            main_db_models.db.session.bulk_save_objects(new_items)
        main_db_models.db.session.commit()

    # -------------------------------------------------------------------------
    # SOCKET EVENT HANDLERS
    # -------------------------------------------------------------------------

    def handle_get_folders(self, data):
        """Returns a list of folders in the specified path."""
        path = data.get('path', '')
        return self.file_manager.get_folders(path)

    def handle_get_files(self, data):
        """Scans the requested path and returns module-related files."""
        # Images-specific filters
        def filter_by_resolution(all_files, text_query):
            self.cse.show_search_status("Gathering resolutions for sorting...")
            progress_callback = SortingProgressCallback(
                self.cse.show_search_status, operation_name="Gathering images resolution "
            )
            scores = []
            for ind, full_path in enumerate(all_files):
                metadata = self.images_search_engine.get_metadata(full_path)
                res = metadata.get('resolution')
                if res and 'x' in str(res):
                    try:
                        w, h = str(res).split('x')
                        scores.append(float(w) * float(h))
                    except (ValueError, TypeError):
                        scores.append(0.0)
                else:
                    scores.append(0.0)
                progress_callback(ind + 1, len(all_files))
            return scores

        def filter_by_proportion(all_files, text_query):
            self.cse.show_search_status("Gathering proportions for sorting...")
            progress_callback = SortingProgressCallback(
                self.cse.show_search_status, operation_name="Gathering images proportion "
            )
            scores = []
            for ind, full_path in enumerate(all_files):
                metadata = self.images_search_engine.get_metadata(full_path)
                res = metadata.get('resolution')
                if res and 'x' in str(res):
                    try:
                        w, h = str(res).split('x')
                        scores.append(float(w) / float(h) if float(h) != 0 else 0.0)
                    except (ValueError, TypeError, ZeroDivisionError):
                        scores.append(0.0)
                else:
                    scores.append(0.0)
                progress_callback(ind + 1, len(all_files))
            return scores

        # Define available filters
        filters = {
            "by_file": self.common_filters.filter_by_file,
            "by_text": self.common_filters.filter_by_text,
            "file_size": self.common_filters.filter_by_file_size,
            "resolution": filter_by_resolution,
            "proportion": filter_by_proportion,
            "similarity": self.common_filters.filter_by_similarity,
            "random": self.common_filters.filter_by_random,
            "rating": self.common_filters.filter_by_rating,
        }

        # Gather domain-specific file information (single-arg, per FileManager contract)
        def get_file_info(full_path):
            files_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            user_rating = files_item.user_rating if files_item else None
            model_rating = files_item.model_rating if files_item else None
            rating_is_stale = (
                model_rating is not None
                and self.images_evaluator.hash is not None
                and files_item is not None
                and files_item.model_hash != self.images_evaluator.hash
            )

            file_data = self.images_search_engine.get_metadata(full_path)
            resolution = file_data.get('resolution') if file_data else None

            return {
                "user_rating": user_rating,
                "model_rating": model_rating,
                "rating_is_stale": rating_is_stale,
                "resolution": resolution or "N/A",
                "file_data": file_data,
            }

        input_params = data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
        })
        return self.file_manager.get_files(**input_params)

    def handle_move_files(self, data):
        """Move selected files to a target folder (VFS-aware).

        Handles conflict resolution via numeric suffixes, moves associated
        .meta sidecars, and reports progress/errors back to the client.
        """
        try:
            files = data['files']
            target_folder = data['target_folder']

            if not files:
                self.socketio.emit('emit_images_page_show_error', {'message': 'No files selected to move.'})
                return

            import fs as _fs
            target_base_url, target_path_in_fs = vfs.resolve_base_and_path_from_url(target_folder)

            moved_count = 0
            errors = []

            for idx, file_path in enumerate(files):
                try:
                    self.cse.show_search_status(f"Moving file {idx + 1}/{len(files)}: {os.path.basename(file_path)}")

                    src_base_url, src_path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
                    base_name = os.path.basename(file_path)
                    dest_path_in_fs = _fs.path.join(target_path_in_fs, base_name)

                    # Open both source and target filesystems
                    with _fs.open_fs(src_base_url) as src_fs, _fs.open_fs(target_base_url) as dest_fs:
                        if not dest_fs.isdir(target_path_in_fs):
                            dest_fs.makedirs(target_path_in_fs, recreate=True)

                        # Conflict resolution: append numeric suffix
                        if dest_fs.exists(dest_path_in_fs):
                            name, ext = os.path.splitext(base_name)
                            counter = 1
                            while dest_fs.exists(_fs.path.join(target_path_in_fs, f"{name}_{counter}{ext}")):
                                counter += 1
                            dest_path_in_fs = _fs.path.join(target_path_in_fs, f"{name}_{counter}{ext}")

                        # Move the file
                        _fs.copy.copy_file(src_fs, src_path_in_fs, dest_fs, dest_path_in_fs)
                        src_fs.remove(src_path_in_fs)
                        moved_count += 1

                        # Move associated .meta file if it exists
                        meta_src = src_path_in_fs + '.meta'
                        if src_fs.exists(meta_src):
                            try:
                                meta_dest = dest_path_in_fs + '.meta'
                                _fs.copy.copy_file(src_fs, meta_src, dest_fs, meta_dest)
                                src_fs.remove(meta_src)
                            except Exception as meta_error:
                                print(f"Warning: Failed to move .meta file for {file_path}: {meta_error}")

                    # Update database entry if exists
                    new_full_path = vfs.join_fs_url(target_base_url, dest_path_in_fs)
                    db_item = main_db_models.FilesLibrary.query.filter_by(file_path=file_path).first()
                    if db_item:
                        db_item.file_path = new_full_path
                        main_db_models.db.session.commit()

                except Exception as e:
                    errors.append(f"{os.path.basename(file_path)}: {str(e)}")
                    print(f"Error moving {file_path}: {e}")

            result_message = f"Successfully moved {moved_count} file(s)"
            if errors:
                result_message += f". {len(errors)} error(s) occurred."

            self.socketio.emit('emit_images_page_move_complete', {
                'success': True,
                'moved_count': moved_count,
                'total_count': len(files),
                'errors': errors,
                'message': result_message
            })
            self.cse.show_search_status(result_message)

        except Exception as e:
            print(f"Critical error in move_files: {e}")
            traceback.print_exc()
            self.socketio.emit('emit_images_page_show_error', {
                'message': f"Critical error while moving files: {str(e)}"
            })

    def handle_open_file_in_folder(self, file_path):
        import src.file_manager as file_manager
        file_manager.open_file_in_folder(file_path)

    def handle_send_file_to_trash(self, file_path):
        if os.path.isfile(file_path):
            send2trash.send2trash(file_path)
            print(f"File '{file_path}' sent to trash.")
        else:
            print(f"Error: File '{file_path}' does not exist.")

    def handle_send_files_to_trash(self, files):
        for file_path in files:
            self.handle_send_file_to_trash(file_path)

    def handle_get_path_to_media_folder(self, data=None):
        self.socketio.emit('emit_images_page_show_path_to_media_folder', self.cfg.images.media_directory)

    def handle_update_path_to_media_folder(self, new_path):
        self.cfg.images.media_directory = new_path
        self.media_directory = new_path

        config_path = os.path.join(self.app_root_folder, 'Anagnorisis-app', 'config.yaml')
        try:
            with open(config_path, 'w') as file:
                OmegaConf.save(self.cfg, file)
        except Exception as e:
            print(f"[ImageModuleServer] Failed to persist config update: {e}")

        self.socketio.emit('emit_images_page_show_path_to_media_folder', self.cfg.images.media_directory)

    def handle_get_image_metadata(self, file_path):
        """Reads the content of the .meta file associated with an image (VFS-aware)."""
        content = ""
        try:
            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path + '.meta')
            import fs as _fs
            with _fs.open_fs(base_url) as my_fs:
                if my_fs.exists(path_in_fs):
                    with my_fs.open(path_in_fs, 'rb') as f:
                        content = f.read().decode('utf-8', errors='ignore')
        except Exception as e:
            print(f"Error reading metadata for {file_path}: {e}")

        self.socketio.emit('emit_images_page_show_image_metadata_content', {
            "content": content, "file_path": file_path
        })


# -------------------------------------------------------------------------
# MODULE FACTORY ENTRY POINT
# -------------------------------------------------------------------------
def register_module(app, socketio, cfg, data_folder):
    """
    Standardized entry point called by the ExtensionManager.
    It instantiates the controller and boots it up.
    """
    module_server = ImageModuleServer(app, socketio, cfg, data_folder)
    module_server.initialize()
    return module_server
