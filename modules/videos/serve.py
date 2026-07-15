import os
import sys
import glob
import json
import subprocess
import shutil
import tempfile
import uuid
import threading
import time
import datetime
import traceback

import numpy as np
import send2trash
from omegaconf import OmegaConf
from flask import send_from_directory

from modules.videos.engine import VideoSearch
from src.socket_events import CommonSocketEvents
from src.file_manager import FileManager
from src.common_filters import CommonFilters
from src.metadata_search import MetadataSearch

import modules.videos.db_models as db_models
from src.universal_evaluator import UniversalEvaluator
from src.recommendation_engine import sort_files_by_recommendation

import src.db_models as main_db_models
import src.virtual_file_system as vfs

from src.utils import convert_size, time_difference, EmbeddingGatheringCallback
from src.scheduler import Scheduler
from src.module_helpers import (
    register_meta_handlers,
    make_scheduled_rating_check,
    make_scheduled_description_check,
)


# -------------------------------------------------------------------------
# Module-level helpers
# -------------------------------------------------------------------------

def generate_preview(video_path, preview_path):
    """Generate a preview image (middle frame) for a video file using moviepy.

    Both arguments must be real local filesystem paths (use
    ``vfs.resolve_to_local_path`` to convert VFS URLs before calling).
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        middle_time = clip.duration / 2
        clip.save_frame(preview_path, t=middle_time)
    except Exception as e:
        print(f"Error generating preview for {video_path}: {e}")


# -------------------------------------------------------------------------
# EVENTS
# -------------------------------------------------------------------------
# Incoming (handled via socketio.on_event):
#   emit_videos_page_get_folders
#   emit_videos_page_get_files
#   emit_videos_page_open_file_in_folder
#   emit_videos_page_send_file_to_trash      (only if ALLOW_FILE_DELETION)
#   emit_videos_page_video_start_playing
#   emit_videos_page_start_streaming
#   emit_videos_page_stop_streaming
#
# Ratings are handled by the UNIFIED emit_set_file_rating event registered in
# src/app_factory/event_manager.py (writes to FilesLibrary + memory).
#
# Outgoing (emitted via socketio.emit):
#   (status events are emitted automatically by CommonSocketEvents)
# -------------------------------------------------------------------------


class VideoModuleServer:
    """
    Class-Based Module Controller for the 'videos' extension.

    Architecture mirrors modules/music/serve.py and modules/images/serve.py:
      * User / model ratings live in the shared FilesLibrary table (keyed by
        file_path) so they work with FileManager.get_unrated_files() and
        CommonFilters.filter_by_rating().
      * Play-tracking data (full_play_count / skip_count / last_played) lives
        in the module-specific VideosLibrary table, joined by file_path.
      * The HLS streaming subsystem (FFmpeg transcoding + Flask routes) is
        preserved as instance state and methods.
      * All file I/O is VFS / remote-server aware.
    """

    def __init__(self, app, socketio, cfg, app_root_folder):
        self.app = app
        self.socketio = socketio
        self.cfg = cfg
        self.app_root_folder = app_root_folder
        self.cse = CommonSocketEvents(socketio, module_name="videos")
        self.media_directory = cfg.videos.media_directory

        if self.media_directory is None:
            raise ValueError("[VideoModuleServer] Videos media folder is not set.")

        # HLS streaming state
        self.active_transcodings = {}

    def initialize(self):
        """Main lifecycle hook to boot up the module."""

        self.cse.show_loading_status('Initializing video search engine...')
        self.videos_search_engine = VideoSearch(cfg=self.cfg)

        self.cse.show_loading_status('Loading video embedding models...')
        self.videos_search_engine.initiate(
            models_folder=self.cfg.main.embedding_models_path,
            cache_folder=self.cfg.main.cache_path,
        )

        self.cse.show_loading_status('Initializing universal evaluator for videos module...')
        self.videos_evaluator = UniversalEvaluator()

        self.cse.show_loading_status('Loading universal evaluator model...')
        self.videos_evaluator.load(
            os.path.join(self.cfg.main.personal_models_path, 'universal_evaluator.pt')
        )

        self.cse.show_loading_status('Setting up file manager...')
        self.file_manager = FileManager(
            app=self.app,
            cfg=self.cfg,
            media_directory=self.media_directory,
            engine=self.videos_search_engine,
            module_name="videos",
            media_formats=self.cfg.videos.media_formats,
            socketio=self.socketio,
            db_schema=main_db_models.FilesLibrary,
        )

        # Create metadata search engine (no EmbeddingProxyGenerator — VideoSearch is a stub)
        self.cse.show_loading_status('Initializing metadata search...')
        self.metadata_search_engine = MetadataSearch(engine=self.videos_search_engine)

        # Create common filters instance
        self.cse.show_loading_status('Setting up filters...')
        self.common_filters = CommonFilters(
            engine=self.videos_search_engine,
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

        # Start the streaming cleanup daemon thread
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True,
                                          name="video-stream-cleanup")
        cleanup_thread.start()

        self.cse.show_loading_status('Videos module ready!')

    def _register_socket_events(self):
        """Maps Socket.IO event strings directly to class methods."""
        self.socketio.on_event('emit_videos_page_get_folders', self.handle_get_folders)
        self.socketio.on_event('emit_videos_page_get_files', self.handle_get_files)
        self.socketio.on_event('emit_videos_page_open_file_in_folder', self.handle_open_file_in_folder)
        self.socketio.on_event('emit_videos_page_video_start_playing', self.handle_video_start_playing)
        self.socketio.on_event('emit_videos_page_start_streaming', self.handle_start_streaming)
        self.socketio.on_event('emit_videos_page_stop_streaming', self.handle_stop_streaming)

        # File deletion handlers (only if explicitly allowed)
        if os.environ.get('ALLOW_FILE_DELETION', 'false').lower() == 'true':
            self.socketio.on_event('emit_videos_page_send_file_to_trash', self.handle_send_file_to_trash)
        else:
            print("Videos module: File deletion handlers disabled (ALLOW_FILE_DELETION=false)")

        # .meta sidecar handlers + full description handler (shared helper)
        register_meta_handlers(
            self.socketio, 'videos', lambda: self.media_directory, self.metadata_search_engine
        )

        # HLS streaming Flask routes
        self.app.add_url_rule(
            '/stream/<stream_id>/master.m3u8',
            'videos_stream_video',
            self._stream_video,
        )
        self.app.add_url_rule(
            '/stream/<stream_id>/<path:filename>',
            'videos_stream_segment',
            self._stream_segment,
        )

    def _register_schedulers(self):
        """Registers background schedulers for the module."""
        app = self.app
        cfg = self.cfg

        _check_and_submit_rating = make_scheduled_rating_check(
            app, 'Videos', self.file_manager, self.videos_evaluator, cfg, 'videos',
            self.update_model_ratings,
        )
        rating_update_interval = OmegaConf.select(cfg, 'videos.rating_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=rating_update_interval,
            fn=_check_and_submit_rating,
            name='Videos: rate unrated files',
            check_fn=lambda: self.videos_evaluator.hash is not None
            and len(self.file_manager.get_unrated_files(self.videos_evaluator.hash)) > 0,
        )

        _check_and_submit_description = make_scheduled_description_check(
            app, 'Videos', self.file_manager, self.metadata_search_engine, cfg, 'videos'
        )
        desc_interval = OmegaConf.select(cfg, 'videos.description_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=desc_interval,
            fn=_check_and_submit_description,
            name='Videos: describe undescribed files',
        )

    def _register_background_tasks(self):
        """One-time migration task.

        Previously user/model ratings for videos were stored in VideosLibrary
        (keyed by hash). They now belong in the shared FilesLibrary table (keyed
        by file_path) so they are consistent with the rest of the framework.
        This task normalizes old relative file_paths to VFS URLs and then copies
        any rating that exists in VideosLibrary but is missing (or unrated) in
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
                candidates = db_models.VideosLibrary.query.filter(
                    db_models.VideosLibrary.file_path.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[VideoModuleServer] DB query failed: {exc}")
                return 0

            to_fix = [row for row in candidates if '://' not in (row.file_path or '')]
            total = len(to_fix)
            if total == 0:
                return 0

            print(f"[VideoModuleServer] Normalizing {total} legacy file_path values to VFS URLs.")
            for i, row in enumerate(to_fix):
                ctx.check()
                ctx.update((i + 1) / total, f'Normalizing path {i + 1}/{total}')
                row.file_path = _normalize_path(row.file_path)

            db_models.db.session.commit()
            print(f"[VideoModuleServer] Normalized {total} paths.")
            return total

        def _check_if_migration_needed():
            try:
                rel_count = db_models.VideosLibrary.query.filter(
                    db_models.VideosLibrary.file_path.isnot(None),
                    ~db_models.VideosLibrary.file_path.like('%://%')
                ).count()
            except Exception as exc:
                print(f"[VideoModuleServer] DB query failed: {exc}")
                rel_count = 0
            if rel_count > 0:
                return True

            try:
                old_entries = db_models.VideosLibrary.query.filter(
                    db_models.VideosLibrary.user_rating.isnot(None)
                    | db_models.VideosLibrary.model_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[VideoModuleServer] DB query failed: {exc}")
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
            _normalize_old_paths(ctx)

            try:
                old_entries = db_models.VideosLibrary.query.filter(
                    db_models.VideosLibrary.user_rating.isnot(None)
                    | db_models.VideosLibrary.model_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[VideoModuleServer] DB query failed: {exc}")
                return

            total = len(old_entries)
            if total == 0:
                print("[VideoModuleServer] No rated videos found in old table.")
                return

            print(f"[VideoModuleServer] {total} rated videos found in old table.")
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
            print("[VideoModuleServer] Ratings copied successfully.")

        if _check_if_migration_needed():
            self.app.task_manager.submit(
                'VideoModuleServer: Copy ratings from old table', _copy_ratings_from_old_table
            )

    # ------------------------------------------------------------------
    # MODEL RATING PIPELINE
    # ------------------------------------------------------------------

    def update_model_ratings(self, files_list, ctx=None):
        """Re-compute AI model ratings and persist into FilesLibrary.

        Videos-specific: metadata-only strategy (no embedding proxy, since
        VideoSearch is a stub).
        """
        print('[VideoModuleServer] update_model_ratings')

        _progress = [0.0]

        def _status(msg):
            if ctx is not None:
                ctx.update(_progress[0], msg)
            else:
                self.cse.show_search_status(msg)

        def _check_if_paused():
            if ctx is not None:
                ctx.check()

        if self.videos_evaluator.hash is None:
            print('[VideoModuleServer] Universal evaluator not trained yet. Skipping.')
            return

        files_list_hash_map = {}
        for ind, file_path in enumerate(files_list):
            _check_if_paused()
            _progress[0] = (ind + 1) / len(files_list) * 0.15
            _status(f"Computing files hashes {ind + 1}/{len(files_list)}")
            file_hash = self.videos_search_engine.get_file_hash(file_path)
            files_list_hash_map[file_path] = file_hash

        _progress[0] = 0.15
        _status(f"Computing metadata embeddings for {len(files_list)} files...")
        all_embeddings = []
        embedding_dim = self.metadata_search_engine.text_embedder.embedding_dim or 1024
        for ind, full_path in enumerate(files_list):
            _check_if_paused()
            _progress[0] = 0.15 + (ind + 1) / len(files_list) * 0.55
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
                print(f"[VideoModuleServer] Embedding failed for {full_path}: {e}")
                all_embeddings.append(np.zeros((1, embedding_dim), dtype=np.float32))

        model_ratings = self.videos_evaluator.predict(all_embeddings)

        _progress[0] = 0.7
        _status("Updating model ratings of files...")
        new_items = []
        update_items = []
        for ind, full_path in enumerate(files_list):
            _check_if_paused()
            model_rating = float(model_ratings[ind])

            db_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            if db_item:
                db_item.model_rating = model_rating
                db_item.model_hash = self.videos_evaluator.hash
                update_items.append(db_item)
            else:
                file_data = {
                    "hash": files_list_hash_map[full_path],
                    "hash_algorithm": self.videos_search_engine.get_hash_algorithm(),
                    "file_path": full_path,
                    "model_rating": model_rating,
                    "model_hash": self.videos_evaluator.hash,
                }
                new_items.append(main_db_models.FilesLibrary(**file_data))

            _progress[0] = 0.7 + (ind + 1) / len(files_list) * 0.3
            _status(f"Updated model ratings for {ind + 1}/{len(files_list)} files.")

        if update_items:
            main_db_models.db.session.bulk_save_objects(update_items)
        if new_items:
            main_db_models.db.session.bulk_save_objects(new_items)
        main_db_models.db.session.commit()

    # ------------------------------------------------------------------
    # PLAY-TRACKING HELPERS (VideosLibrary, joined by file_path)
    # ------------------------------------------------------------------

    def _get_or_create_play_record(self, file_path, file_hash):
        """Return the VideosLibrary play-tracking row for file_path, reclaiming
        a legacy hash-keyed row if one exists, or creating a minimal one."""
        record = db_models.VideosLibrary.query.filter_by(file_path=file_path).first()
        if record is None and file_hash is not None:
            record = db_models.VideosLibrary.query.filter_by(hash=file_hash).first()
            if record is not None:
                record.file_path = file_path

        if record is None:
            record = db_models.VideosLibrary(
                hash=file_hash,
                hash_algorithm=self.videos_search_engine.get_hash_algorithm(),
                file_path=file_path,
                full_play_count=0,
                skip_count=0,
            )
            db_models.db.session.add(record)
            db_models.db.session.commit()
        return record

    # -------------------------------------------------------------------------
    # SOCKET EVENT HANDLERS
    # -------------------------------------------------------------------------

    def handle_get_folders(self, data):
        path = data.get('path', '')
        return self.file_manager.get_folders(path)

    def handle_get_files(self, data):
        """Scans the requested path and returns module-related files."""
        def filter_by_recommendation(all_files, text_query):
            all_paths = list(all_files)

            self.cse.show_search_status("Filtering by recommendation: loading play data from DB")
            video_data = db_models.VideosLibrary.query.with_entities(
                db_models.VideosLibrary.file_path,
                db_models.VideosLibrary.hash,
                db_models.VideosLibrary.user_rating,
                db_models.VideosLibrary.model_rating,
                db_models.VideosLibrary.full_play_count,
                db_models.VideosLibrary.skip_count,
                db_models.VideosLibrary.last_played,
            ).filter(db_models.VideosLibrary.file_path.in_(all_paths)).all()

            # Reclaim legacy hash-keyed rows
            covered_paths = {row.file_path for row in video_data}
            missing_paths = [p for p in all_paths if p not in covered_paths]
            legacy_rows = []
            if missing_paths:
                hash_to_path = {}
                for p in missing_paths:
                    try:
                        h = self.videos_search_engine.get_file_hash(p)
                    except OSError:
                        # File has been moved, deleted, or is unreadable since the
                        # directory scan. Legacy hash-keyed recovery is best-effort
                        # only — skip and move on.  No rating can be returned for a
                        # file that can't be opened anyway.
                        continue
                    if h is not None:
                        hash_to_path[h] = p
                if hash_to_path:
                    legacy_rows = db_models.VideosLibrary.query.with_entities(
                        db_models.VideosLibrary.file_path,
                        db_models.VideosLibrary.hash,
                        db_models.VideosLibrary.user_rating,
                        db_models.VideosLibrary.model_rating,
                        db_models.VideosLibrary.full_play_count,
                        db_models.VideosLibrary.skip_count,
                        db_models.VideosLibrary.last_played,
                    ).filter(db_models.VideosLibrary.hash.in_(list(hash_to_path.keys()))).all()
                    legacy_rows = [
                        (hash_to_path[r.hash], r) for r in legacy_rows if r.hash in hash_to_path
                    ]

            # Ratings live in FilesLibrary; merge them onto the play-tracking rows.
            rating_data = main_db_models.FilesLibrary.query.with_entities(
                main_db_models.FilesLibrary.file_path,
                main_db_models.FilesLibrary.user_rating,
                main_db_models.FilesLibrary.model_rating,
            ).filter(main_db_models.FilesLibrary.file_path.in_(all_paths)).all()

            path_to_data = {
                row.file_path: {
                    'file_path': row.file_path,
                    'user_rating': row.user_rating,
                    'model_rating': row.model_rating,
                    'full_play_count': row.full_play_count,
                    'skip_count': row.skip_count,
                    'last_played': row.last_played,
                }
                for row in video_data
            }
            for canonical_path, row in legacy_rows:
                path_to_data[canonical_path] = {
                    'file_path': canonical_path,
                    'user_rating': row.user_rating,
                    'model_rating': row.model_rating,
                    'full_play_count': row.full_play_count,
                    'skip_count': row.skip_count,
                    'last_played': row.last_played,
                }
            for row in rating_data:
                entry = path_to_data.setdefault(row.file_path, {
                    'file_path': row.file_path, 'user_rating': None, 'model_rating': None,
                    'full_play_count': 0, 'skip_count': 0, 'last_played': None,
                })
                entry['user_rating'] = row.user_rating
                entry['model_rating'] = row.model_rating

            full_video_data_for_sorting = [
                path_to_data.get(fp, {
                    'file_path': fp, 'user_rating': None, 'model_rating': None,
                    'full_play_count': 0, 'skip_count': 0, 'last_played': None,
                })
                for fp in all_paths
            ]

            self.cse.show_search_status("Filtering by recommendation: sorting files")
            scores = sort_files_by_recommendation(all_paths, full_video_data_for_sorting)
            return scores

        filters = {
            "by_text": self.common_filters.filter_by_text,
            "random": self.common_filters.filter_by_random,
            "rating": self.common_filters.filter_by_rating,
            "recommendation": filter_by_recommendation,
        }

        def get_file_info(full_path):
            # Ratings come from FilesLibrary
            files_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            user_rating = files_item.user_rating if files_item else None
            model_rating = files_item.model_rating if files_item else None
            rating_is_stale = (
                model_rating is not None
                and self.videos_evaluator.hash is not None
                and files_item is not None
                and files_item.model_hash != self.videos_evaluator.hash
            )

            # Play-tracking comes from VideosLibrary
            play_item = db_models.VideosLibrary.query.filter_by(file_path=full_path).first()
            last_played = "Never"
            if play_item:
                if play_item.last_played:
                    last_played_timestamp = play_item.last_played.timestamp()
                    last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())

            # Preview generation (VFS-aware: resolve to local path for moviepy)
            basename = os.path.basename(full_path)
            local_path, _temp = vfs.resolve_to_local_path(full_path)
            preview_local = os.path.join(os.path.dirname(local_path), basename + ".preview.png")
            preview_path = os.path.join(os.path.dirname(full_path), basename + ".preview.png")

            if not os.path.exists(preview_local):
                self.cse.show_search_status(f"Generating preview for {basename}...")
                try:
                    generate_preview(local_path, preview_local)
                    self.cse.show_search_status(f"Generated preview for {basename}.")
                except Exception as e:
                    print(f"Error generating preview for {basename}: {e}")
                    self.cse.show_search_status(f"Failed to generate preview for {basename}.")

            file_size = os.path.getsize(local_path) if os.path.exists(local_path) else 0

            return {
                "user_rating": user_rating,
                "model_rating": model_rating,
                "rating_is_stale": rating_is_stale,
                "last_played": last_played,
                "preview_path": preview_path,
                "file_size": convert_size(file_size),
                "resolution": "N/A",
                "length": "N/A",
            }

        input_params = data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
        })
        return self.file_manager.get_files(**input_params)

    def handle_open_file_in_folder(self, file_path):
        import src.file_manager as file_manager
        file_manager.open_file_in_folder(file_path)

    def handle_send_file_to_trash(self, file_path):
        if os.path.isfile(file_path):
            send2trash.send2trash(file_path)
            print(f"File '{file_path}' sent to trash.")
        else:
            print(f"Error: File '{file_path}' does not exist.")

    def handle_video_start_playing(self, data):
        """Records that a video started playing (updates last_played in VideosLibrary).

        The frontend sends ``{ file_path: ... }`` or a bare file_path string.
        """
        if isinstance(data, dict):
            file_path = data.get('file_path')
        else:
            file_path = data

        if file_path is None:
            print('[VideoModuleServer] video_start_playing: no file_path provided, skipping.')
            return

        file_hash = self.videos_search_engine.get_file_hash(file_path)
        record = self._get_or_create_play_record(file_path, file_hash)
        record.last_played = datetime.datetime.now()
        db_models.db.session.commit()

    # ------------------------------------------------------------------
    # HLS STREAMING SUBSYSTEM
    # ------------------------------------------------------------------

    def handle_start_streaming(self, file_path):
        """Start streaming a video file. Returns {stream_id, stream_url}."""
        # Resolve VFS URL → local path for FFmpeg
        local_path, _temp = vfs.resolve_to_local_path(file_path)

        if not os.path.isfile(local_path):
            print(f"Error: File not found at {local_path}")
            return {"error": "File not found"}

        stream_id = str(uuid.uuid4())
        self.active_transcodings[stream_id] = {
            'path': local_path,
            'process': None,
            'start_time': time.time(),
        }

        return {
            'stream_id': stream_id,
            'stream_url': f'/stream/{stream_id}/master.m3u8',
        }

    def handle_stop_streaming(self, stream_id):
        """Stop streaming and clean up resources."""
        if stream_id in self.active_transcodings:
            print(f"Stopping stream {stream_id}")
            self._cleanup_transcoding(stream_id)
            return {"status": "success", "message": "Stream stopped and resources cleaned up"}
        else:
            print(f"Stream {stream_id} not found or already stopped")
            return {"status": "error", "message": "Stream not found"}

    def _stream_video(self, stream_id):
        """Flask route: Stream a video with the given stream_id (HLS via FFmpeg)."""
        if stream_id not in self.active_transcodings:
            return "Stream not found", 404

        # If stream is already set up, just serve the playlist
        if 'temp_dir' in self.active_transcodings[stream_id]:
            master_playlist = os.path.join(self.active_transcodings[stream_id]['temp_dir'], "master.m3u8")
            if os.path.exists(master_playlist):
                return send_from_directory(self.active_transcodings[stream_id]['temp_dir'], "master.m3u8")

        video_path = self.active_transcodings[stream_id]['path']
        print(f"Starting stream for {stream_id} from {video_path}")

        temp_dir = tempfile.mkdtemp()
        self.active_transcodings[stream_id]['temp_dir'] = temp_dir
        master_playlist = os.path.join(temp_dir, "master.m3u8")

        # Get video duration
        try:
            metadata_cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'json', video_path,
            ]
            metadata_output = subprocess.check_output(metadata_cmd).decode('utf-8').strip()
            metadata = json.loads(metadata_output)
            duration = float(metadata['format']['duration'])
            self.active_transcodings[stream_id]['duration'] = duration
        except Exception as e:
            print(f"Error getting video metadata: {e}")
            duration = 0

        # Build FFmpeg command for HLS
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', video_path,
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-c:a', 'aac',
            '-ar', '44100',
            '-ac', '2',
            '-b:a', '128k',
            '-f', 'hls',
            '-hls_time', '2',
            '-hls_list_size', '0',
            '-hls_flags', 'independent_segments+split_by_time+append_list',
            '-hls_segment_type', 'mpegts',
            '-hls_playlist_type', 'event',
            '-force_key_frames', 'expr:gte(t,n_forced*2)',
            '-g', '48',
            '-start_number', '0',
            '-hls_segment_filename', os.path.join(temp_dir, 'segment_%03d.ts'),
            master_playlist,
        ]

        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.active_transcodings[stream_id]['process'] = process

        # Wait for the master playlist + initial segments
        start_time = time.time()
        while not os.path.exists(master_playlist) and time.time() - start_time < 10:
            time.sleep(0.1)
            if process.poll() is not None:
                return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500

        if not os.path.exists(master_playlist):
            return "Timeout waiting for playlist to be created", 500

        # Wait for at least 3 segments
        segments_ready = 0
        segment_pattern = os.path.join(temp_dir, "segment_*.ts")
        while segments_ready < 3 and time.time() - start_time < 10:
            segments_ready = len(glob.glob(segment_pattern))
            if segments_ready >= 3:
                break
            time.sleep(0.2)
            if process.poll() is not None:
                return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500

        print(f"Starting playback with {segments_ready} segments ready")
        return send_from_directory(temp_dir, "master.m3u8")

    def _stream_segment(self, stream_id, filename):
        """Flask route: Serve HLS segment files."""
        if stream_id not in self.active_transcodings or 'temp_dir' not in self.active_transcodings[stream_id]:
            return "Stream not found", 404
        temp_dir = self.active_transcodings[stream_id]['temp_dir']
        return send_from_directory(temp_dir, filename)

    def _cleanup_transcoding(self, stream_id):
        """Clean up FFmpeg process, temp files, and dict entry for a stream."""
        if stream_id not in self.active_transcodings:
            return

        entry = self.active_transcodings[stream_id]

        if entry.get('process'):
            try:
                entry['process'].terminate()
                print(f"Process for stream {stream_id} terminated")
            except Exception as e:
                print(f"Error terminating process: {e}")

        if 'error_log' in entry:
            try:
                entry['error_log'].close()
            except Exception:
                pass

        if 'temp_dir' in entry:
            try:
                shutil.rmtree(entry['temp_dir'])
                print(f"Removed temp directory for stream {stream_id}")
            except Exception as e:
                print(f"Error removing temp directory: {e}")

        del self.active_transcodings[stream_id]
        print(f"Stream {stream_id} completely cleaned up")

    def _cleanup_loop(self):
        """Daemon thread: kill streams older than 1 hour."""
        while True:
            try:
                current_time = time.time()
                for stream_id in list(self.active_transcodings.keys()):
                    if current_time - self.active_transcodings[stream_id]['start_time'] > 3600:
                        print(f"Cleaning up expired stream: {stream_id}")
                        self._cleanup_transcoding(stream_id)
                time.sleep(300)
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
                time.sleep(60)


# -------------------------------------------------------------------------
# MODULE FACTORY ENTRY POINT
# -------------------------------------------------------------------------
def register_module(app, socketio, cfg, data_folder):
    """
    Standardized entry point called by the ExtensionManager.
    It instantiates the controller and boots it up.
    """
    module_server = VideoModuleServer(app, socketio, cfg, data_folder)
    module_server.initialize()
    return module_server
