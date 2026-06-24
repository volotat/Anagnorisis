import os
import datetime

import numpy as np
from omegaconf import OmegaConf

from modules.music.engine import MusicSearch
from src.socket_events import CommonSocketEvents
from src.file_manager import FileManager
from src.common_filters import CommonFilters
from src.metadata_search import MetadataSearch

import modules.music.db_models as db_models
from src.universal_evaluator import UniversalEvaluator
from src.embedding_proxy import EmbeddingProxyGenerator

import src.db_models as main_db_models

from src.utils import convert_length, time_difference, SortingProgressCallback, EmbeddingGatheringCallback
from src.recommendation_engine import sort_files_by_recommendation
from src.scheduler import Scheduler
from src.module_helpers import (
    register_meta_handlers,
    make_scheduled_rating_check,
    make_scheduled_description_check,
)

# -------------------------------------------------------------------------
# EVENTS
# -------------------------------------------------------------------------
# Incoming (handled via socketio.on_event):
#   emit_music_page_get_folders
#   emit_music_page_get_files
#   emit_music_page_get_song_details
#   emit_music_page_set_song_play_rate
#   emit_music_page_set_song_rating
#   emit_music_page_update_song_info
#   emit_music_page_song_start_playing
#   emit_music_page_get_path_to_media_folder
#   emit_music_page_update_path_to_media_folder
#   emit_music_page_open_file_in_folder
#
# Outgoing (emitted via socketio.emit):
#   emit_music_page_show_song_details
#   emit_music_page_show_path_to_media_folder
#   (status events are emitted automatically by CommonSocketEvents)
# -------------------------------------------------------------------------


class MusicModuleServer:
    """
    Class-Based Module Controller for the 'music' extension.

    Architecture mirrors modules/text/serve.py:
      * User / model ratings live in the shared FilesLibrary table (keyed by
        file_path) so they work with FileManager.get_unrated_files() and
        CommonFilters.filter_by_rating().
      * Play-tracking data (full_play_count / skip_count / last_played) lives
        in the module-specific MusicLibrary table, joined by file_path.
      * All file I/O is VFS / remote-server aware.
    """

    def __init__(self, app, socketio, cfg, app_root_folder):
        # 1. Store core dependencies
        self.app = app
        self.socketio = socketio
        self.cfg = cfg
        self.app_root_folder = app_root_folder
        self.cse = CommonSocketEvents(socketio, module_name="music")
        self.media_directory = cfg.music.media_directory

        if self.media_directory is None:
            raise ValueError("[MusicModuleServer] Music media folder is not set.")

    def initialize(self):
        """Main lifecycle hook to boot up the module."""

        self.cse.show_loading_status('Initializing music search engine...')
        self.music_search_engine = MusicSearch(cfg=self.cfg)

        self.cse.show_loading_status('Loading audio embedding models...')
        self.music_search_engine.initiate(
            models_folder=self.cfg.main.embedding_models_path,
            cache_folder=self.cfg.main.cache_path,
        )

        self.cse.show_loading_status('Initializing universal evaluator for music module...')
        self.music_evaluator = UniversalEvaluator()

        self.cse.show_loading_status('Loading universal evaluator model...')
        self.music_evaluator.load(
            os.path.join(self.cfg.main.personal_models_path, 'universal_evaluator.pt')
        )

        self.cse.show_loading_status('Setting up file manager...')
        self.file_manager = FileManager(
            app=self.app,
            cfg=self.cfg,
            media_directory=self.media_directory,
            engine=self.music_search_engine,
            module_name="music",
            media_formats=self.cfg.music.media_formats,
            socketio=self.socketio,
            db_schema=main_db_models.FilesLibrary,
        )

        # Create metadata search engine
        self.cse.show_loading_status('Initializing metadata search...')
        self.metadata_search_engine = MetadataSearch(engine=self.music_search_engine)

        # Set up embedding proxy so that files without an OmniDescriptor description
        # still receive a meaningful text representation (CLAP tags + fingerprint).
        self.cse.show_loading_status('Initializing embedding proxy...')
        _tag_list = list(OmegaConf.select(self.cfg, 'music.embedding_tags', default=[]) or [])
        _tag_threshold_raw = OmegaConf.select(self.cfg, 'music.embedding_tags_threshold', default=0.25)
        _tag_threshold = float(_tag_threshold_raw) if _tag_threshold_raw is not None else None
        self.music_proxy_gen = EmbeddingProxyGenerator(
            engine=self.music_search_engine,
            tag_list=_tag_list,
            threshold=_tag_threshold,
            cache_path=self.cfg.main.cache_path,
            model_name=getattr(self.cfg.audio_embedder, 'model_name', 'CLAP'),
        )
        self.metadata_search_engine.embedding_proxy = self.music_proxy_gen

        # Create common filters instance
        self.cse.show_loading_status('Setting up filters...')
        self.common_filters = CommonFilters(
            engine=self.music_search_engine,
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

        self.cse.show_loading_status('Music module ready!')

    def _register_socket_events(self):
        """Maps Socket.IO event strings directly to class methods."""
        self.socketio.on_event('emit_music_page_get_folders', self.handle_get_folders)
        self.socketio.on_event('emit_music_page_get_files', self.handle_get_files)
        self.socketio.on_event('emit_music_page_get_song_details', self.handle_get_song_details)
        self.socketio.on_event('emit_music_page_set_song_play_rate', self.handle_set_play_rate)
        self.socketio.on_event('emit_music_page_set_song_rating', self.handle_set_song_rating)
        self.socketio.on_event('emit_music_page_update_song_info', self.handle_update_song_info)
        self.socketio.on_event('emit_music_page_song_start_playing', self.handle_song_start_playing)
        self.socketio.on_event('emit_music_page_get_path_to_media_folder', self.handle_get_path_to_media_folder)
        self.socketio.on_event('emit_music_page_update_path_to_media_folder', self.handle_update_path_to_media_folder)
        self.socketio.on_event('emit_music_page_open_file_in_folder', self.handle_open_file_in_folder)

        # .meta sidecar handlers + full description handler (shared helper)
        register_meta_handlers(
            self.socketio, 'music', lambda: self.media_directory, self.metadata_search_engine
        )

    def _register_schedulers(self):
        """Registers background schedulers for the module."""
        app = self.app
        cfg = self.cfg

        # Proactively rate unrated files using the shared factory (writes to FilesLibrary)
        _check_and_submit_rating = make_scheduled_rating_check(
            app, 'Music', self.file_manager, self.music_evaluator, cfg, 'music',
            self.update_model_ratings,
        )
        rating_update_interval = OmegaConf.select(cfg, 'music.rating_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=rating_update_interval,
            fn=_check_and_submit_rating,
            name='Music: rate unrated files',
            check_fn=lambda: self.music_evaluator.hash is not None
            and len(self.file_manager.get_unrated_files(self.music_evaluator.hash)) > 0,
        )

        # Proactively describe undescribed files using the shared factory
        _check_and_submit_description = make_scheduled_description_check(
            app, 'Music', self.file_manager, self.metadata_search_engine, cfg, 'music'
        )
        desc_interval = OmegaConf.select(cfg, 'music.description_update_interval_minutes', default=None)
        Scheduler(
            app,
            interval_minutes=desc_interval,
            fn=_check_and_submit_description,
            name='Music: describe undescribed files',
        )

    def _register_background_tasks(self):
        """One-time migration task.

        Previously user/model ratings for music were stored in MusicLibrary.
        They now belong in the shared FilesLibrary table (keyed by file_path)
        so they are consistent with the rest of the framework. This task copies
        any user rating that exists in MusicLibrary but is missing (or unrated)
        in FilesLibrary. It is a no-op once everything has been migrated.
        """

        def _check_if_migration_needed():
            try:
                old_entries = db_models.MusicLibrary.query.filter(
                    db_models.MusicLibrary.user_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[MusicModuleServer] DB query failed: {exc}")
                return False

            for old_entry in old_entries:
                if not old_entry.file_path:
                    continue
                new_entry = main_db_models.FilesLibrary.query.filter_by(
                    file_path=old_entry.file_path
                ).first()
                if not new_entry or new_entry.user_rating is None:
                    return True  # Migration needed for at least one file
            return False

        def _copy_ratings_from_old_table(ctx):
            try:
                old_entries = db_models.MusicLibrary.query.filter(
                    db_models.MusicLibrary.user_rating.isnot(None)
                ).all()
            except Exception as exc:
                print(f"[MusicModuleServer] DB query failed: {exc}")
                return

            total = len(old_entries)
            if total == 0:
                print("[MusicModuleServer] No user-rated tracks found in old table.")
                return

            print(f"[MusicModuleServer] {total} user-rated tracks found in old table.")
            for i, old_entry in enumerate(old_entries):
                ctx.check()
                ctx.update((i + 1) / total, f'Copying rating for {i + 1}/{total}: {old_entry.file_path}')

                if not old_entry.file_path:
                    continue

                new_entry = main_db_models.FilesLibrary.query.filter_by(
                    file_path=old_entry.file_path
                ).first()
                if new_entry:
                    # Don't overwrite an existing user rating
                    if new_entry.user_rating is None:
                        new_entry.user_rating = old_entry.user_rating
                        new_entry.user_rating_date = old_entry.user_rating_date
                else:
                    new_entry = main_db_models.FilesLibrary(
                        file_path=old_entry.file_path,
                        hash=old_entry.hash,
                        hash_algorithm=old_entry.hash_algorithm,
                        user_rating=old_entry.user_rating,
                        user_rating_date=old_entry.user_rating_date,
                    )
                    main_db_models.db.session.add(new_entry)

            main_db_models.db.session.commit()
            print("[MusicModuleServer] User ratings copied successfully.")

        if _check_if_migration_needed():
            self.app.task_manager.submit(
                'MusicModuleServer: Copy ratings from old table', _copy_ratings_from_old_table
            )

    # ------------------------------------------------------------------
    # MODEL RATING PIPELINE
    # ------------------------------------------------------------------

    def update_model_ratings(self, files_list, ctx=None):
        """
        Re-compute AI model ratings for the given files and persist them into
        FilesLibrary (keyed by file_path).

        Music-specific pipeline (kept from the legacy module):
          1. Compute CLAP audio embeddings (fast — disk-cached).
          2. Pre-populate the embedding-proxy cache (CLAP tags + fingerprint).
          3. Build a text description (incl. proxy section) + Jina embedding.
          4. Predict with the universal evaluator and persist.
        """
        print('[MusicModuleServer] update_model_ratings')

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

        # Skip if the universal evaluator has not been trained yet
        if self.music_evaluator.hash is None:
            print('[MusicModuleServer] Universal evaluator not trained yet. Skipping model rating update.')
            return

        files_list_hash_map = {}
        for ind, file_path in enumerate(files_list):
            _check_if_paused()
            _progress[0] = (ind + 1) / len(files_list) * 0.1
            _status(f"Computing files hashes {ind + 1}/{len(files_list)}")
            file_hash = self.music_search_engine.get_file_hash(file_path)
            files_list_hash_map[file_path] = file_hash

        # Step 1: Compute CLAP embeddings (fast — results are disk-cached)
        _progress[0] = 0.1
        _status(f"Computing CLAP embeddings for {len(files_list)} files...")
        embeddings = self.music_search_engine.process_audio(
            files_list, callback=_embedding_callback, media_folder=self.media_directory
        )  # [N, D] tensor

        # Step 2: Pre-populate proxy cache before the Jina embedding phase.
        # This ensures generate_full_description() always hits the proxy cache and
        # never needs to load CLAP while Jina is active.
        _progress[0] = 0.3
        _status("Preparing embedding proxies...")
        for i, fp in enumerate(files_list):
            _check_if_paused()
            try:
                self.music_proxy_gen.compute_proxy_section(
                    files_list_hash_map[fp], embeddings[i].cpu().numpy()
                )
            except Exception as e:
                print(f"[MusicModuleServer] Proxy generation failed for {fp}: {e}")
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
                print(f"[MusicModuleServer] Embedding failed for {full_path}: {e}")
                all_embeddings.append(np.zeros((1, embedding_dim), dtype=np.float32))

        # Step 4: Predict with the universal evaluator and persist into FilesLibrary
        model_ratings = self.music_evaluator.predict(all_embeddings)

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
                db_item.model_hash = self.music_evaluator.hash
                update_items.append(db_item)
            else:
                file_data = {
                    "hash": files_list_hash_map[full_path],
                    "hash_algorithm": self.music_search_engine.get_hash_algorithm(),
                    "file_path": full_path,
                    "model_rating": model_rating,
                    "model_hash": self.music_evaluator.hash,
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
    # PLAY-TRACKING HELPERS (MusicLibrary, joined by file_path)
    # ------------------------------------------------------------------

    def _get_or_create_play_record(self, file_path, file_hash):
        """Return the MusicLibrary play-tracking row for file_path, creating a
        minimal one if it does not exist yet."""
        record = db_models.MusicLibrary.query.filter_by(file_path=file_path).first()
        if record is None:
            record = db_models.MusicLibrary(
                hash=file_hash,
                hash_algorithm=self.music_search_engine.get_hash_algorithm(),
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
        """Returns a list of folders in the specified path."""
        path = data.get('path', '')
        return self.file_manager.get_folders(path)

    def handle_get_files(self, data):
        """Scans the requested path and returns module-related files."""
        # Music-specific filters
        def filter_by_length(all_files, text_query):
            self.cse.show_search_status("Gathering durations for sorting...")

            durations = {}
            progress_callback = SortingProgressCallback(
                self.cse.show_search_status, operation_name="Gathering music duration "
            )
            for ind, full_path in enumerate(all_files):
                file_metadata = self.music_search_engine.get_metadata(full_path)
                if 'duration' not in file_metadata:
                    raise Exception(f"Duration not found for file: {full_path}")

                durations[full_path] = file_metadata['duration']
                progress_callback(ind + 1, len(all_files))

            none_durations = [fp for fp, d in durations.items() if d is None]
            if none_durations:
                print("Files with None duration:", none_durations)

            return durations

        def filter_by_recommendation(all_files, text_query):
            all_paths = list(all_files)

            self.cse.show_search_status("Filtering by recommendation: loading play data from DB")
            music_data = db_models.MusicLibrary.query.with_entities(
                db_models.MusicLibrary.file_path,
                db_models.MusicLibrary.user_rating,
                db_models.MusicLibrary.model_rating,
                db_models.MusicLibrary.full_play_count,
                db_models.MusicLibrary.skip_count,
                db_models.MusicLibrary.last_played,
            ).filter(db_models.MusicLibrary.file_path.in_(all_paths)).all()

            # Ratings live in FilesLibrary; merge them onto the play-tracking rows.
            rating_data = main_db_models.FilesLibrary.query.with_entities(
                main_db_models.FilesLibrary.file_path,
                main_db_models.FilesLibrary.user_rating,
                main_db_models.FilesLibrary.model_rating,
            ).filter(main_db_models.FilesLibrary.file_path.in_(all_paths)).all()

            keys = ['file_path', 'user_rating', 'model_rating',
                    'full_play_count', 'skip_count', 'last_played']
            # Start from play-tracking rows (carry full_play_count / skip_count / last_played)
            path_to_data = {
                row.file_path: {
                    'file_path': row.file_path,
                    'user_rating': row.user_rating,
                    'model_rating': row.model_rating,
                    'full_play_count': row.full_play_count,
                    'skip_count': row.skip_count,
                    'last_played': row.last_played,
                }
                for row in music_data
            }
            # Overlay authoritative ratings from FilesLibrary
            for row in rating_data:
                entry = path_to_data.setdefault(row.file_path, {
                    'file_path': row.file_path, 'user_rating': None, 'model_rating': None,
                    'full_play_count': 0, 'skip_count': 0, 'last_played': None,
                })
                entry['user_rating'] = row.user_rating
                entry['model_rating'] = row.model_rating

            full_music_data_for_sorting = [
                path_to_data.get(fp, {
                    'file_path': fp, 'user_rating': None, 'model_rating': None,
                    'full_play_count': 0, 'skip_count': 0, 'last_played': None,
                })
                for fp in all_paths
            ]

            self.cse.show_search_status("Filtering by recommendation: sorting files")
            scores = sort_files_by_recommendation(all_paths, full_music_data_for_sorting)
            return scores

        # Define available filters
        filters = {
            "by_file": self.common_filters.filter_by_file,
            "by_text": self.common_filters.filter_by_text,
            "file_size": self.common_filters.filter_by_file_size,
            "length": filter_by_length,
            "similarity": self.common_filters.filter_by_similarity,
            "random": self.common_filters.filter_by_random,
            "rating": self.common_filters.filter_by_rating,
            "recommendation": filter_by_recommendation,
        }

        # Gather domain-specific file information (single-arg, per FileManager contract)
        def get_file_info(full_path):
            # Ratings come from FilesLibrary
            files_item = main_db_models.FilesLibrary.query.filter_by(file_path=full_path).first()
            user_rating = files_item.user_rating if files_item else None
            model_rating = files_item.model_rating if files_item else None
            rating_is_stale = (
                model_rating is not None
                and self.music_evaluator.hash is not None
                and files_item is not None
                and files_item.model_hash != self.music_evaluator.hash
            )

            # Play-tracking comes from MusicLibrary
            play_item = db_models.MusicLibrary.query.filter_by(file_path=full_path).first()
            last_played = "Never"
            full_play_count = 0
            skip_count = 0
            if play_item:
                full_play_count = play_item.full_play_count or 0
                skip_count = play_item.skip_count or 0
                if play_item.last_played:
                    last_played_timestamp = play_item.last_played.timestamp()
                    last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())

            audiofile_data = self.music_search_engine.get_metadata(full_path)

            return {
                "user_rating": user_rating,
                "model_rating": model_rating,
                "rating_is_stale": rating_is_stale,
                "full_play_count": full_play_count,
                "skip_count": skip_count,
                "last_played": last_played,
                "audiofile_data": audiofile_data,
                "length": convert_length(audiofile_data['duration']),
            }

        input_params = data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
        })
        return self.file_manager.get_files(**input_params)

    def handle_get_song_details(self, data):
        """Returns detailed metadata + ratings for a single track."""
        file_path = data.get('file_path', '')
        audiofile_data = self.music_search_engine.get_metadata(file_path)
        audiofile_data['hash'] = self.music_search_engine.get_file_hash(file_path)

        files_item = main_db_models.FilesLibrary.query.filter_by(file_path=file_path).first()
        audiofile_data['user_rating'] = files_item.user_rating if files_item else None
        audiofile_data['model_rating'] = files_item.model_rating if files_item else None
        audiofile_data['file_path'] = file_path

        self.socketio.emit('emit_music_page_show_song_details', audiofile_data)

    def handle_set_play_rate(self, data):
        """Increments full_play_count or skip_count when a song ends / is skipped."""
        if not data:
            return

        file_path = data.get('file_path')
        skip_score_change = data.get('skip_score_change')
        if file_path is None:
            return

        print('[MusicModuleServer] Set song play rate:', file_path, skip_score_change)
        file_hash = self.music_search_engine.get_file_hash(file_path)
        song = db_models.MusicLibrary.query.filter_by(file_path=file_path).first()
        if song is None:
            song = self._get_or_create_play_record(file_path, file_hash)

        if skip_score_change == 1:
            song.full_play_count = (song.full_play_count or 0) + 1
        if skip_score_change == -1:
            song.skip_count = (song.skip_count or 0) + 1

        db_models.db.session.commit()

    def handle_set_song_rating(self, data):
        """Saves a user rating for a track into FilesLibrary (keyed by file_path)."""
        file_path = data.get('file_path')
        song_score = data.get('score')

        print('[MusicModuleServer] Set song rating:', file_path, song_score)

        db_item = main_db_models.FilesLibrary.query.filter_by(file_path=file_path).first()
        if db_item is None:
            file_hash = self.music_search_engine.get_file_hash(file_path)
            db_item = main_db_models.FilesLibrary(
                hash=file_hash,
                hash_algorithm=self.music_search_engine.get_hash_algorithm(),
                file_path=file_path,
            )
            main_db_models.db.session.add(db_item)

        db_item.user_rating = float(song_score)
        db_item.user_rating_date = datetime.datetime.now()
        main_db_models.db.session.commit()

    def handle_update_song_info(self, data):
        # NOTE: It is very important to update the hash of the file in the database
        # after the metadata has been updated to not lose user rating of the song
        # and other connected data.
        pass

    def handle_song_start_playing(self, data):
        """Records that a track started playing (updates last_played in MusicLibrary).

        The frontend may send either a bare hash (legacy) or a dict with file_path.
        """
        # Normalize input: accept {'file_path': ...}, {'hash': ...}, or a bare value.
        if isinstance(data, dict):
            file_path = data.get('file_path')
        else:
            file_path = None

        if file_path is None:
            # Legacy callers sent a hash; we can't reliably resolve a path, so no-op.
            print('[MusicModuleServer] song_start_playing: no file_path provided, skipping.')
            return

        song = db_models.MusicLibrary.query.filter_by(file_path=file_path).first()
        if song is None:
            file_hash = self.music_search_engine.get_file_hash(file_path)
            song = self._get_or_create_play_record(file_path, file_hash)

        song.last_played = datetime.datetime.now()
        db_models.db.session.commit()

    def handle_get_path_to_media_folder(self, data=None):
        self.socketio.emit('emit_music_page_show_path_to_media_folder', self.cfg.music.media_directory)

    def handle_update_path_to_media_folder(self, new_path):
        self.cfg.music.media_directory = new_path
        self.media_directory = new_path

        # Update the configuration file on disk
        config_path = os.path.join(self.app_root_folder, 'Anagnorisis-app', 'config.yaml')
        try:
            with open(config_path, 'w') as file:
                OmegaConf.save(self.cfg, file)
        except Exception as e:
            print(f"[MusicModuleServer] Failed to persist config update: {e}")

        self.socketio.emit('emit_music_page_show_path_to_media_folder', self.cfg.music.media_directory)

    def handle_open_file_in_folder(self, file_path):
        # Reuse the shared implementation in src.file_manager
        import src.file_manager as file_manager
        file_manager.open_file_in_folder(file_path)


# -------------------------------------------------------------------------
# MODULE FACTORY ENTRY POINT
# -------------------------------------------------------------------------
def register_module(app, socketio, cfg, data_folder):
    """
    Standardized entry point called by the ExtensionManager.
    It instantiates the controller and boots it up.
    """
    module_server = MusicModuleServer(app, socketio, cfg, data_folder)
    module_server.initialize()
    return module_server
