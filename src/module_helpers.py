"""
module_helpers.py — Shared helpers that eliminate boilerplate across modules.

Each function either registers socket handlers or returns a callable that
can be passed to ``Scheduler``.
"""

import os
from omegaconf import OmegaConf
import random

from src.app_factory.path_guard import PathNotAuthorized
import src.app_factory.path_guard as path_guard

# ---------------------------------------------------------------------------
# .meta file handlers + full description handler
# ---------------------------------------------------------------------------

def register_meta_handlers(socketio, module_name, metadata_search, app=None):
    """Register get/save .meta and get_full_description socket handlers.

    Args:
        socketio:        Flask-SocketIO instance.
        module_name:     e.g. "images" — used to build event names.
        metadata_search: MetadataSearch instance.
        app:             Flask app (needed for path authorization; callers that
                         register the handlers from within a module server
                         should pass self.app).
    """
    prefix = f'emit_{module_name}_page'

    @socketio.on(f'{prefix}_get_external_metadata_file_content')
    def get_external_metadata_file_content(file_path):
        metadata_file_path = file_path + ".meta"
        content = ""
        try:
            if app is not None:
                path_guard.authorize_client_url(app, file_path)
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            print(f"Read external metadata for {file_path}")
        except PathNotAuthorized as e:
            print(f"[MetaHandlers] Blocked unauthorized read: {e}")
        except Exception as e:
            print(f"Error reading external metadata for {file_path}: {e}")
        return {"content": content, "file_path": file_path}

    @socketio.on(f'{prefix}_save_external_metadata_file_content')
    def save_external_metadata_file_content(data):
        file_path = data['file_path']
        metadata_content = data['metadata_content']
        metadata_file_path = file_path + ".meta"
        try:
            if app is not None:
                path_guard.authorize_client_url(app, file_path)
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            print(f"Saved metadata for {file_path}")
            # The sidecar is part of the description but not of its cache key
            # (stat-ing every .meta on every probe would mean a network call per
            # remote file), so the one place that knows it changed drops the
            # stale embedding here.
            metadata_search.invalidate(file_path)
        except PathNotAuthorized as e:
            print(f"[MetaHandlers] Blocked unauthorized write: {e}")
        except Exception as e:
            print(f"Error saving metadata for {file_path}: {e}")


from src.file_manager import FileManager

def make_scheduled_embedding_check(app, label, file_manager: FileManager, engine, cfg, cfg_key):
    """Return a callable for ``Scheduler`` that finds files without cached
    embeddings and submits an embedding task to ``app.task_manager``.

    The walker is hard-coded to ``osfs:///mnt/media/`` — i.e. LOCAL files only.
    A content embedding is made from the file's own bytes, so including a
    remote file here would mean downloading it. Remote files
    only get an embedding if the user explicitly opens them in a context
    that triggers MemorySystem (direct user action), or if they are
    accessed through the search hot path which embeds the query on CPU.

    Capped per cycle via ``cfg.<cfg_key>.embedding_update_batch_size``
    so a single run cannot monopolise the task queue. Whatever isn't
    processed this cycle is picked up on the next scheduled tick.
    """
    def _check_and_submit_embedding():
        if not engine.model_hash:
            return

        media_formats = OmegaConf.select(cfg, f'{cfg_key}.media_formats', default=[]) or []
        all_files = file_manager._walk_files_cached("osfs:///mnt/media/", set(media_formats)) #file_manager.media_directory
        if not all_files:
            return

        # Cheap probe — RAM-only, stat-based cache key, no model load.
        unindexed = []
        for fp in all_files:
            cache_key = engine.make_embedding_cache_key(fp)
            if engine._fast_cache.get(cache_key) is None:
                unindexed.append(fp)

        if not unindexed:
            return

        total = len(unindexed)

        # --- cap the per-cycle batch so we never monopolise the queue ---
        batch_cfg = OmegaConf.select(cfg, f'{cfg_key}.embedding_update_batch_size', default=None)
        batch_size = min(batch_cfg, total) if batch_cfg else total

        # Sample files randomly to ensure that all folders gets roughly equal indexing distribution
        batch = random.sample(unindexed, batch_size)
        # -------------------------------------------------------------------------

        base_name = f'{label}: compute missing embeddings'
        count_str = f'{batch_size} of {total}' if batch_size < total else f'{total}'

        def task(ctx):
            ctx.update(0.0, f'Computing embeddings for {batch_size} of {total} files...')
            try:
                engine.process_files(
                    batch,
                    callback=lambda i, n: ctx.update(
                        i / n, f'Computing embeddings for {i+1}/{n} of {total}...'
                    ),
                    media_folder=file_manager.media_directory,
                    generate_embs_if_not_in_cache=True,
                )
            except Exception as e:
                print(f'[{label}: embedding] Failed: {e}')

        return app.task_manager.submit(f'{base_name} ({count_str})', task)

    return _check_and_submit_embedding

def make_scheduled_rating_check(app, label, file_manager: FileManager, evaluator, cfg, cfg_key, update_model_ratings_fn):
    """Return a callable for ``Scheduler`` that submits rating tasks.

    Args:
        app:                      Flask app (must have ``app.task_manager``).
        label:                    Human label, e.g. ``"Images"``.
        file_manager:             FileManager instance for this module.
        evaluator:                Evaluator instance (must have ``.hash``).
        cfg:                      OmegaConf config object.
        cfg_key:                  Config prefix, e.g. ``"images"``.
        update_model_ratings_fn:  The module's ``update_model_ratings()`` callable.
    """
    def _check_and_submit_rating():
        if evaluator.hash is None:
            return
        candidates = file_manager.get_unrated_files(evaluator.hash)
        total = len(candidates)
        base_name = f'{label}: rate unrated files'
        batch_size = OmegaConf.select(cfg, f'{cfg_key}.rating_update_batch_size', default=None)
        batch_size = min(batch_size, total) if batch_size else total
        count_str = f"{batch_size} of {total}" if batch_size < total else f"{total}"

        def task(ctx):
            files_list = candidates[:batch_size]
            ctx.update(0.0, f'Rating {len(files_list)} of {total} files...')
            update_model_ratings_fn(files_list, ctx=ctx)

        return app.task_manager.submit(f'{base_name} ({count_str})', task)

    return _check_and_submit_rating
