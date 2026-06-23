"""
module_helpers.py — Shared helpers that eliminate boilerplate across modules.

Each function either registers socket handlers or returns a callable that
can be passed to ``Scheduler``.
"""

import os
from omegaconf import OmegaConf


# ---------------------------------------------------------------------------
# .meta file handlers + full description handler
# ---------------------------------------------------------------------------

def register_meta_handlers(socketio, module_name, media_directory_ref, metadata_search_engine):
    """Register get/save .meta and get_full_description socket handlers.

    Args:
        socketio:               Flask-SocketIO instance.
        module_name:            e.g. ``"images"`` — used to build event names.
        media_directory_ref:    A callable returning the current media directory
                                path (to support ``nonlocal`` mutation).
        metadata_search_engine: MetadataSearch instance.
    """
    prefix = f'emit_{module_name}_page'

    @socketio.on(f'{prefix}_get_external_metadata_file_content')
    def get_external_metadata_file_content(file_path):
        media_directory = media_directory_ref()
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"
        content = ""
        try:
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            print(f"Read external metadata for {file_path}")
        except Exception as e:
            print(f"Error reading external metadata for {file_path}: {e}")
        return {"content": content, "file_path": file_path}

    @socketio.on(f'{prefix}_save_external_metadata_file_content')
    def save_external_metadata_file_content(data):
        media_directory = media_directory_ref()
        file_path = data['file_path']
        metadata_content = data['metadata_content']
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"
        try:
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            print(f"Saved metadata for {file_path}")
        except Exception as e:
            print(f"Error saving metadata for {file_path}: {e}")

    @socketio.on(f'{prefix}_get_full_metadata_description')
    def get_full_metadata_description(file_path):
        media_directory = media_directory_ref()
        full_path = os.path.join(media_directory, file_path)
        content = metadata_search_engine.generate_full_description(full_path, media_directory, generate_desc_if_not_in_cache=False)
        return {"content": content, "file_path": file_path}


# ---------------------------------------------------------------------------
# Scheduled task factories
# ---------------------------------------------------------------------------

from src.file_manager import FileManager

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


def make_scheduled_description_check(app, label, file_manager: FileManager, metadata_search_engine, cfg, cfg_key):
    """Return a callable for ``Scheduler`` that submits description tasks.

    Args:
        app:                      Flask app (must have ``app.task_manager``).
        label:                    Human label, e.g. ``"Images"``.
        file_manager:             FileManager instance for this module.
        metadata_search_engine:   MetadataSearch instance.
        cfg:                      OmegaConf config object.
        cfg_key:                  Config prefix, e.g. ``"images"``.
    """
    def _check_and_submit_description():
        all_files = file_manager.list_all_files()
        candidates = metadata_search_engine.get_undescribed_files(all_files)
        if candidates is None:
            candidates = all_files
        base_name = f'{label}: describe undescribed files'
        batch_size = OmegaConf.select(cfg, f'{cfg_key}.description_update_batch_size', default=100)
        batch_size = min(batch_size, len(candidates))
        batch = candidates[:batch_size]
        n_total = len(candidates)
        count_label = f'{batch_size} of {n_total}' if batch_size < n_total else str(n_total)

        def task(ctx):
            try:
                for i, fp in enumerate(batch):
                    ctx.check()
                    ctx.update(i / len(batch), f'Describing file {i + 1}/{len(batch)} of {n_total}...')
                    try:
                        metadata_search_engine._get_auto_description(fp, generate_desc_if_not_in_cache=True)
                    except Exception as e:
                        print(f'[{label}: describe] Failed for {fp}: {e}')
            finally:
                metadata_search_engine.omni_descriptor.unload()

        return app.task_manager.submit(f'{base_name} ({count_label})', task)

    return _check_and_submit_description
