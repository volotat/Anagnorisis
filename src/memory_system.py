"""
memory_system.py — Universal durable memory for rated files.

When a user rates a file (anywhere in the app), a rich "memory" .md file is
written to ``project_config/memory/<YYYY-MM-DD>/<soft_hash>.md`` capturing
everything we know about the file at that moment:

  * tags + fingerprint from the media type's embedding model,
  * a natural-language description from the OmniDescriptor,
  * internal metadata (TinyTag / PIL size, etc.),
  * the contents of the ``{file}.meta`` sidecar if present.

The rating is written as line 1 and *only* line 1, so that training can strip
it and embed the rest: the evaluator must never see the score inside the text
it is learning to predict. Keep it that way — moving the rating anywhere below
line 1 would silently teach the model to read the answer instead of the file.

These memory files are the single source of truth for training the evaluator:
even if the original file is later moved, renamed, or disappears (especially
from a remote server), the description is preserved so the model can still
learn what kind of content was rated how.

Which handling a file gets follows from its media type (see
anagnorisis_core/media_types.py), so this stays decoupled from the modules: a rated
file is described the same way whether or not a module owns its content kind.
"""

import os
import datetime
import threading

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.media import description, models
from anagnorisis_core.media.media_types import get_registry
from anagnorisis_core.models.descriptor import OmniDescriptor
from anagnorisis_core.models.embedder import get_omni_embedder
from src.app_factory.event_manager import EventManager


class MemorySystem:
    """Writes durable memory .md files for rated files.

    Instantiated once at app creation and held on ``app.memory_system``.
    ``save_memory`` is the public entry point — always enqueues a background
    task (via the shared task manager) so the rating socket returns immediately
    and so GPU-heavy work is serialized against other background tasks.
    """

    def __init__(self, cfg, cache_path, memory_path, models_folder,
                 personal_models_path, task_manager):
        self.cfg = cfg
        self.cache_path = cache_path
        self.memory_path = memory_path
        self.models_folder = models_folder
        self.personal_models_path = personal_models_path
        self._task_manager = task_manager

        os.makedirs(self.memory_path, exist_ok=True)

        # Models are NOT loaded here — this constructor must return instantly so
        # the Flask server can start and show the loading page immediately. Heavy
        # embedding/omni models are loaded lazily on first use (inside a background
        # task) via _ensure_initialized().
        self._initialized = False
        self._init_lock = threading.Lock()
        self._omni = None
        self.types = get_registry(cfg)

    def _ensure_initialized(self):
        """Lazily load all embedding/omni models on first use (thread-safe).

        Called from inside background tasks (save_memory / migration), so the
        heavy GPU model loading never blocks app startup or the Flask server.
        """
        if self._initialized:
            return
        with self._init_lock:
            if self._initialized:
                return
            cfg = self.cfg
            models_folder = self.models_folder

            print("[MemorySystem] Loading embedding/omni models (first use)...")

            # The shared content embedder — loaded here so a rated file always
            # gets a proxy section, even if no module has touched that media
            # type this session.
            get_omni_embedder(cfg).initiate(models_folder)

            # --- Omni descriptor ---
            self._omni = OmniDescriptor(cfg)
            self._omni.initiate(models_folder)
            self._omni.unload()  # free VRAM until first use

            self._initialized = True
            print("[MemorySystem] Models loaded and ready.")

    # ------------------------------------------------------------------
    # Memory text builder
    # ------------------------------------------------------------------

    def build_memory_text(self, file_path, soft_hash, rating):
        """Assemble the memory .md content: the rating, then the description.

        The body is the *same* text metadata search embeds and the UI shows as
        "full search description" — one call to the shared builder, not a second
        arrangement of the same sections. That equality is the point: training
        strips line 1 and embeds the rest, so the evaluator learns on exactly the
        text it will later score. When the two drifted apart, the model was being
        trained on one shape and asked to judge another.

        Nothing else goes in the header. The soft hash is the filename and the
        capture date is the containing folder, so writing them into the body
        would only break that equality to repeat what the path already says.

        Only called from ``save_memory``, which is triggered by
        ``emit_set_file_rating`` — an explicit user action. Reading a remote file
        here is therefore intentional, unlike in any background pass.
        """
        # Lazily load embedding/omni models on first use (inside a background task,
        # so this never blocks app startup).
        self._ensure_initialized()

        media_type = self.types.for_file(file_path)
        body = description.build_description(
            file_path,
            cfg=self.cfg,
            media_type=media_type,
            describe=lambda fp: self._describe_now(fp, media_type),
        )
        return f"Rating: {rating}\n" + body

    def _describe_now(self, file_path, media_type):
        """Describe the file with the omni model, releasing VRAM afterwards.

        Unlike the background passes this generates on demand — the user just
        rated this file, so it is worth the seconds — and unlike them it is
        allowed to read a remote file.
        """
        method_name = models.describe_method_for(media_type.name if media_type else None)
        if method_name is None:
            return ''
        try:
            method = getattr(self._omni, method_name)
            if method_name == 'describe_text':
                content = description.read_text_content(file_path)
                text = method(content) if content is not None else ''
            else:
                text = method(file_path)
            return (text or '').strip()
        except Exception as exc:
            print(f"[MemorySystem] Omni description failed for {file_path}: {exc}")
            return ''
        finally:
            try:
                self._omni.unload()  # free VRAM as soon as we are done
            except Exception:
                pass

    def _memory_file_path(self, soft_hash, when=None):
        date_folder = (when or datetime.date.today()).isoformat()
        folder = os.path.join(self.memory_path, date_folder)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, f"{soft_hash}.md")

    def _memory_file_exists(self, soft_hash):
        """True if any dated folder already holds <soft_hash>.md."""
        if not os.path.isdir(self.memory_path):
            return False
        target = f"{soft_hash}.md"
        for entry in os.scandir(self.memory_path):
            if entry.is_dir() and os.path.exists(os.path.join(entry.path, target)):
                return True
        return False

    def _write_atomic(self, file_path, text):
        """Write text to file_path atomically (temp file + rename)."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        tmp = file_path + ".tmp"
        with open(tmp, 'w', encoding='utf-8') as f:
            f.write(text)
        os.replace(tmp, file_path)

    # ------------------------------------------------------------------
    # Public entry point (always a background task)
    # ------------------------------------------------------------------

    def save_memory(self, file_path, rating, soft_hash=None):
        """Enqueue a background task that writes (or refreshes) the memory .md
        for ``file_path``. Returns immediately so the caller is never blocked.

        The ``rating`` is stored as the first line of the .md so training can
        parse it cheaply without any DB join, then strip it before embedding so
        the evaluator never sees the score in the text it predicts.

        If the file has disappeared, we still write a minimal memory record
        (rating + soft hash + last-known path) so it is not lost for training.
        """
        if soft_hash is None:
            try:
                soft_hash = EventManager.get_file_soft_hash(file_path)
            except Exception as exc:
                print(f"[MemorySystem] Could not compute soft hash for {file_path}: {exc}")
                return

        def _task(ctx):
            ctx.check()
            ctx.update(0.0, f'Building memory for {os.path.basename(file_path)}')
            try:
                text = self.build_memory_text(file_path, soft_hash, rating)
            except FileNotFoundError:
                # File gone — record what we still know (durable, no model deps).
                text = self._minimal_memory_text(file_path, soft_hash, rating)
            except Exception as exc:
                print(f"[MemorySystem] build_memory_text failed for {file_path}: {exc}")
                text = self._minimal_memory_text(file_path, soft_hash, rating, note=f"build error: {exc}")
            self._write_atomic(self._memory_file_path(soft_hash), text)
            ctx.update(1.0, f'Memory written for {os.path.basename(file_path)}')

        self._task_manager.submit(
            f'Write memory: {soft_hash[:8]}', _task
        )

    def _minimal_memory_text(self, file_path, soft_hash, rating, note=None):
        file_name = os.path.basename(file_path)
        lines = [
            f"Rating: {rating}",  # line 1
            f"Soft Hash: {soft_hash}",
            f"Hash Algorithm: {EventManager.soft_hash_algorithm}",
            f"File Path: {file_path}",
            f"File Name: {file_name}",
            f"Captured At: {datetime.datetime.now().isoformat()}",
            "",
            "# Note:",
            note or "Original file was not available; captured metadata is limited.",
            "",
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # One-time DB -> memory migration
    # ------------------------------------------------------------------

    def migrate_db_ratings_to_memory(self, ctx):
        """Write a memory .md for every FilesLibrary row with a user_rating that
        does not already have one. Idempotent: skips hashes that already have a
        memory file.
        """
        import src.db_models as db_models

        try:
            rows = db_models.FilesLibrary.query.filter(
                db_models.FilesLibrary.user_rating.isnot(None)
            ).all()
        except Exception as exc:
            print(f"[MemorySystem] Migration DB query failed: {exc}")
            return

        total = len(rows)
        if total == 0:
            print("[MemorySystem] No rated files to migrate.")
            return

        print(f"[MemorySystem] Migrating up to {total} rated files to memory.")
        migrated = 0
        for i, row in enumerate(rows):
            ctx.check()
            ctx.update((i + 1) / total, f'Migrating {i + 1}/{total}')

            soft_hash = row.hash
            file_path = row.file_path
            if not soft_hash or not file_path:
                continue
            if self._memory_file_exists(soft_hash):
                continue  # already captured

            try:
                text = self.build_memory_text(file_path, soft_hash, row.user_rating)
            except FileNotFoundError:
                text = self._minimal_memory_text(file_path, soft_hash, row.user_rating)
            except Exception as exc:
                print(f"[MemorySystem] Migration build failed for {file_path}: {exc}")
                text = self._minimal_memory_text(file_path, soft_hash, row.user_rating, note=f"build error: {exc}")

            # Date the memory folder by when the user actually rated the file,
            # not by today, so the chronological memory history reflects reality.
            rating_date = row.user_rating_date.date() if row.user_rating_date else None
            self._write_atomic(self._memory_file_path(soft_hash, when=rating_date), text)
            migrated += 1

        print(f"[MemorySystem] Migration complete — {migrated} memory files written.")
