"""
memory_system.py — Universal durable memory for rated files.

When a user rates a file (anywhere in the app), a rich "memory" .md file is
written to ``project_config/memory/<YYYY-MM-DD>/<soft_hash>.md`` capturing
everything we know about the file at that moment:

  * tags + fingerprint from the per-modality embedding model (CLAP / SigLIP),
  * a natural-language description from the OmniDescriptor,
  * internal metadata (TinyTag / PIL size, etc.),
  * the contents of the ``{file}.meta`` sidecar if present.

The rating itself is **never** written into the .md text — it lives in the
``FilesLibrary`` table, keyed by soft hash, so the universal evaluator cannot
"cheat" by reading a score from the text it is learning to predict.

These memory files are the single source of truth for training the evaluator:
even if the original file is later moved, renamed, or disappears (especially
from a remote server), the description is preserved so the model can still
learn what kind of content was rated how.

The memory writer owns the embedding / omni models directly (AudioEmbedder,
ImageEmbedder, OmniDescriptor) rather than reaching into per-module engines,
so it stays decoupled from the individual modules.
"""

import os
import datetime
import threading
import numpy as np
import torch
import fs

from omegaconf import OmegaConf

import src.virtual_file_system as vfs
from src.caching import TwoLevelCache
from src.embedding_proxy import EmbeddingProxyGenerator
from src.audio_embedder import AudioEmbedder, get_shared_audio_embedder
from src.image_embedder import ImageEmbedder
from src.omni_descriptor import OmniDescriptor
from src.app_factory.event_manager import EventManager


# How much of a .meta sidecar to read into a memory file.
_MAX_META_LINES = 300
_MAX_META_CHARS = 30_000
# Internal-metadata string-value length cap (drops base64 cover art, etc.).
_MAX_META_VALUE_LEN = 1000


class EmbedderAdapter:
    """Thin adapter that lets ``EmbeddingProxyGenerator`` use a bare
    ``AudioEmbedder`` / ``ImageEmbedder`` instead of a full ``*Search`` engine.

    ``EmbeddingProxyGenerator`` calls ``engine.process_text(tag)``,
    ``engine.get_file_hash(path)``, ``engine._get_model_hash_postfix()`` and
    reads ``engine._fast_cache`` / ``engine.model_hash``. The bare embedders
    only expose ``embed_text`` / ``embed_audio`` / ``embed_image`` + ``model_hash``,
    so we provide the missing surface here.
    """

    def __init__(self, embedder, cache_path, cache_prefix, model_hash_postfix, hash_algorithm="md5:v2"):
        self._embedder = embedder
        self.model_hash = embedder.model_hash
        self._postfix = model_hash_postfix
        self.hash_algorithm = hash_algorithm
        self._fast_cache = TwoLevelCache(
            cache_dir=os.path.join(cache_path, 'memory_embedder', cache_prefix),
            name=cache_prefix,
        )

    def process_text(self, text):
        """Embed a tag string and return a 1-D torch tensor (D,)."""
        arr = self._embedder.embed_text(text)  # np.ndarray (D,)
        return torch.from_numpy(np.asarray(arr, dtype=np.float32)).ravel()

    def _get_model_hash_postfix(self):
        return self._postfix

    def get_file_hash(self, file_path):
        """md5 over the file bytes via VFS, cached on (path, size, mtime).

        Mirrors ``BaseSearchEngine.get_file_hash`` but without requiring a
        ``*Search`` engine instance.
        """
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
        with fs.open_fs(base_url) as my_fs:
            info = my_fs.getinfo(path_in_fs, namespaces=['details'])
            size = info.size
            modified_sec = info.get('details', 'modified')
            mtime_ns = int(modified_sec * 1e9) if modified_sec is not None else 0

            cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{self.hash_algorithm}"
            cached = self._fast_cache.get(cache_key)
            if cached is not None:
                return cached

            file_hash = vfs.calculate_file_hash(my_fs, path_in_fs)
            self._fast_cache.set(cache_key, file_hash)
            return file_hash


class MemorySystem:
    """Owns the embedding / omni models and writes durable memory .md files.

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
        self._audio_embedder = None
        self._audio_adapter = None
        self._audio_proxy = None
        self._image_embedder = None
        self._image_adapter = None
        self._image_proxy = None
        self._omni = None

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
            cache_path = self.cache_path
            models_folder = self.models_folder

            print("[MemorySystem] Loading embedding/omni models (first use)...")

            # --- Audio (CLAP) ---
            self._audio_embedder = get_shared_audio_embedder(cfg, models_folder)
            self._audio_adapter = EmbedderAdapter(
                self._audio_embedder, cache_path, 'audio', 'v1.2'
            )
            self._audio_proxy = EmbeddingProxyGenerator(
                engine=self._audio_adapter,
                tag_list=list(OmegaConf.select(cfg, 'music.embedding_tags', default=[]) or []),
                threshold=OmegaConf.select(cfg, 'music.embedding_tags_threshold', default=None),
                cache_path=cache_path,
                model_name='CLAP',
            )

            # --- Image (SigLIP) ---
            self._image_embedder = ImageEmbedder(cfg)
            self._image_embedder.initiate(models_folder)
            self._image_adapter = EmbedderAdapter(
                self._image_embedder, cache_path, 'image', 'v1.1'
            )
            self._image_proxy = EmbeddingProxyGenerator(
                engine=self._image_adapter,
                tag_list=list(OmegaConf.select(cfg, 'images.embedding_tags', default=[]) or []),
                threshold=OmegaConf.select(cfg, 'images.embedding_tags_threshold', default=None),
                cache_path=cache_path,
                model_name='SigLIP',
            )

            # --- Omni (MiniCPM-o) ---
            self._omni = OmniDescriptor(cfg)
            self._omni.initiate(models_folder)
            self._omni.unload()  # free VRAM until first use

            self._initialized = True
            print("[MemorySystem] Models loaded and ready.")

    # ------------------------------------------------------------------
    # Extension -> modality dispatch
    # ------------------------------------------------------------------

    def _modality_for_extension(self, ext):
        """Return 'audio' | 'image' | 'video' | 'text' | None for a file extension.

        Reads the per-modality ``media_formats`` config lists (the single source
        of truth, mirroring ``MetadataSearch._extension_to_describe_method``).
        """
        if not ext:
            return None
        cfg = self.cfg
        for modality, cfg_key in (('audio', 'music'), ('image', 'images'),
                                   ('video', 'videos'), ('text', 'text')):
            try:
                fmts = set(OmegaConf.select(cfg, f'{cfg_key}.media_formats', default=[]) or [])
            except Exception:
                fmts = set()
            if ext in fmts:
                return modality
        return None

    # ------------------------------------------------------------------
    # Memory text builder
    # ------------------------------------------------------------------

    def build_memory_text(self, file_path, soft_hash, rating):
        """Assemble the full memory .md content for a file.

        The user rating is written as the very first line so it is trivial to
        parse (and strip before embedding — see universal_train._parse_memory_file).
        Each section is wrapped defensively so a failed model call or a missing
        file does not lose the rest of the description.
        """
        # Lazily load embedding/omni models on first use (inside a background task,
        # so this never blocks app startup).
        self._ensure_initialized()

        file_name = os.path.basename(file_path)
        ext = os.path.splitext(file_name)[1].lower()
        modality = self._modality_for_extension(ext)

        parts = [
            f"Rating: {rating}",  # line 1 — parsed & stripped before embedding
            f"Soft Hash: {soft_hash}",
            f"Hash Algorithm: {EventManager.soft_hash_algorithm}",
            f"File Path: {file_path}",
            f"File Name: {file_name}",
            f"Captured At: {datetime.datetime.now().isoformat()}",
            "",
        ]

        # 1. Embedding proxy (tags + fingerprint) — audio (CLAP) / image (SigLIP)
        parts.extend(self._collect_proxy_section(file_path, modality))

        # 2. OmniDescriptor natural-language description
        parts.extend(self._collect_omni_section(file_path, modality))

        # 3. Internal metadata (TinyTag / PIL size, etc.)
        parts.extend(self._collect_internal_metadata(file_path, modality))

        # 4. .meta sidecar
        parts.extend(self._collect_meta_section(file_path, file_name))

        return "\n".join(parts)

    def _collect_proxy_section(self, file_path, modality):
        if modality == 'audio':
            proxy = self._audio_proxy
        elif modality == 'image':
            proxy = self._image_proxy
        else:
            # video has no real embedder; text embeddings are chunked/incompatible.
            return []

        # Cache-aware path: proxy cache → engine cache → cold compute
        try:
            section = proxy.get_cached_proxy_text(file_path)
            if section and section.strip():
                return [section.strip(), ""]
        except Exception as exc:
            print(f"[MemorySystem] Proxy section failed for {file_path}: {exc}")
        return []

    def _adapter_for(self, modality):
        return self._audio_adapter if modality == 'audio' else self._image_adapter

    def _collect_omni_section(self, file_path, modality):
        if modality is None:
            return []
        method_name = {
            'audio': 'describe_audio_sampled',
            'image': 'describe_image',
            'video': 'describe_video_sampled',
            'text': 'describe_text',
        }.get(modality)
        if method_name is None:
            return []
        try:
            method = getattr(self._omni, method_name)
            if modality == 'text':
                content = self._read_text_content(file_path)
                description = method(content) if content is not None else ""
            else:
                description = method(file_path)
            self._omni.unload()  # free VRAM as soon as we are done
            if description and description.strip():
                return ["# Automatic description:", description.strip(), ""]
        except Exception as exc:
            print(f"[MemorySystem] Omni description failed for {file_path}: {exc}")
            try:
                self._omni.unload()
            except Exception:
                pass
        return []

    def _collect_internal_metadata(self, file_path, modality):
        if modality is None:
            return []
        try:
            metadata = self._get_internal_metadata(file_path, modality)
            if not metadata:
                return []
            lines = ["# Internal metadata:"]
            for key, value in metadata.items():
                if isinstance(value, str) and len(value) <= _MAX_META_VALUE_LEN and value.strip():
                    lines.append(f"{key}: {value}")
            lines.append("")
            return lines if len(lines) > 2 else []
        except Exception as exc:
            print(f"[MemorySystem] Internal metadata failed for {file_path}: {exc}")
        return []

    def _get_internal_metadata(self, file_path, modality):
        """Per-modality internal metadata getter (decoupled from *Search engines)."""
        if modality == 'audio':
            from modules.music.engine import get_audiofile_data
            return get_audiofile_data(file_path)
        if modality == 'image':
            from modules.images.engine import get_image_metadata
            return get_image_metadata(file_path)
        # video / text: no rich internal metadata extractor today.
        return {}

    def _collect_meta_section(self, file_path, file_name):
        try:
            meta_text = self._read_meta_sidecar(file_path)
            if meta_text and meta_text.strip():
                return [
                    f"# External metadata from '{file_name}.meta' file:",
                    meta_text,
                    "",
                ]
        except Exception as exc:
            print(f"[MemorySystem] .meta read failed for {file_path}: {exc}")
        return []

    # ------------------------------------------------------------------
    # VFS-aware file readers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_text_content(file_path, cap_chars=30_000):
        """Read a text file's content (for omni text description), capped."""
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
        with fs.open_fs(base_url) as my_fs:
            with my_fs.open(path_in_fs, 'rb') as f:
                return f.read(cap_chars).decode('utf-8', errors='ignore')

    @staticmethod
    def _read_meta_sidecar(file_path):
        """Read file_path + '.meta' via VFS, capped at _MAX_META_LINES/_CHARS."""
        meta_url = file_path + '.meta'
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(meta_url)
        with fs.open_fs(base_url) as my_fs:
            if not my_fs.exists(path_in_fs):
                return ""
            lines = []
            total = 0
            with my_fs.open(path_in_fs, 'rb') as f:
                for i, raw in enumerate(f):
                    if i >= _MAX_META_LINES:
                        break
                    line = raw.decode('utf-8', errors='ignore')
                    if total + len(line) > _MAX_META_CHARS:
                        break
                    lines.append(line)
                    total += len(line)
            return "".join(lines)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

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
