"""metadata_indexer.py — the background passes that keep the metadata caches warm.

The describing and embedding themselves live in anagnorisis_core; what stays
here is the scheduling: which files to pick, how big a batch, and running it on
the task queue so it can be paused from the Task Manager. That is the part that
needs the Flask app, which is why it did not move.

Metadata search only ever reads from cache on the search path, so something has
to fill it. That is these two schedulers, and there are two rather than one
because the passes have nothing in common but their input:

  descriptions  OmniDescriptor, ~seconds per file, local files only (it reads
                content), small batches, coarse interval.
  embeddings    the text embedder, ~milliseconds per file, local *and* remote
                (a remote description is filename + path + .meta), large
                batches, fine interval.

Running them as one task would thrash VRAM between two large models; running
them as two lets the fast pass keep up while the slow one grinds, and gives
each its own pause button in the Task Manager.

They are also the only schedulers in the app that are not per-module: one pass
over every configured server covers every media type at once, which is both
less work than four module passes and the reason a file needs no owning module
to be searchable.
"""

import random

from omegaconf import OmegaConf

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.storage.file_walker import get_file_walker
from anagnorisis_core.search.metadata_search import get_metadata_search
from src.scheduler import Scheduler


def _release_embedder(cfg):
    """Free the embedder's VRAM so the descriptor can have the card.

    Imported lazily: this module is loaded during startup, and the embedder
    pulls in torch. Failing here is not worth aborting a description pass —
    the worst case is that the descriptor refuses to load and retries later.
    """
    try:
        from anagnorisis_core.models.embedder import get_omni_embedder
        get_omni_embedder(cfg).unload()
    except Exception as exc:
        print(f'[Metadata: describe] Could not release the embedder: {exc}')


class MetadataIndexer:
    """Owns the two universal metadata schedulers."""

    def __init__(self, app, cfg):
        self.app = app
        self.cfg = cfg
        self.metadata_search = get_metadata_search(cfg)
        self.walker = get_file_walker(app.user_cfg.servers, cfg.main.cache_path)
        self.extensions = set(self.metadata_search.types.all_extensions())

        self._register_schedulers()

    # ------------------------------------------------------------------

    def _register_schedulers(self):
        """Registers the background metadata passes."""
        app = self.app
        cfg = self.cfg

        # Generate the OmniDescriptor descriptions that everything else builds on.
        description_interval = OmegaConf.select(
            cfg, 'metadata_search.auto_description_interval_minutes', default=30
        )
        Scheduler(
            app,
            interval_minutes=description_interval,
            fn=self._check_and_submit_descriptions,
            name='Metadata: describe undescribed files',
        )

        # Embed the assembled descriptions so metadata search can find them.
        embedding_interval = OmegaConf.select(
            cfg, 'metadata_search.embedding_interval_minutes', default=10
        )
        Scheduler(
            app,
            interval_minutes=embedding_interval,
            fn=self._check_and_submit_embeddings,
            name='Metadata: compute missing metadata embeddings',
        )

    # ------------------------------------------------------------------
    # Description pass
    # ------------------------------------------------------------------

    def _check_and_submit_descriptions(self):
        """Submit a batch of files for OmniDescriptor description.

        Local files only: describing a file means reading its content, and
        remote content is never fetched by a background pass. Remote files are
        still searchable through their filename, path and .meta sidecar, and
        get a full description when the user rates them (MemorySystem).
        """
        local_files = [fp for fp in self._all_files() if vfs.is_local_url(fp)]
        if not local_files:
            return

        candidates = self.metadata_search.get_undescribed_files(local_files)
        if candidates is None:
            # The descriptor has never been loaded, so its model hash — and with
            # it every description cache key — is unknown. Load it on the task
            # queue, where GPU work is serialized, then release the VRAM again:
            # the hash is persisted, so the next cycle can probe without it.
            return self.app.task_manager.submit(
                'Metadata: load descriptor', self._load_descriptor_task
            )
        if not candidates:
            return

        batch, count_label = self._take_batch(candidates, 'auto_description_batch_size', 10)

        def task(ctx):
            # The descriptor needs almost the whole card, so it cannot start
            # while the embedder is still resident — it would refuse to load and
            # this batch would be skipped. Both restart transparently on their
            # next use, so releasing the embedder here costs only its reload.
            _release_embedder(self.cfg)
            try:
                for i, fp in enumerate(batch):
                    ctx.check()
                    ctx.update(i / len(batch), f'Describing file {i + 1}/{count_label}...')
                    try:
                        self.metadata_search._get_auto_description(
                            fp, generate_desc_if_not_in_cache=True
                        )
                    except Exception as e:
                        print(f'[Metadata: describe] Failed for {fp}: {e}')
            finally:
                self.metadata_search.omni_descriptor.unload()

        return self.app.task_manager.submit(
            f'Metadata: describe undescribed files ({count_label})', task
        )

    def _load_descriptor_task(self, ctx):
        ctx.update(0.0, 'Loading the descriptor to learn its model hash...')
        _release_embedder(self.cfg)
        try:
            self.metadata_search.load_descriptor()
        finally:
            self.metadata_search.omni_descriptor.unload()

    # ------------------------------------------------------------------
    # Embedding pass
    # ------------------------------------------------------------------

    def _check_and_submit_embeddings(self):
        """Submit a batch of files for metadata embedding (local and remote)."""
        all_files = self._all_files()
        if not all_files:
            return

        candidates = self.metadata_search.get_unembedded_files(all_files)
        if not candidates:
            return

        batch, count_label = self._take_batch(candidates, 'embedding_batch_size', 500)

        def task(ctx):
            ctx.update(0.0, f'Computing metadata embeddings for {count_label} files...')
            # process_files reports one status line per file, which is also the
            # natural place to honour a pause or cancel.
            done = {'count': 0}

            def report(message):
                ctx.check()
                done['count'] += 1
                ctx.update(done['count'] / len(batch), message)

            self.metadata_search.process_files(
                batch, callback=report, generate_embs_if_not_in_cache=True
            )

        return self.app.task_manager.submit(
            f'Metadata: compute missing metadata embeddings ({count_label})', task
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _all_files(self) -> list[str]:
        """Every file of every known media type, on every configured server."""
        return self.walker.walk("/", self.extensions)

    def _take_batch(self, candidates: list[str], batch_size_key: str, default: int):
        """Cap a candidate list per cycle so one pass cannot monopolise the queue.

        The batch is sampled at random rather than taken from the front. The
        walker returns files in depth-first directory order, so a prefix would
        index one folder to completion before touching the next — on a large
        library the last folders would wait days for their first result. A
        random sample spreads early coverage across the whole collection.

        This changes only the order: every cycle re-probes what is still
        missing, so the backlog shrinks the same way regardless.
        """
        total = len(candidates)
        configured = OmegaConf.select(
            self.cfg, f'metadata_search.{batch_size_key}', default=default
        )
        size = min(configured, total) if configured else total
        label = f'{size} of {total}' if size < total else str(total)
        return random.sample(candidates, size), label
