"""Where a generated description is put.

Two answers, and they are not variations on a theme — they are different kinds of
thing:

* **The cache** is derived and disposable. It belongs to one machine, is keyed by
  which model produced it, and can be thrown away and rebuilt. This is what the
  application uses.
* **A `.meta` sidecar** is a document. It lives next to the file, travels with it
  when the folder is copied, can be opened and edited by the person hosting it,
  and is the only thing a peer ever sees. This is what the data server writes.

Both are handed the same raw description and decide what to persist. The cache
keeps the model's sentences alone; the sidecar assembles the full public
description around them, because that file has to stand on its own.
"""
import datetime
import os
from typing import Optional, Protocol

from anagnorisis_core.media import description
from anagnorisis_core.storage import virtual_file_system as vfs


class Sink(Protocol):
    """Somewhere a description can be stored."""

    def is_done(self, file_path: str) -> bool:
        """True if this file needs no work — checked before spending any GPU."""

    def write(self, file_path: str, auto_description: str) -> bool:
        """Persist the description. False means nothing was written."""


class CacheSink:
    """Stores the description in the two-level cache, keyed by model.

    This is the application's sink: descriptions are an expensive derived
    artifact, and the cache is where derived artifacts live. Nothing is written
    into the user's media folders.
    """

    def __init__(self, metadata_search):
        # Reuses MetadataSearch's cache and key formula rather than rebuilding
        # them, so a description written here is one the app can find.
        self._search = metadata_search

    def is_done(self, file_path: str) -> bool:
        key = self._search.make_description_cache_key(file_path)
        return self._search._fast_cache.get(key) is not None

    def write(self, file_path: str, auto_description: str) -> bool:
        if not auto_description:
            return False
        key = self._search.make_description_cache_key(file_path)
        self._search._fast_cache.set(key, auto_description)
        return True


class MetaSink:
    """Writes `<file>.meta` next to the file, and never overwrites one.

    The rules all follow from the sidecar belonging to the host rather than to
    us:

    * **Present means done.** Whatever is in there is what they want published —
      written by a model, by a cloud service, or typed by hand. We cannot tell
      and should not care.
    * **Regeneration is deletion.** To get a better description, delete the file
      and run again. That needs no staleness comparison, no force flag, and no
      way to destroy someone's writing by accident.

    The description is deliberately *generic*: no path, because that would
    publish the server's directory layout, and no sidecar section, because this
    is the sidecar.
    """

    def __init__(self, cfg, stamp: bool = True):
        self._cfg = cfg
        self._stamp = stamp

    @staticmethod
    def sidecar_path(file_path: str) -> str:
        """The sidecar's location on disk.

        Files are addressed as VFS URLs (``osfs:///…``) throughout, but writing
        one means touching the real filesystem, so the scheme is resolved away
        here. Sidecars are only ever written beside local files, so this never
        copies anything.
        """
        local = vfs.resolve_to_local_path(file_path)[0] if '://' in file_path else file_path
        return local + '.meta'

    def is_done(self, file_path: str) -> bool:
        return os.path.exists(self.sidecar_path(file_path))

    def _provenance(self) -> str:
        """One compact line saying what produced this.

        One line because a reader of this file takes only its first 300 lines and
        30 000 characters, and that budget belongs to the description. It is
        information, not a control signal: nothing reads it back to decide
        whether to regenerate.
        """
        omni = getattr(self._cfg, 'omni', None)
        embedder = getattr(self._cfg, 'embedder', None)
        today = datetime.date.today().isoformat()
        return (
            "Anagnorisis-Generated: "
            f"descriptor={getattr(omni, 'model_name', 'unknown')} "
            f"embedder={getattr(embedder, 'model_name', 'unknown')} "
            f"at={today}\n\n"
        )

    def write(self, file_path: str, auto_description: str) -> bool:
        target = self.sidecar_path(file_path)   # resolved to a real path
        if os.path.exists(target):
            return False            # never overwrite; see the class docstring

        text = description.build_description(
            file_path,
            cfg=self._cfg,
            describe=lambda _fp: auto_description,
            include_path=False,          # do not publish the server's layout
            include_meta_sidecar=False,  # we are writing it
        )
        if self._stamp:
            text = self._provenance() + text

        # Write via a temporary file in the same directory then rename, so a
        # crash or a full disk cannot leave a half-written description that
        # would then be treated as finished forever.
        tmp = target + '.partial'
        try:
            with open(tmp, 'w', encoding='utf-8') as fh:
                fh.write(text)
            os.replace(tmp, target)
        except OSError as exc:
            print(f"[MetaSink] Could not write {target}: {exc}")
            try:
                os.unlink(tmp)
            except OSError:
                pass
            return False
        return True


def build_sink(name: str, *, cfg, metadata_search=None) -> Sink:
    """Resolve a sink by name, for the command line."""
    if name == 'meta':
        return MetaSink(cfg)
    if name == 'cache':
        if metadata_search is None:
            raise ValueError("the cache sink needs a MetadataSearch instance")
        return CacheSink(metadata_search)
    raise ValueError(f"unknown sink {name!r}; expected 'cache' or 'meta'")
