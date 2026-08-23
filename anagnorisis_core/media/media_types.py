"""media_types.py — the content-kind taxonomy shared across the whole app.

A *media type* is a kind of content — ``audio``, ``images``, ``videos``,
``text``, ``documents`` — defined by the file extensions belonging to it plus
the handling that kind needs.  Everything that used to be answered by asking a
module ("is this an image?", "which tags apply?", "how do I read its internal
metadata?") is answered here instead, which is what lets the metadata subsystem
run over every file on every server without knowing that modules exist.

Definitions live in ``media_types/media_types.yaml`` and are merged into the
config at startup, so they can be overridden from ``config.yaml``.  Tag
vocabularies live in ``media_types/tags/<type>.yaml``; they are loaded here
rather than into the config so the bulky lists never get pickled into the
embedding subprocesses.

Deliberately free of model names — which model handles which type is decided in
``anagnorisis_core/models.py``.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import Iterable, Optional

import yaml
from omegaconf import OmegaConf


def bundled_media_types_dir() -> str:
    """The taxonomy that ships with this package.

    Kept as ordinary files rather than importlib resources because a user is
    meant to be able to read and copy them, and because the loader below takes
    plain paths so an override directory works the same way.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'media_types')


@dataclass(frozen=True)
class MediaType:
    """One content kind: the extensions it covers and how to handle them."""

    name: str
    extensions: tuple[str, ...]
    # Name of the internal-metadata reader in anagnorisis_core/extractors/,
    # or None when the type has no readable internal metadata.
    metadata_extractor: Optional[str] = None
    # Cosine-similarity cutoff for zero-shot tags; None = take the top 15.
    tags_threshold: Optional[float] = None
    # Flat tag vocabulary, loaded from media_types/tags/<name>.yaml.
    tags: tuple[str, ...] = ()


class MediaTypeRegistry:
    """Immutable lookup over the configured media types."""

    def __init__(self, cfg):
        raw = OmegaConf.select(cfg, 'media_types', default=None)
        if not raw:
            raise ValueError(
                "No media types configured. Expected a 'media_types:' section, "
                "normally provided by media_types/media_types.yaml."
            )

        # The taxonomy ships with the package so a bare install works; a caller
        # that sets main.media_types_path overrides it, which is how a user's
        # edited tag vocabularies keep applying.
        tags_dir = os.path.join(
            OmegaConf.select(cfg, 'main.media_types_path', default=None)
            or bundled_media_types_dir(),
            'tags',
        )

        self._types: dict[str, MediaType] = {}
        self._by_extension: dict[str, MediaType] = {}

        for name, spec in OmegaConf.to_object(raw).items():
            extensions = _normalise_extensions(name, (spec or {}).get('extensions'))
            media_type = MediaType(
                name=name,
                extensions=extensions,
                metadata_extractor=(spec or {}).get('metadata_extractor'),
                tags_threshold=_optional_float((spec or {}).get('tags_threshold')),
                tags=_load_tags(os.path.join(tags_dir, f'{name}.yaml')),
            )
            self._types[name] = media_type
            for ext in extensions:
                clash = self._by_extension.get(ext)
                if clash is not None:
                    raise ValueError(
                        f"Extension '{ext}' is claimed by both media types "
                        f"'{clash.name}' and '{name}'. Every extension must belong "
                        f"to exactly one type."
                    )
                self._by_extension[ext] = media_type

    # ---- lookup -------------------------------------------------------

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._types)

    def get(self, name: str) -> Optional[MediaType]:
        return self._types.get(name)

    def for_file(self, file_path: str) -> Optional[MediaType]:
        """Return the media type owning *file_path*'s extension, or None."""
        return self._by_extension.get(os.path.splitext(file_path)[1].lower())

    def all_extensions(self) -> tuple[str, ...]:
        """Every known extension — the scope of a full metadata index."""
        return tuple(self._by_extension)

    def extensions_for(self, names: Iterable[str]) -> list[str]:
        """Extensions of the named types, in declaration order.

        Raises if a name is unknown, so a typo in a module's ``media_types``
        list fails at startup instead of silently indexing nothing.
        """
        extensions: list[str] = []
        for name in names:
            media_type = self._types.get(name)
            if media_type is None:
                raise ValueError(
                    f"Unknown media type '{name}'. Known types: "
                    f"{', '.join(self.names)}."
                )
            extensions.extend(media_type.extensions)
        return extensions


# ---- module-level accessor --------------------------------------------

_REGISTRY: Optional[MediaTypeRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry(cfg) -> MediaTypeRegistry:
    """Return the process-wide registry, building it on first use."""
    global _REGISTRY
    if _REGISTRY is None:
        with _REGISTRY_LOCK:
            if _REGISTRY is None:
                _REGISTRY = MediaTypeRegistry(cfg)
    return _REGISTRY


# ---- parsing helpers --------------------------------------------------

def _normalise_extensions(type_name: str, raw) -> tuple[str, ...]:
    """Lowercase extensions with a guaranteed leading dot, order preserved."""
    if not raw:
        raise ValueError(f"Media type '{type_name}' declares no extensions.")
    seen: dict[str, None] = {}
    for ext in raw:
        ext = str(ext).strip().lower()
        if not ext:
            continue
        seen.setdefault(ext if ext.startswith('.') else f'.{ext}', None)
    if not seen:
        raise ValueError(f"Media type '{type_name}' declares no usable extensions.")
    return tuple(seen)


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


def _load_tags(path: str) -> tuple[str, ...]:
    """Load and flatten a tag vocabulary file; missing file means no tags.

    Vocabularies are written as ``- [a, b, c]`` rows for compactness, so one
    level of nesting is flattened away here — every consumer sees a flat tuple.
    """
    if not os.path.exists(path):
        return ()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            rows = yaml.safe_load(f) or []
    except Exception as exc:
        print(f"[MediaTypes] Failed to read tag vocabulary {path}: {exc}")
        return ()

    tags: list[str] = []
    for row in rows:
        if isinstance(row, (list, tuple)):
            tags.extend(str(tag) for tag in row)
        else:
            tags.append(str(row))
    return tuple(tags)
