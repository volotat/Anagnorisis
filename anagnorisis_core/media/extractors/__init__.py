"""Internal-metadata readers — one per media type.

A reader takes a VFS path and returns a flat ``dict`` of metadata fields. Which
reader a file gets is decided by its media type's ``metadata_extractor`` field
in ``media_types/media_types.yaml``; a type that names no reader contributes no
internal metadata.

Readers live here rather than in the modules because both MetadataSearch and
MemorySystem need them without knowing whether the owning module is installed —
and because a module engine wanting file metadata should not have to be the one
that implements it. Module engines delegate to these.
"""

from anagnorisis_core.media.extractors.audio import read_audio_metadata
from anagnorisis_core.media.extractors.images import read_image_metadata
from anagnorisis_core.media.extractors.stat import read_stat_metadata

READERS = {
    'audio': read_audio_metadata,
    'images': read_image_metadata,
    'stat': read_stat_metadata,
}


def read(extractor_name, file_path: str) -> dict:
    """Return internal metadata for *file_path* via the named reader.

    Returns ``{}`` when the name is None (type has no internal metadata) or
    unknown, and swallows reader failures — a missing metadata section must
    never cost a file its description.
    """
    reader = READERS.get(extractor_name) if extractor_name else None
    if reader is None:
        if extractor_name:
            print(f"[Extractors] Unknown metadata_extractor '{extractor_name}'")
        return {}
    try:
        return reader(file_path) or {}
    except Exception as exc:
        print(f"[Extractors] '{extractor_name}' failed for {file_path}: {exc}")
        return {}


__all__ = ['read', 'READERS', 'read_audio_metadata', 'read_image_metadata',
           'read_stat_metadata']
