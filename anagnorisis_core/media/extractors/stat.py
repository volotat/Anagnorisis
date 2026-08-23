"""Filesystem-level metadata — the fallback for content we cannot parse."""

import os

import fs

import anagnorisis_core.storage.virtual_file_system as vfs


def read_stat_metadata(file_path: str) -> dict:
    """
    Extracts basic metadata from any file via VFS-aware stat calls.

    Used by media types with no embedded metadata block (unlike images' EXIF or
    audio's ID3 tags), where filesystem-level information is all there is:

      - file_size        : integer byte count
      - modification_time: ISO 8601 datetime of last modification
      - creation_time    : ISO 8601 datetime of creation (if FS supports it)
      - access_time      : ISO 8601 datetime of last access (if FS supports it)
      - extension        : lowercase file extension (txt, md, html, …)

    File content is NEVER read, so this stays O(1) regardless of file size —
    important since text libraries often contain multi-GB logs.

    All values are stringified so that ``generate_full_description`` can drop
    them straight into the embedding payload.
    """
    metadata = {}
    try:
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)

        with fs.open_fs(base_url) as my_fs:
            info = my_fs.getinfo(path_in_fs, namespaces=['details'])

            # File size — cheap, useful as a "document size" hint.
            if getattr(info, 'size', None) is not None:
                metadata['file_size'] = f"{info.size} bytes"

            # Timestamps — stringified via isoformat() so they fit
            # directly into the embedding text payload.
            for attr, key in (
                ('modified', 'modification_time'),
                ('created',  'creation_time'),
                ('accessed', 'access_time'),
            ):
                dt = getattr(info, attr, None)
                if dt is None:
                    continue
                metadata[key] = (
                    dt.isoformat() if hasattr(dt, 'isoformat') else str(dt)
                )

            # Extension — the cheapest possible computation, and it tells the
            # embedding "this is markdown" / "this is a PDF" before it has seen
            # a single line of content.
            ext = os.path.splitext(path_in_fs)[1].lstrip('.').lower()
            if ext:
                metadata['extension'] = ext

    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")

    return metadata
