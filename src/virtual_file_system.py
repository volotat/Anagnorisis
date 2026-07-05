import urllib.parse
import fs.path
import hashlib

def resolve_base_and_path_from_url(fs_url):
    """
    Parses any PyFilesystem2 URL, opens the base filesystem connection,
    and returns a tuple of (base_url, path_in_fs).
    
    Correctly handles the structural differences between local (osfs://) 
    and remote (webdav://, ftp://, sftp://, etc.) URL parsing, without
    tripping on '#' or '?' characters inside directory or filenames.
    """
    if '://' not in fs_url:
        # Fallback for raw system paths
        return "osfs:///", fs.path.abspath(fs.path.normpath(fs_url))

    protocol, remainder = fs_url.split('://', 1)
    
    # 1. SPECIAL CASE: Local Filesystem (OSFS)
    if protocol == 'osfs':
        base_url = "osfs:///"
        # The remainder is the raw absolute path on disk (e.g., /mnt/media/...)
        # We preserve the entire string, preventing '#' from being treated as a fragment
        path_in_fs = fs.path.abspath(fs.path.normpath(remainder))
        return base_url, path_in_fs
    
    # 2. REMOTE CASE: (webdav://, ftp://, sftp://, etc.)
    # Split on the first single '/' to separate the network location (authority) from the path
    if '/' in remainder:
        authority, raw_path = remainder.split('/', 1)
        path_in_fs = '/' + raw_path
    else:
        authority = remainder
        path_in_fs = '/'
        
    # Reconstruct the base connection URL (ensuring a trailing slash)
    base_url = f"{protocol}://{authority}/"
    path_in_fs = fs.path.abspath(fs.path.normpath(path_in_fs))
    
    return base_url, path_in_fs

def calculate_file_hash(my_fs, path_in_fs, chunk_size=65536, hash_algorithm=hashlib.md5):
    """
    Reads a file chunk-by-chunk from any PyFilesystem2 target
    and returns its hash hex digest.
    """
    hasher = hash_algorithm()
    
    # Open in binary read mode
    with my_fs.open(path_in_fs, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
            
    return hasher.hexdigest()

def join_fs_url(base_url: str, relative_path: str) -> str:
    """
    Safely joins a PyFilesystem2 URL with a relative path.
    Preserves protocols, authorities, and trailing slashes (e.g., osfs:/// or
    webdav://host:port/).

    NOTE: ``relative_path`` is usually an absolute POSIX path (e.g. ``/Music/song.mp3``).
    For remote backends (webdav/sftp/ftp/...) the base URL carries a non-empty
    authority (``host:port/``); blindly ``fs.path.join``-ing an absolute path onto it
    would discard the authority and yield a broken URL (e.g. ``webdav:///Music/...``).
    We therefore make the path relative before joining whenever an authority is present,
    and keep it absolute for authority-less URLs such as ``osfs:///``.
    """
    if '://' in base_url:
        # Split into 'osfs' and '/mnt/media/' or 'webdav' and 'host:port/'
        protocol, path_segment = base_url.split('://', 1)

        if path_segment:
            # Remote: path_segment is the authority (host:port[/]); append the path
            # relatively so posixpath doesn't discard the authority.
            joined_path = fs.path.join(path_segment, relative_path.lstrip('/'))
        else:
            # Local osfs: no authority — keep the absolute path so the triple-slash
            # form (osfs:///mnt/...) is preserved.
            joined_path = relative_path if relative_path.startswith('/') else '/' + relative_path

        return f"{protocol}://{joined_path}"
    else:
        # Fallback if it is already a plain local path
        return fs.path.join(base_url, relative_path)

import os
import tempfile
import fs
from typing import Optional

def resolve_to_local_path(file_path: str) -> tuple[str, Optional[str]]:
    """
    Convert a VFS URL to a local filesystem path that librosa / cv2 / ffmpeg
    / PIL can actually open. Those libraries have no concept of PyFilesystem2
    URL schemes like ``osfs://``.

    Returns ``(local_path, temp_to_cleanup)``:
    - Local paths and ``osfs://`` URLs are returned unchanged (the latter
        with the prefix stripped) and ``temp_to_cleanup`` is ``None``.
    - Remote URLs (``webdav://``, ``sftp://``, ``ftp://``) are streamed
        to a temp file with the original extension; the caller is responsible
        for deleting ``temp_to_cleanup`` once done.

    Raises on unrecoverable parse / download errors.
    """
    # Already a real local path (no scheme) — nothing to do.
    if '://' not in file_path:
        return file_path, None

    try:
        base_url, path_in_fs = resolve_base_and_path_from_url(file_path)
    except Exception as exc:
        print(f"[VirtualFileSystem] Could not parse VFS URL '{file_path}': {exc}")
        return file_path, None  # Pass through; caller will fail with a clearer error.

    # Docker mounts /mnt/media/ into the subprocess, so the path after
    # 'osfs://' is already a valid filesystem path.
    if base_url.startswith('osfs://'):
        return path_in_fs, None

    # Remote URL → download to a temp file with the original extension
    # so the downstream libraries can read it as a regular file.
    try:
        with fs.open_fs(base_url) as my_fs:
            data = my_fs.readbytes(path_in_fs)
        ext = os.path.splitext(path_in_fs)[1] or '.bin'
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        with open(temp_path, 'wb') as tmp:
            tmp.write(data)
        return temp_path, temp_path
    except Exception as exc:
        print(f"[VirtualFileSystem] Failed to download remote file '{file_path}': {exc}")
        raise