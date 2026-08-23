"""Identifying a file by its content, cheaply.

A path is not an identity: files get renamed, moved between drives and re-sorted
into new folders, and a rating attached to a path would quietly rot. So a file is
identified by a *soft hash* — a few sampled blocks of its content plus its size,
rather than the whole thing.

Sampling rather than hashing in full is what makes this usable: a two-hour video
is fingerprinted from 5 MiB, and over a network only those blocks are fetched.
The trade is that two files sharing their sampled regions and their exact size
would collide, which is why this is called *soft*. Memory files are named after
it, so a rated file keeps its rating when it moves.
"""
import fs
import xxhash

import anagnorisis_core.storage.virtual_file_system as vfs

SOFT_HASH_BLOCK_SIZE = 1 * 1024 * 1024   # 1 MiB per sample
SOFT_HASH_SAMPLES = 5                    # head, middle, tail pattern
# Part of every memory file, so a change here must be a visible version bump
# rather than a silent redefinition of what identity means.
SOFT_HASH_ALGORITHM = f"xxh3s:s{SOFT_HASH_SAMPLES}m{SOFT_HASH_BLOCK_SIZE}:v1.2"


def _xxh3_hash_stream(my_fs, path_in_fs: str, chunk_size: int = 16 * 1024 * 1024) -> str:
    h = xxhash.xxh3_128()
    # Open the file via PyFilesystem2 binary read mode (no buffering argument needed)
    with my_fs.open(path_in_fs, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()

def _xxh3_hash_sampled(my_fs, path_in_fs: str, size: int, block: int, samples: int) -> str:
    # Compute positions: head, evenly spaced, tail
    if samples <= 1:
        positions = [0]
    elif samples == 2:
        positions = [0, max(0, size - block)]
    else:
        step = (size - block) // (samples - 1)
        positions = [min(i * step, max(0, size - block)) for i in range(samples)]
        positions[0] = 0
        positions[-1] = max(0, size - block)

    h = xxhash.xxh3_128()
    with my_fs.open(path_in_fs, 'rb') as f:
        for pos in positions:
            f.seek(pos)  # PyFilesystem2 handles the remote HTTP Range seek seamlessly
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    # Mix file size to reduce collisions across similarly sampled files
    h.update(size.to_bytes(8, byteorder='little', signed=False))
    return h.hexdigest()
# ---------------- END: Fast Soft-Hashing Mechanism -----------------


def get_file_soft_hash(file_path: str) -> str:
    """
    Extremely fast content fingerprint for large files using sampled xxh3_128:
    - Reads fixed-size blocks from head, middle, and tail (samples=3) to minimize I/O.
    - Mixes in file size to reduce collisions between similar files.
    - For small files (<= total sampled bytes), falls back to full-file streaming xxh3_128.
    The cache key includes size and mtime_ns, so recomputation happens only on change.
    """
    # 1. Parse and extract the base_url and internal path
    base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)

    with fs.open_fs(base_url) as my_fs:
        # 2. Get file details (size, mtime_ns) via the PyFilesystem2 'details' namespace
        try:
            info = my_fs.getinfo(path_in_fs, namespaces=['details'])
        except Exception as e:
            raise FileNotFoundError(f"File not found on filesystem: {file_path}. Error: {e}")

        size = info.size
        modified_sec = info.get('details', 'modified')
        mtime_ns = int(modified_sec * 1e9) if modified_sec is not None else 0

        # cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{SOFT_HASH_ALGORITHM}"
        # cached = cls._fast_cache.get(cache_key)
        # if cached is not None:
        #     return cached

        if size <= SOFT_HASH_BLOCK_SIZE * SOFT_HASH_SAMPLES:
            # Small files: stream whole file (still very fast)
            digest = _xxh3_hash_stream(my_fs, path_in_fs)
            result = f"{digest}"
        else:
            # Large files: sample head/middle/tail (only downloads specified blocks over network)
            digest = _xxh3_hash_sampled(my_fs, path_in_fs, size=size, block=SOFT_HASH_BLOCK_SIZE, samples=SOFT_HASH_SAMPLES)
            result = f"{digest}"

        # cls._fast_cache.set(cache_key, result)
        return result
