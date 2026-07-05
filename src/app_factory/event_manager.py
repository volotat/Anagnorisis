import datetime
import xxhash
import fs
import src.db_models as db_models
import src.virtual_file_system as vfs

# --------------- Fast Soft-Hashing Mechanism -----------------
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

class EventManager:
    """Manages file-related operations, including secure access, rating, and database interactions."""

    # Sampling params (tuned for speed vs. collision resistance)
    soft_hash_block_size = 1 * 1024 * 1024  # 1 MiB per sample
    soft_hash_samples = 5                   # head, middle, tail pattern
    soft_hash_algorithm = f"xxh3s:s{soft_hash_samples}m{soft_hash_block_size}:v1.2" # Sampled xxh3_128 with 5 samples of 1 MiB each

    @classmethod
    def get_file_soft_hash(cls, file_path: str) -> str:
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

            # cache_key = f"HASH_OF_FILE::{file_path}::{size}::{mtime_ns}::{cls.soft_hash_algorithm}"
            # cached = cls._fast_cache.get(cache_key)
            # if cached is not None:
            #     return cached

            if size <= cls.soft_hash_block_size * cls.soft_hash_samples:
                # Small files: stream whole file (still very fast)
                digest = _xxh3_hash_stream(my_fs, path_in_fs)
                result = f"{digest}"
            else:
                # Large files: sample head/middle/tail (only downloads specified blocks over network)
                digest = _xxh3_hash_sampled(my_fs, path_in_fs, size=size, block=cls.soft_hash_block_size, samples=cls.soft_hash_samples)
                result = f"{digest}"

            # cls._fast_cache.set(cache_key, result)
            return result

    @classmethod
    def init_socket_events(cls, app, socketio):
        """Initializes FileManager socket events."""

        @socketio.on('emit_set_file_rating')
        def set_file_rating(data):
            file_path = data['file_path']
            file_rating = data['rating']

            file_soft_hash = cls.get_file_soft_hash(file_path)

            print('[AppFactory:FileManager] Set file rating:', file_path, file_rating)

            files_db_item = db_models.FilesLibrary.query.filter_by(file_path=file_path).first()

            if files_db_item is None:
                # Create new instance if there is no entry in the database
                files_data = {
                    "hash": file_soft_hash,
                    "hash_algorithm": cls.soft_hash_algorithm,
                    "file_path": file_path,
                    "user_rating": float(file_rating),
                    "user_rating_date": datetime.datetime.now()
                }
                files_db_item = db_models.FilesLibrary(**files_data)
                db_models.db.session.add(files_db_item)
                db_models.db.session.commit()
            else:
                files_db_item.hash = file_soft_hash
                files_db_item.hash_algorithm = cls.soft_hash_algorithm
                files_db_item.user_rating = float(file_rating)
                files_db_item.user_rating_date = datetime.datetime.now()
                db_models.db.session.commit()

            # Write/refresh the durable memory .md for this file (background task,
            # non-blocking). The rating is stored as the first line of the .md and
            # stripped before embedding at train time, so the evaluator never sees
            # the score in the text it predicts.
            memory_system = getattr(app, 'memory_system', None)
            if memory_system is not None:
                memory_system.save_memory(file_path, float(file_rating), soft_hash=file_soft_hash)