import datetime
import xxhash
import fs
import src.db_models as db_models
import anagnorisis_core.storage.virtual_file_system as vfs

# --------------- Fast Soft-Hashing Mechanism -----------------
# Content identity lives in the core, so the CLI, the annotator and the app
# all name a file the same way. Memory files are named after this hash, so a
# second implementation drifting would orphan ratings.
from anagnorisis_core.storage.soft_hash import (
    SOFT_HASH_ALGORITHM, SOFT_HASH_BLOCK_SIZE, SOFT_HASH_SAMPLES,
    get_file_soft_hash as _core_get_file_soft_hash,
)


class EventManager:
    """Manages file-related operations, including secure access, rating, and database interactions."""

    # Sampling params (tuned for speed vs. collision resistance)
    soft_hash_block_size = SOFT_HASH_BLOCK_SIZE
    soft_hash_samples = SOFT_HASH_SAMPLES
    soft_hash_algorithm = SOFT_HASH_ALGORITHM

    @classmethod
    @staticmethod
    def get_file_soft_hash(file_path: str) -> str:
        """Content fingerprint of a file. See anagnorisis_core.storage.soft_hash."""
        return _core_get_file_soft_hash(file_path)

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