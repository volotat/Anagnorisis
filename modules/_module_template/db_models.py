"""
db_models.py — SQLAlchemy database model (OPTIONAL)

Only create this file if your module needs to persist per-file data across
restarts (user ratings, play counts, model ratings, etc.).  If your module
doesn't need a database table, simply omit this file — the main application
handles its absence gracefully.

When you do create it:
  • Import ``db`` from ``src.db_models`` — do NOT create your own SQLAlchemy
    instance.
  • The ``hash`` column should be unique and serves as the content-based
    identifier for each file (computed by the search engine).
  • Keep the standard columns (hash, user_rating, model_rating, etc.) for
    compatibility with FileManager, CommonFilters, and CSV export/import.
  • Add ``as_dict()`` so the rest of the framework can serialise rows easily.

See BEST_PRACTICES.md for guidance on when to create this file.
"""

from src.db_models import db
from datetime import datetime


class ExampleLibrary(db.Model):
    """
    One row per unique file managed by this module.

    Columns present in every media module (keep these for compatibility):
      id, hash, hash_algorithm, file_path, user_rating, user_rating_date,
      model_rating, model_hash

    Add any module-specific columns below the separator comment.
    """
    id = db.Column(db.Integer, unique=True, primary_key=True)

    # --- Content identity --------------------------------------------------
    hash = db.Column(db.String, nullable=True, unique=True)
    hash_algorithm = db.Column(db.String, nullable=True, default=None)
    file_path = db.Column(db.String, nullable=True)  # Relative to media_directory

    # --- Ratings -----------------------------------------------------------
    user_rating = db.Column(db.Float, nullable=True)
    user_rating_date = db.Column(db.DateTime, nullable=True)
    model_rating = db.Column(db.Float, nullable=True)
    model_hash = db.Column(db.String, nullable=True)  # Hash of the evaluator that produced model_rating

    # --- Module-specific columns (add yours here) -------------------------
    # duration = db.Column(db.Float, nullable=True)       # e.g. for audio/video
    # full_play_count = db.Column(db.Integer, default=0)  # e.g. for music/video
    # skip_count = db.Column(db.Integer, default=0)

    def as_dict(self):
        """Return all columns as a plain Python dict (used by CSV export, etc.)."""
        return {col.name: getattr(self, col.name) for col in self.__table__.columns}
