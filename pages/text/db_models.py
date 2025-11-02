from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.db_models import db
from sqlalchemy.types import DateTime # Import DateTime type explicitly

# Define the database model for text files
class TextLibrary(db.Model):
    id = db.Column(db.Integer, unique=True, primary_key=True)
    hash = db.Column(db.String, nullable=True, unique=True)
    hash_algorithm = db.Column(db.String, nullable=True, default='')
    file_path = db.Column(db.String, nullable=True) # Relative path within media directory
    user_rating = db.Column(db.Float, nullable=True) # For future rating feature
    user_rating_date = db.Column(db.DateTime, nullable=True)
    model_rating = db.Column(db.Float, nullable=True) # For future evaluator model
    model_hash = db.Column(db.String, nullable=True) # Hash of the evaluator model

    def as_dict(self):
        return {column.name: getattr(self, column.name) for column in self.__table__.columns}