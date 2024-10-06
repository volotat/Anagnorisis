from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.db_models import db

class ImagesLibrary(db.Model):
  id = db.Column(db.Integer, unique=True, primary_key=True)
  hash = db.Column(db.String, nullable=False)
  fast_hash = db.Column(db.String, nullable=True)
  file_descriptor = db.Column(db.String, nullable=True)
  file_path = db.Column(db.String, nullable=True)
  user_rating = db.Column(db.Integer, nullable=True)
  user_rating_date = db.Column(db.DateTime, nullable=True)
  model_rating = db.Column(db.Integer, nullable=True)

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}