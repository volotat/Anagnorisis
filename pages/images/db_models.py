from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.db_models import db

class ImagesLibrary(db.Model):
  id = db.Column(db.Integer, unique=True, primary_key=True)
  hash = db.Column(db.String, nullable=True, unique=True) #, index=True
  hash_algorithm = db.Column(db.String, nullable=True, default=None)
  file_path = db.Column(db.String, nullable=True)
  user_rating = db.Column(db.Float, nullable=True)
  user_rating_date = db.Column(db.DateTime, nullable=True)
  model_rating = db.Column(db.Float, nullable=True)
  model_hash = db.Column(db.String, nullable=True)

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}