from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.db_models import db

class VideosLibrary(db.Model):
  id = db.Column(db.Integer, unique=True, primary_key=True)
  hash = db.Column(db.String, nullable=True, unique=True)
  hash_algorithm = db.Column(db.String, nullable=True, default='')
  file_path = db.Column(db.String, nullable=True) 
  user_rating = db.Column(db.Float, nullable=True) 
  user_rating_date = db.Column(db.DateTime, nullable=True)
  model_rating = db.Column(db.Float, nullable=True)
  model_hash = db.Column(db.String, nullable=True) 
  full_play_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')
  skip_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')
  last_played = db.Column(db.DateTime, nullable=True, default=None)

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}
