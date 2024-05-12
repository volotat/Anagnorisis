from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from src.db_models import db

class MusicLibrary(db.Model):
  hash = db.Column(db.String, primary_key=True, unique=True, nullable=False)
  file_path = db.Column(db.String, nullable=True)
  url_path = db.Column(db.String, nullable=True)
  title = db.Column(db.String, nullable=True)
  artist = db.Column(db.String, nullable=True)
  album = db.Column(db.String, nullable=True)
  track_num = db.Column(db.Integer, nullable=True)
  genre = db.Column(db.String, nullable=True)
  date = db.Column(db.String, nullable=True)
  duration = db.Column(db.Float, nullable=True)
  bitrate = db.Column(db.String, nullable=True)
  lyrics = db.Column(db.String, nullable=True)
  user_rating = db.Column(db.Integer, nullable=True)
  model_rating = db.Column(db.Integer, nullable=True)
  skip_score = db.Column(db.Integer, nullable=True, default=20, server_default='20')
  full_play_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')
  skip_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')
  last_played = db.Column(db.DateTime, nullable=True)

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}