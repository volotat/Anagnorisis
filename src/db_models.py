from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class News(db.Model):
  hash = db.Column(db.String, primary_key=True, unique=True, nullable=False)
  title = db.Column(db.String, nullable=True)
  text = db.Column(db.String, nullable=True)
  summary = db.Column(db.String, nullable=True)
  coordinates = db.Column(db.JSON, nullable=True)
  datetime = db.Column(db.DateTime, default=datetime.utcnow)
  importance = db.Column(db.Integer, nullable=True)

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
  skip_score = db.Column(db.Integer, nullable=True, default=20, server_default='20')
  full_play_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')
  skip_count = db.Column(db.Integer, nullable=True, default=0, server_default='0')

  def as_dict(self):
    return {column.name: getattr(self, column.name) for column in self.__table__.columns}