from flask import Flask, render_template
from flask_socketio import SocketIO
from flask_migrate import Migrate

#from sqlalchemy import TypeDecorator, String

import llm_engine
from datetime import datetime
import sys

import markdown
from omegaconf import OmegaConf

def init_app():
  cfg = OmegaConf.load("config.yaml")

  app = Flask(__name__)
  #app.debug = True

  app.config['SECRET_KEY'] = cfg.main.flask_secret_key
  app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"

  socketio = SocketIO(app, cors_allowed_origins="*")

  import src.db_models
  src.db_models.db.init_app(app)

  migrate = Migrate(app, src.db_models.db)
  migrate.init_app(app, src.db_models.db)

  predictor = None 

  #### MAIN PAGE FUNCTIONALITY
  @app.route('/')
  def index():
    # Read the Markdown file and convert it to HTML
    with open('README.md', 'r') as file:
      markdown_text = file.read()
      html = markdown.markdown(markdown_text)

    return render_template('main.html', content=html, cfg=cfg)

  #### CHAT PAGE FUNCTIONALITY
  import src.chat_page
  src.chat_page.init_socket_events(socketio, predictor)

  @app.route('/chat')
  def page_chat():
    return render_template('chat.html', cfg=cfg)

  #### NEWS PAGE FUNCTIONALITY
  import src.news_page
  src.news_page.init_socket_events(socketio, predictor)

  @app.route('/news')
  def page_news():
    return render_template('news.html', cfg=cfg)
  
  #### MUSIC PAGE FUNCTIONALITY
  import src.music_page
  src.music_page.init_socket_events(socketio, predictor, app=app, cfg=cfg.music_page)

  @app.route('/music')
  def page_music():
    return render_template('music.html', cfg=cfg)

  #### FINE-TUNING PAGE FUNCTIONALITY
  import src.fine_tune_page
  src.fine_tune_page.init_socket_events(socketio, predictor)

  @app.route('/fine-tune')
  def page_fine_tune():
    return render_template('fine-tune.html', cfg=cfg)

  #### RUNNING THE APPLICATION
  socketio.run(app, host=cfg.main.host, port=cfg.main.port)

#### Starting the application
if __name__ == '__main__':
  init_app()
  