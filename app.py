from flask import Flask, render_template, render_template_string, send_from_directory
from flask_socketio import SocketIO
from flask_migrate import Migrate

#from sqlalchemy import TypeDecorator, String

#import llm_engine
from datetime import datetime
import sys

import markdown
from markdown.extensions.codehilite import CodeHiliteExtension
from pymdownx.arithmatex import ArithmatexExtension

from omegaconf import OmegaConf
import os

from flask_sqlalchemy import SQLAlchemy
from importlib import import_module

from src.db_models import db

cfg = OmegaConf.load("config.yaml")

app = Flask(__name__, template_folder='pages', static_folder='static')
#app.debug = True

app.config['SECRET_KEY'] = cfg.main.flask_secret_key
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{cfg.main.database_path}"

socketio = SocketIO(app, cors_allowed_origins="*")

# List all folders in the extensions directory
extension_names = [entry.name for entry in os.scandir('pages') if entry.is_dir()]
extension_names = [entry for entry in extension_names if not entry.startswith('_')]

# Import models from each extension
for extension_name in extension_names:
  # Check if the extension has a db_models.py file
  if not os.path.exists(f'pages/{extension_name}/db_models.py'):
    continue

  # Import the module
  module = import_module(f'pages.{extension_name}.db_models')

  # Get the attributes of the module
  for attr_name in dir(module):
    attr = getattr(module, attr_name)

    # If the attribute is a SQLAlchemy model, add it to the SQLAlchemy instance
    if isinstance(attr, db.Model):
      db.Model.metadata.tables[attr.__tablename__] = attr.__table__

# Initialize Flask-Migrate
db.init_app(app)
migrate = Migrate(app, db)

markdown_extensions = [
    'tables',
    ArithmatexExtension(generic=True, preview=False, smart_dollar=False),
    'fenced_code', 
    'codehilite',
    'nl2br',
]

#### MAIN PAGE FUNCTIONALITY
@app.route('/')
def index():
  # Read the Markdown file and convert it to HTML
  with open('README.md', 'r') as file:
    markdown_text = file.read()
    html = markdown.markdown(markdown_text, extensions=markdown_extensions)

  return render_template('wiki.html', content=html, cfg=cfg, pages=extension_names, current_page='wiki')

#### WIKI PAGE FUNCTIONALITY
@app.route('/wiki/<page_name>')
def page_wiki(page_name):
  # Assume your Markdown files are in a folder named 'wiki'
  file_path = os.path.join('wiki', f'{page_name}')

  if os.path.exists(file_path):
    with open(file_path, 'r') as file:
      markdown_text = file.read()
      html = markdown.markdown(markdown_text, extensions=markdown_extensions)
  else:
    # Handle the case where the file doesn't exist
    html = '<p>Page not found</p>'
  return render_template('wiki.html', content=html, cfg=cfg, pages=extension_names, current_page='wiki')


#### SERVING FILES FROM PAGES FOLDER, TO MAKE EXTENSIONS FILES ACCESSIBLE
@app.route('/pages/<path:filename>')
def custom_static(filename):
    return send_from_directory('pages', filename)

#### EXTENSIONS PAGES FUNCTIONALITY
def create_route(ext_name):
  def extension_route():
    with open(f'pages/{ext_name}/page.html', 'r') as f:
      page_content = f.read()
    page_template = """
    {% extends "base.html"%}
    {% block content %}
    """ + page_content + """
    {% endblock %}
    """
    return render_template_string(page_template, cfg=cfg, pages=extension_names, current_page=ext_name)
  return extension_route

# Initialize the socket events for each extension
for extension_name in extension_names:
    if not os.path.exists(f'pages/{extension_name}/serve.py'):
        continue

    serve_module_path = f'pages.{extension_name}.serve'
    try:
        module = import_module(serve_module_path)
        if hasattr(module, 'init_socket_events') and callable(module.init_socket_events):
            module.init_socket_events(socketio, app=app, cfg=cfg)

            # Create a route for the extension
            app.add_url_rule(f'/{extension_name}', f'{extension_name}_route', create_route(extension_name))
        else:
            print(f"Warning: Module {serve_module_path} does not have a callable init_socket_events function.")
    except ImportError as e:
        print(f"Warning: Could not import module {serve_module_path}: {e}")
    except Exception as e:
        print(f"Error initializing extension {extension_name}: {e}")

#### EXPORT DATABASE TO CSV FUNCTIONALITY
import src.db_models
from flask import send_file
import io

@app.route('/export_database_csv')
def export_database_csv():
    """
    Exports all database tables to a CSV file, excluding BLOB data, and sends it as a response.
    """
    csv_data = src.db_models.export_db_to_csv(db.session, excluded_columns=['embedding'])  # Exclude embedding column
    
    csv_file = io.BytesIO()
    csv_file.write(csv_data.encode('utf-8'))
    csv_file.seek(0)

    return send_file(
        csv_file,
        mimetype='text/csv',
        as_attachment=True,
        download_name='database_export.csv'
    )

@socketio.on('emit_import_database_csv')
def import_database_csv(csv_data):
    """
    Handles the import of data from a CSV string into the database.
    """
    print('Importing the database from csv')
    src.db_models.import_db_from_csv(db.session, csv_data)
    print('Database has been imported successfully')

#### RUNNING THE APPLICATION
socketio.run(app, host=cfg.main.host, port=cfg.main.port)
