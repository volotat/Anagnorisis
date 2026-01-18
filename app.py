from flask import Flask, render_template, render_template_string, send_from_directory
from flask_socketio import SocketIO
from flask_migrate import Migrate
from flask_httpauth import HTTPBasicAuth 

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
from contextlib import contextmanager

from src.db_models import db
from src.config_loader import load_config
from src.log_streamer import LogStreamer

# Add after your imports, before loading config
import argparse
import os
import shutil

import traceback
import time

import threading

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Anagnorisis Application')
#     parser.add_argument('--data-folder', type=str, default='/project_data', help='Path to data folder')
#     return parser.parse_args()

# Get data folder from command line
# args = parse_arguments()
# data_folder = str(args.data_folder)
script_folder = os.path.dirname(os.path.abspath(__file__))
print(f"Script folder: {script_folder}")

# Create logs directory if it doesn't exist
logs_folder = os.path.join(script_folder, 'logs')
if not os.path.exists(logs_folder):
    os.makedirs(logs_folder, exist_ok=True)

# Check if running in a Docker container
# if os.environ.get('RUNNING_IN_DOCKER') == 'true':
#     print("Running in Docker container")
#     # If running in Docker, use the data folder from the environment variable
#     if data_folder.startswith('/'):
#         # If it's an absolute path, make it relative to /app
#         data_folder = os.path.join('/app', data_folder[1:])
#     else:
#         # If it's already relative, just prepend /app
#         data_folder = os.path.join('/app', data_folder)

data_folder = script_folder

print(f"Using data folder: {data_folder}")

# Load default configuration
config_path = os.path.join(script_folder, 'config.yaml')
cfg = OmegaConf.load(config_path)

# Load local configuration if it exists
project_config_folder_path = cfg.main.get('project_config_directory', 'project_config')

#local_config_path = os.path.join(project_config_folder_path, 'config.yaml')
database_path = os.path.join(project_config_folder_path, 'database', 'project.db')
migrations_path = os.path.join(project_config_folder_path, 'database', 'migrations')
embedding_models_path = os.path.join(script_folder, 'models')
personal_models_path = os.path.join(project_config_folder_path, 'models')
cache_path = os.path.join(project_config_folder_path, 'cache')

if not os.path.exists(project_config_folder_path):
    os.makedirs(project_config_folder_path, exist_ok=True)

if not os.path.exists(os.path.join(project_config_folder_path, 'database')):
    os.makedirs(os.path.join(project_config_folder_path, 'database'), exist_ok=True)

if not os.path.exists(os.path.join(project_config_folder_path, 'models')):
    os.makedirs(os.path.join(project_config_folder_path, 'models'), exist_ok=True)

if not os.path.exists(os.path.join(project_config_folder_path, 'cache')):
    os.makedirs(os.path.join(project_config_folder_path, 'cache'), exist_ok=True)

cfg.main.database_path = database_path
cfg.main.migrations_path = migrations_path
cfg.main.embedding_models_path = embedding_models_path
cfg.main.personal_models_path = personal_models_path
cfg.main.cache_path = cache_path

# Ensure database path is properly set up
# If the database path is relative, make it absolute within the data folder
#if not os.path.isabs(cfg.main.database_path):
#    database_path = os.path.join(script_folder, cfg.main.database_path)
#else:
#    database_path = cfg.main.database_path

print(f"Database path set to: {database_path}")

# Ensure database directory exists
db_dir = os.path.dirname(database_path)
if db_dir and not os.path.exists(db_dir):
    print(f"Creating database directory: {db_dir}")
    os.makedirs(db_dir, exist_ok=True)

# Now initialize the Flask app with the properly configured database path
app = Flask(__name__, template_folder='pages', static_folder='static')
app.config['SECRET_KEY'] = cfg.main.flask_secret_key
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{database_path}"

# Initialize Flask-HTTPAuth
auth = HTTPBasicAuth()

# If these environment variables are not set, os.environ.get() will return None.
# If they are set to an empty string, they will be empty.
AUTH_USERNAME = os.environ.get('ANAGNORISIS_USERNAME')
AUTH_PASSWORD = os.environ.get('ANAGNORISIS_PASSWORD')

# Authentication is active only if BOTH username and password are set and non-empty.
AUTH_ACTIVE = bool(AUTH_USERNAME and AUTH_PASSWORD)

if AUTH_ACTIVE:
    print("HTTP Basic Authentication is ACTIVE.")
else:
    print("HTTP Basic Authentication is INACTIVE (ANAGNORISIS_USERNAME or ANAGNORISIS_PASSWORD not set or empty).")

@auth.verify_password
def verify_password(username, password):
    # Only perform verification if authentication is active
    if AUTH_ACTIVE and username == AUTH_USERNAME and password == AUTH_PASSWORD:
        print(f"Authentication successful for user: {username}")
        return username
    print(f"Authentication failed for user: {username}")
    return None # Important: return None if auth is active but credentials are wrong

@auth.error_handler
def unauthorized():
    return "Unauthorized access. Please provide valid credentials.", 401

if AUTH_ACTIVE:
    # If auth is active, use auth.login_required
    auth_decorator = auth.login_required
else:
    # If auth is inactive, use a no-op decorator (does nothing)
    def no_auth_decorator(f):
        return f
    auth_decorator = no_auth_decorator

@app.before_request
@auth_decorator # This decorator will either require login or do nothing
def before_request_auth():
    pass

# Set the socketio parameters
socketio = SocketIO(app, cors_allowed_origins="*", path="/socket.io")


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
migrate = Migrate(app, db, directory=migrations_path)

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
@auth_decorator
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

# # Initialize the socket events for each extension
# for extension_name in extension_names:
#     if not os.path.exists(f'pages/{extension_name}/serve.py'):
#         continue

#     serve_module_path = f'pages.{extension_name}.serve'
#     try:
#         module = import_module(serve_module_path)
#         if hasattr(module, 'init_socket_events') and callable(module.init_socket_events):
#             module.init_socket_events(socketio, app=app, cfg=cfg, data_folder=data_folder)

#             # Create a route for the extension
#             app.add_url_rule(f'/{extension_name}', f'{extension_name}_route', create_route(extension_name))
#         else:
#             print(f"Warning: Module {serve_module_path} does not have a callable init_socket_events function.")
#     except ImportError as e:
#         print(f"Warning: Could not import module {serve_module_path}: {e}")
#         print(traceback.format_exc())
#     except Exception as e:
#         print(f"Error initializing extension {extension_name}: {e}")
#         print(traceback.format_exc())

# def register_extensions(app, socketio, cfg, data_folder):
#     """
#     Registers extensions, routes, and socket events.
#     Must be called inside if __name__ == '__main__' to avoid multiprocessing recursion.
#     """
#     # Initialize the socket events for each extension
#     for extension_name in extension_names:
#         if not os.path.exists(f'pages/{extension_name}/serve.py'):
#             continue

#         serve_module_path = f'pages.{extension_name}.serve'
#         try:
#             module = import_module(serve_module_path)
#             if hasattr(module, 'init_socket_events') and callable(module.init_socket_events):
#                 module.init_socket_events(socketio, app=app, cfg=cfg, data_folder=data_folder)

#                 # Create a route for the extension
#                 app.add_url_rule(f'/{extension_name}', f'{extension_name}_route', create_route(extension_name))
#             else:
#                 print(f"Warning: Module {serve_module_path} does not have a callable init_socket_events function.")
#         except ImportError as e:
#             print(f"Warning: Could not import module {serve_module_path}: {e}")
#             print(traceback.format_exc())
#         except Exception as e:
#             print(f"Error initializing extension {extension_name}: {e}")
#             print(traceback.format_exc())


# Global state to track module readiness
MODULE_STATUS = {name: False for name in extension_names}

@contextmanager
def allow_route_modifications(app):
    """
    Context manager to temporarily allow route registration after the app has started.
    This bypasses Flask's safety check to allow background initialization of modules
    that contain @app.route decorators.
    """
    # We monkey-patch the internal check method to do nothing temporarily
    if hasattr(app, '_check_setup_finished'):
        original_check = app._check_setup_finished
        # Replace with a no-op lambda
        app._check_setup_finished = lambda *args, **kwargs: None
        try:
            yield
        finally:
            # Restore the original check
            app._check_setup_finished = original_check
    else:
        # Fallback for older Flask versions or if internal API changes
        yield

def create_real_route(ext_name):
    """Returns the actual extension route function."""
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

def background_init_extension(app, socketio, cfg, data_folder, ext_name):
    """Initializes a single extension in the background."""
    print(f"[{ext_name}] Starting background initialization...")

    # Emit initial status
    socketio.emit('emit_loading_status', {
        'module': ext_name,
        'status': 'Starting initialization...'
    })
    
    serve_module_path = f'pages.{ext_name}.serve'
    try:
        # Use app_context for the import as well
        with app.app_context():
            socketio.emit('emit_loading_status', {
                'module': ext_name,
                'status': 'Loading module...'
            })
            
            module = import_module(serve_module_path)
            if hasattr(module, 'init_socket_events') and callable(module.init_socket_events):
                # This is the heavy blocking call.
                # We wrap it to allow it to register routes even though the app is running.
                with allow_route_modifications(app):
                    module.init_socket_events(socketio, app=app, cfg=cfg, data_folder=data_folder)
            else:
                print(f"[{ext_name}] Warning: No init_socket_events found.")
            
        print(f"[{ext_name}] Initialization complete.")
        MODULE_STATUS[ext_name] = True
        
        # Emit final status update (this gets stored in _GLOBAL_MODULE_STATUS)
        socketio.emit('emit_loading_status', {
            'module': ext_name,
            'status': 'Initialization complete'
        })
        
        # Notify any clients waiting on the loading screen
        socketio.emit(f'module_ready_{ext_name}', {'status': 'ready'})
        
    except Exception as e:
        print(f"[{ext_name}] Error initializing: {e}")
        traceback.print_exc()
        # Emit error to the loading screen so the user sees it
        socketio.emit('emit_show_search_status', f"Error initializing {ext_name}: {str(e)}")

def register_extensions_async(app, socketio, cfg, data_folder):
    """
    Registers routes immediately with a wrapper.
    Starts initialization in a SINGLE background thread to avoid import race conditions.
    """
    # Filter valid extensions first
    valid_extensions = []
    for extension_name in extension_names:
        if os.path.exists(f'pages/{extension_name}/serve.py'):
            valid_extensions.append(extension_name)

    # 1. Register wrappers for all valid extensions
    for extension_name in valid_extensions:
        # We bind extension_name as a default arg to capture it in the closure
        def view_wrapper(name=extension_name):
            if not MODULE_STATUS.get(name, False):
                return render_template('loading.html', module_name=name)
            else:
                # Render the actual page content
                return create_real_route(name)()

        # Register the URL rule immediately
        app.add_url_rule(f'/{extension_name}', f'{extension_name}_route', view_wrapper)

    # 2. Define a sequential initialization function
    def sequential_initializer():
        print("Starting sequential initialization of extensions...")
        # Give the server a moment to fully start up
        time.sleep(1)
        for extension_name in valid_extensions:
            background_init_extension(app, socketio, cfg, data_folder, extension_name)
        print("All extensions initialized.")

    # 3. Start the SINGLE background task
    socketio.start_background_task(sequential_initializer)   

#### EXPORT DATABASE TO CSV FUNCTIONALITY
import src.db_models
from flask import send_file
import io

@app.route('/export_database_csv')
def export_database_csv():
    """
    Exports all database tables to a CSV file, excluding BLOB data, and sends it as a response.
    """
    csv_data = src.db_models.export_db_to_csv(db.session, excluded_columns=['embedding', 'chunk_embeddings'])  # Exclude embedding column
    
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

def create_db_if_not_exists():
    """Create database tables if they don't exist"""
    db_path = database_path

    # If the path is relative, make it absolute
    if not os.path.isabs(db_path):
        db_path = os.path.join(script_folder, db_path)

    # Get the directory of the database file
    db_dir = os.path.dirname(db_path)
    
    # Create directory if it doesn't exist
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir)
    
    # Check if database file exists
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}. Creating database...")
        with app.app_context():
            db.create_all()
        print("Database created successfully.")
    else:
        print(f"Database found at {db_path}.")

def migrate_database():
    """Run database migrations if needed"""
    from flask_migrate import upgrade, migrate, init
    import os.path

    mg_path = cfg.main.migrations_path
    migrations_dir = os.path.join(script_folder, mg_path)
    print(f"Using migrations directory: {migrations_dir}")

    # Configure Flask-Migrate to use the custom directory
    migrate.directory = migrations_dir
    
    with app.app_context():
        # Check if migrations directory exists
        if not os.path.exists(migrations_dir):
            print("Initializing migrations...")
            init(directory=migrations_dir)
            
        # Generate migration
        print("Generating migrations...")
        migrate(directory=migrations_dir)
        
        # Apply migrations
        print("Applying migrations...")
        upgrade(directory=migrations_dir)
        
        print("Database migration completed successfully.")  

from flask import request, abort
from urllib.parse import unquote

#### PREVENT PATH TRAVERSAL
def _looks_like_path(s: str) -> bool:
    return ('/' in s) or ('\\' in s)

def _has_parent_segment(value: str) -> bool:
    # Decode once or twice to defeat %2e%2e / double-encoding tricks
    v = unquote(unquote(value))
    v = v.replace('\\', '/')
    parts = [p for p in v.split('/') if p]  # drop empty parts
    # Only block when a segment is exactly '..'
    return any(p == '..' for p in parts)

@app.before_request 
def block_path_traversal():
    # Keep absolute sensitive paths
    dangerous_substrings = ['/etc/', '/proc/']

    # Collect values to check
    check_values = [request.path]
    check_values.extend(request.args.values())
    check_values.extend(request.form.values())

    if request.is_json and request.json:
        def extract_strings(obj):
            if isinstance(obj, dict):
                for v in obj.values():
                    yield from extract_strings(v)
            elif isinstance(obj, list):
                for item in obj:
                    yield from extract_strings(item)
            elif isinstance(obj, str):
                yield obj
        check_values.extend(extract_strings(request.json))

    for value in check_values:
        if not isinstance(value, str):
            continue
        # Fast reject for truly dangerous absolute paths
        v_dec = unquote(unquote(value)).lower()
        if any(s in v_dec for s in dangerous_substrings):
            print(f"Path traversal attempt blocked: {value}")
            abort(403)
        # Only apply '..' check to path-like strings; allow filenames like '....flac'
        if _looks_like_path(value) and _has_parent_segment(value):
            print(f"Parent directory traversal blocked: {value}")
            abort(403)

# --- Real-time Log Streaming Setup ---
# Assuming your log file is named 'anagnorisis-app.log' in a 'logs' directory
container_name = os.environ.get('CONTAINER_NAME', 'container')
log_file_name = f"{container_name}_log.txt"
log_file_path = os.path.join(script_folder, 'logs', log_file_name)
log_streamer = LogStreamer(socketio, log_file_path)

from pages.socket_events import CommonSocketEvents

@socketio.on('connect')
def handle_connect():
    """Send full log history and current module loading status to a client when they connect."""
    
    # Send log history
    log_streamer.send_history(request.sid)
    
    # Send current module loading statuses
    module_statuses = CommonSocketEvents.get_all_module_statuses()
    for module_name, status_info in module_statuses.items():
        socketio.emit('emit_loading_status', {
            'module': module_name,
            'status': status_info['status']
        }, room=request.sid)

        # print(f"Sent loading status for module {module_name} to client {request.sid}")
        # print(f"Status: {status_info['status']}")

#### RUNNING THE APPLICATION
if __name__ == '__main__':
    print("Starting the application...")

    # Initialize extensions here to prevent recursion in subprocesses
    # register_extensions(app, socketio, cfg, data_folder)

    # Initialize the database
    create_db_if_not_exists()

    # Migrate the database if necessary
    migrate_database()

    # Register extensions asynchronously
    register_extensions_async(app, socketio, cfg, data_folder)

    # Start the log watcher in a background thread
    socketio.start_background_task(log_streamer.watch)

    # Run the application
    socketio.run(app, 
                host=cfg.main.host, 
                port=cfg.main.port, 
                allow_unsafe_werkzeug=True, 
                )
