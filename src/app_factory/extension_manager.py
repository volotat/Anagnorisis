import os
import time
import traceback
from importlib import import_module
from contextlib import contextmanager
from flask import render_template, render_template_string
from src.db_models import db
from src.task_manager import TaskManager
from src.socket_events import CommonSocketEvents

class ExtensionManager:
    """Manages dynamic discovery, DB model binding, and background initialization of app modules."""
    
    @classmethod
    def get_extension_names(cls, root_folder):
        """List all folders in the extensions directory"""
        modules_dir = os.path.join(root_folder, 'modules')
        if not os.path.exists(modules_dir):
            return []
        extension_names = [entry.name for entry in os.scandir(modules_dir) if entry.is_dir()]
        return [entry for entry in extension_names if not entry.startswith('_')]

    @classmethod
    def init_models(cls, app):
        """Dynamically binds module db_models to SQLAlchemy metadata."""
        extension_names = cls.get_extension_names(app.root_folder)
        
        # Import models from each extension
        for extension_name in extension_names:
            # Check if the extension has a db_models.py file
            if not os.path.exists(os.path.join(app.root_folder, 'modules', extension_name, 'db_models.py')):
                continue

            # Import the module
            module = import_module(f'modules.{extension_name}.db_models')

            # Get the attributes of the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)

                # If the attribute is a SQLAlchemy model, add it to the SQLAlchemy instance
                if isinstance(attr, db.Model):
                    db.Model.metadata.tables[attr.__tablename__] = attr.__table__

    @classmethod
    def init_socket_events(cls, app, socketio):
        """Initializes TaskManager, CSV imports, and async background module loading."""
        
        # Initialize the background task manager (queue + worker thread)
        task_manager = TaskManager(socketio, app)
        app.task_manager = task_manager

        @socketio.on('emit_import_database_csv')
        def import_database_csv(csv_data):
            """
            Handles the import of data from a CSV string into the database.
            """
            print('Importing the database from csv')
            from src.db_models import import_db_from_csv
            import_db_from_csv(db.session, csv_data)
            print('Database has been imported successfully')
            
        cls._register_extensions_async(app, socketio)

    @classmethod
    def _register_extensions_async(cls, app, socketio):
        """
        Registers routes immediately with a wrapper.
        Starts initialization in a SINGLE background thread to avoid import race conditions.
        """
        extension_names = cls.get_extension_names(app.root_folder)
        MODULE_STATUS = {name: False for name in extension_names}

        # Filter valid extensions first
        valid_extensions = []
        for extension_name in extension_names:
            if os.path.exists(os.path.join(app.root_folder, 'modules', extension_name, 'serve.py')):
                valid_extensions.append(extension_name)

        # 1. Register wrappers for all valid extensions
        for extension_name in valid_extensions:
            # We bind extension_name as a default arg to capture it in the closure
            def view_wrapper(name=extension_name):
                if not MODULE_STATUS.get(name, False):
                    return render_template('loading.html', module_name=name)
                else:
                    return cls._create_real_route(name, app.cfg, extension_names)()

            # Register the URL rule immediately
            app.add_url_rule(f'/{extension_name}', f'{extension_name}_route', view_wrapper)

        # 2. Define a sequential initialization function
        def sequential_initializer():
            print("Starting sequential initialization of extensions...")
            # Give the server a moment to fully start up
            time.sleep(1)
            for extension_name in valid_extensions:
                cls._background_init_extension(app, socketio, app.cfg, app.root_folder, extension_name, MODULE_STATUS)
            print("All extensions initialized.")

        # 3. Start the SINGLE background task
        socketio.start_background_task(sequential_initializer)

    @staticmethod
    def _create_real_route(ext_name, cfg, extension_names):
        """Returns the actual extension route function."""
        def extension_route():
            with open(f'modules/{ext_name}/page.html', 'r') as f:
                page_content = f.read()
            page_template = """
            {% extends "base.html"%}
            {% block content %}
            """ + page_content + """
            {% endblock %}
            """
            return render_template_string(page_template, cfg=cfg, pages=extension_names, current_page=ext_name)
        return extension_route

    @staticmethod
    def _background_init_extension(app, socketio, cfg, data_folder, ext_name, MODULE_STATUS):
        """Initializes a single extension in the background."""
        print(f"[{ext_name}] Starting background initialization...")

        # Emit initial status
        socketio.emit('emit_loading_status', {
            'module': ext_name,
            'status': 'Starting initialization...'
        })
        
        serve_module_path = f'modules.{ext_name}.serve'
        try:
            # Use app_context for the import as well
            with app.app_context():
                socketio.emit('emit_loading_status', {
                    'module': ext_name,
                    'status': 'Loading module...'
                })
                
                module = import_module(serve_module_path)

                # This is the heavy blocking call.
                # We wrap it to allow it to register routes even though the app is running.
                with ExtensionManager._allow_route_modifications(app):
                    # NEW ARCHITECTURE: Look for the clean factory hook
                    if hasattr(module, 'register_module') and callable(module.register_module):
                        module.register_module(app, socketio, cfg, data_folder)
                    
                    # OLD ARCHITECTURE: Fallback for legacy modules
                    elif hasattr(module, 'init_socket_events') and callable(module.init_socket_events):
                        module.init_socket_events(socketio, app=app, cfg=cfg, data_folder=data_folder)
                    
                    else:
                        print(f"[{ext_name}] Warning: No register_module or init_socket_events found.")
                
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

    @staticmethod
    @contextmanager
    def _allow_route_modifications(app):
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