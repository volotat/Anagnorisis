from flask import Flask
from flask_socketio import SocketIO

# Import our dedicated managers
from .config_manager import ConfigManager
from .database_manager import DatabaseManager
from .security_manager import SecurityManager
from .route_manager import RouteManager
from .extension_manager import ExtensionManager
from .watchdog_manager import WatchdogManager
from .event_manager import EventManager

def create_app(root_folder):
    """
    APPLICATION FACTORY PATTERN
    This function coordinates the initialization of the entire Anagnorisis application.
    """

    # # -------------------------------------------------------------------
    # # ML WARM-UP / THREADING FIX
    # # Hugging Face 'transformers' uses a custom lazy-loading module that is 
    # # NOT thread-safe. We must pre-warm it heavily on the main thread here 
    # # so it gets securely cached in sys.modules before background tasks start.
    # print("Pre-warming ML libraries to ensure thread-safety...")
    # try:
    #     import transformers
    #     from transformers import AutoConfig
    #     import src.base_search_engine
    # except ImportError as e:
    #     print(f"Warning during ML warm-up: {e}")
    # # -------------------------------------------------------------------

    # 1. Base Setup: Load Configuration & Paths
    cfg, user_cfg, paths = ConfigManager.setup(root_folder)

    # 2. Initialize Flask App & SocketIO
    app = Flask(__name__, template_folder='../../modules', static_folder='../../static')
    app.config['SECRET_KEY'] = cfg.main.flask_secret_key
    
    # Attach shared configuration securely to the app instance to prevent variable-passing.
    app.cfg = cfg
    app.user_cfg = user_cfg
    app.paths = paths
    app.root_folder = root_folder

    socketio = SocketIO(app, cors_allowed_origins="*", path="/socket.io")

    # 3. Initialize Sub-Systems (Strict execution order is maintained here)
    # Each manager binds its specific logic exclusively via the app instance.
    SecurityManager.init_app(app)
    DatabaseManager.init_app(app)
    ExtensionManager.init_models(app) # Must happen before db creation!
    DatabaseManager.create_and_migrate(app)
    RouteManager.init_app(app)
    ExtensionManager.init_socket_events(app, socketio)
    WatchdogManager.init_watchdog(app, socketio)
    EventManager.init_socket_events(app, socketio)

    return app, socketio, cfg