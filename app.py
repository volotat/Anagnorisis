import os
import multiprocessing
from src.app_factory import create_app

# Define the absolute root of the project ONCE at the top level.
# This is safe for child processes to read.
ROOT_FOLDER = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    # 1. ENFORCE MULTIPROCESSING SAFETY
    # This guarantees PyTorch and CUDA do not hang when spawning child workers.
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # 2. CREATE THE APP HERE
    # By keeping this inside __main__, we guarantee that background ML 
    # child processes do not accidentally start their own Flask servers.
    app, socketio, cfg = create_app(ROOT_FOLDER)

    print("Starting the application...")
    
    # 3. RUN THE SERVER
    socketio.run(
        app, 
        host=cfg.main.host, 
        port=cfg.main.port, 
        allow_unsafe_werkzeug=True
    )