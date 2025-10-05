import time
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class LogStreamer:
    def __init__(self, socketio, log_file_path):
        self.socketio = socketio
        self.log_file_path = log_file_path
        self.last_read_position = 0
        self._observer = Observer()

        # Read initial position
        try:
            with open(self.log_file_path, 'r') as f:
                f.seek(0, 2) # Go to the end of the file
                self.last_read_position = f.tell()
        except FileNotFoundError:
            print(f"Log file not found at {self.log_file_path}. Will be created.")
            pass # File might not exist yet

    def send_history(self, sid):
        """Sends the entire log file history to a specific client."""
        print(f"Sending log history to client {sid}")
        try:
            with open(self.log_file_path, 'r') as f:
                history = f.read()
                if history:
                    self.socketio.emit('log_history', {'data': history}, room=sid)
        except FileNotFoundError:
            # If the file doesn't exist, do nothing.
            pass
        except Exception as e:
            print(f"Error sending log history: {e}")


    def _read_new_lines(self):
        """Reads new lines from the log file and emits them."""
        try:
            with open(self.log_file_path, 'r') as f:
                f.seek(self.last_read_position)
                new_lines = f.read()
                if new_lines:
                    # Emit the new content to all clients
                    self.socketio.emit('log_update', {'data': new_lines})
                self.last_read_position = f.tell()
        except Exception as e:
            print(f"Error reading log file: {e}")

    def watch(self):
        """Starts watching the log file for changes."""
        event_handler = FileSystemEventHandler()

        def on_modified(event):
            if event.src_path == self.log_file_path:
                self._read_new_lines()

        event_handler.on_modified = on_modified

        # Start watching the directory containing the log file
        log_dir = os.path.dirname(self.log_file_path)
        self._observer.schedule(event_handler, log_dir, recursive=False)
        self._observer.start()
        print(f"Started watching log file: {self.log_file_path}")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._observer.stop()
        self._observer.join()