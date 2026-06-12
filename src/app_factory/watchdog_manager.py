import os
import threading
import time
from flask import request
from src.log_streamer import LogStreamer
from src.socket_events import CommonSocketEvents

class WatchdogManager:
    """Manages Background Threads like the LogStreamer and Memory/Subprocess Watchdog."""
    
    @classmethod
    def init_watchdog(cls, app, socketio):
        # --- Real-time Log Streaming Setup ---
        # Assuming your log file is named 'anagnorisis-app.log' in a 'logs' directory
        container_name = os.environ.get('CONTAINER_NAME', 'container')
        log_file_name = f"{container_name}_log.txt"
        log_file_path = os.path.join(app.root_folder, 'logs', log_file_name)
        log_streamer = LogStreamer(socketio, log_file_path)

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

        # Start the log watcher in a background thread
        socketio.start_background_task(log_streamer.watch)

        # ── Subprocess watchdog ──────────────────────────────────────────────────
        # Logs subprocess health and process memory every 60 s so that silent
        # failures (OOM kills, CUDA faults after suspend/resume, etc.) leave a
        # clear timestamp trail in the app log.
        def _subprocess_watchdog():
            # Proxy singletons — imported lazily so the watchdog doesn't force them
            # to be created if they were never used.
            _proxy_modules = {
                'TextEmbedder':       ('src.text_embedder',      'TextEmbedder'),
                'ImageEmbedder':      ('src.image_embedder',     'ImageEmbedder'),
                'AudioEmbedder':      ('src.audio_embedder',     'AudioEmbedder'),
                'OmniDescriptor':     ('src.omni_descriptor',    'OmniDescriptor'),
                'UniversalEvaluator': ('src.universal_evaluator','UniversalEvaluator'),
            }
            while True:
                time.sleep(60)
                try:
                    # Memory: read /proc/self/status (Linux, always available)
                    mem_mb = None
                    try:
                        with open('/proc/self/status') as _f:
                            for _line in _f:
                                if _line.startswith('VmRSS:'):
                                    mem_mb = int(_line.split()[1]) // 1024
                                    break
                    except Exception:
                        pass

                    # Subprocess statuses
                    parts = []
                    for name, (mod_path, cls_name) in _proxy_modules.items():
                        try:
                            import importlib as _il
                            mod = _il.import_module(mod_path)
                            inst = getattr(mod, cls_name)._instance
                            if inst is None:
                                parts.append(f'{name}=unused')
                            elif inst._process is None:
                                parts.append(f'{name}=idle')
                            elif inst._process.is_alive():
                                parts.append(f'{name}=alive(pid={inst._process.pid})')
                            else:
                                parts.append(f'{name}=DEAD(exit={inst._process.exitcode})')
                        except Exception:
                            parts.append(f'{name}=?')

                    mem_str = f'  RAM={mem_mb}MB' if mem_mb is not None else ''
                    print(f'[Watchdog]{mem_str}  ' + '  '.join(parts), flush=True)
                except Exception as _exc:
                    print(f'[Watchdog] error: {_exc}', flush=True)

        _wd = threading.Thread(target=_subprocess_watchdog, daemon=True, name='SubprocessWatchdog')
        _wd.start()
        # ────────────────────────────────────────────────────────────────────────