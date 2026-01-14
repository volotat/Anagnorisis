from flask_socketio import SocketIO
from flask import request
import threading, time 

# Global shared state across all CommonSocketEvents instances
_GLOBAL_LOCK = threading.Lock()
_GLOBAL_PER_SID = {}  # sid -> {'last': float, 'timer': threading.Timer|None, 'pending': str|None}

class CommonSocketEvents:
    def __init__(self, socketio: SocketIO, module_name: str = ""):
        self.socketio = socketio
        self.module_name = module_name

        # All instances share state to avoid cross-instance races
        self._lock = _GLOBAL_LOCK
        self._per_sid = _GLOBAL_PER_SID
        self._interval = 0.25  # seconds

    def _fire(self, sid=None):
        if sid is None:
            return
        with self._lock:
            st2 = self._per_sid.get(sid)
            if not st2:
                return
            msg = st2['pending']
            st2['pending'] = None
            st2['timer'] = None
            st2['last'] = time.time()
        if msg is not None:
            self.socketio.emit('emit_show_search_status', msg, room=sid)

    def show_search_status(self, status: str, sid: str | None = None):
        # If we are in a background thread (no request context), broadcast to all
        try:
            current_sid = request.sid if sid is None else sid
        except (RuntimeError, AttributeError):
            # We are likely in a background thread without a request context
            current_sid = None

        if current_sid is None:
            # Broadcast mode (for initialization phase)
            self.socketio.emit('emit_show_search_status', status)
            return

        # Existing logic for specific client throttling
        now = time.time()
        can_emit_now = False
        with self._lock:
            st = self._per_sid.setdefault(current_sid, {'last': 0.0, 'timer': None, 'pending': None})
            can_emit_now = (now - st['last']) >= self._interval and st['timer'] is None
            if can_emit_now:
                st['last'] = now
            else:
                st['pending'] = status
                if st['timer'] is None:
                    delay = max(0.0, self._interval - (now - st['last']))
                    st['timer'] = threading.Timer(delay, self._fire, args=[current_sid])
                    st['timer'].daemon = True
                    st['timer'].start()

        if can_emit_now:
            self.socketio.emit('emit_show_search_status', status, room=current_sid)

    def show_loading_status(self, status: str, module_name: str | None = None):
        """
        Emit module-specific loading status during initialization.
        This broadcasts to all clients and is specialized per module.
        Unlike show_search_status, this is designed for initialization phase
        and shows status for each module independently.
        
        Args:
            module_name: Name of the module (e.g., 'music', 'text', 'images', 'videos')
            status: Status message to display
        """
        if module_name is None:
            _module_name = self.module_name
        else:
            _module_name = module_name

        self.socketio.emit('emit_loading_status', {
            'module': _module_name,
            'status': status
        })

    @staticmethod
    def cleanup_sid(sid: str):
        # Cancel any pending timer and forget state
        with _GLOBAL_LOCK:
            st = _GLOBAL_PER_SID.pop(sid, None)
            if st and st.get('timer'):
                try:
                    st['timer'].cancel()
                except Exception:
                    pass