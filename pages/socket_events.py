from flask_socketio import SocketIO
from flask import request
import threading, time 

class CommonSocketEvents:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio

        # per-sid emit state
        self._lock = threading.Lock()
        self._per_sid = {}  # sid -> {'last': float, 'timer': threading.Timer|None, 'pending': str|None}
        self._interval = 0.25  # seconds

    def _fire(self, sid=None):
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

    def show_search_status(self, status: str):
        sid = request.sid
        now = time.time()
        with self._lock:
            st = self._per_sid.setdefault(sid, {'last': 0.0, 'timer': None, 'pending': None})
            can_emit_now = (now - st['last']) >= self._interval and st['timer'] is None
            if can_emit_now:
                st['last'] = now
            else:
                st['pending'] = status
                if st['timer'] is None:
                    delay = max(0.0, self._interval - (now - st['last']))
                    st['timer'] = threading.Timer(delay, self._fire, args=[sid])
                    st['timer'].daemon = True
                    st['timer'].start()

        if can_emit_now:
            self.socketio.emit('emit_show_search_status', status, room=sid)