from flask_socketio import SocketIO
from flask import request

class CommonSocketEvents:
    def __init__(self, socketio: SocketIO):
        self.socketio = socketio

    def show_search_status(self, status: str):
        self.socketio.emit('emit_show_search_status', status, room=request.sid)