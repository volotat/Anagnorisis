from datetime import datetime
import src.db_models
import datasets
import random
import json

import llm_engine

def init_socket_events(socketio, predictor):
  chat_messages = []

  @socketio.on("emit_chat_message")
  def handle_chat_message(msg, use_headers = True):
    nonlocal predictor
    if predictor is None:
      predictor = llm_engine.TextPredictor(socketio)
    
    if use_headers:
      chat_messages.append(["### HUMAN:", msg['message']])

      text = ''
      for head, message in chat_messages:
        text += f"{head}\n{message}\n"
      text += "### RESPONSE:\n"
      new_text = predictor.predict_from_text(text)
      chat_messages.append(["### RESPONSE:", new_text])
    else:
      chat_messages.append(["", msg['message']])

      text = ''
      for head, message in chat_messages:
        text += f"{head}{message}"
      new_text = predictor.predict_from_text(text)
      chat_messages.append(["", new_text])

    socketio.emit("emit_chat_messages", chat_messages)

  @socketio.on("emit_clean_history")
  def handle_clean_history():
    nonlocal chat_messages
    chat_messages=[]
    socketio.emit("emit_chat_messages", chat_messages)

  @socketio.on("chat_connect")
  def handle_connect():
    print("User has been connected to the chat")
    socketio.emit("emit_chat_messages", chat_messages)