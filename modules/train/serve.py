from datetime import datetime
import src.db_models
import datasets
import random
import json

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
from peft import PeftModel, LoraConfig
from transformers import TrainingArguments
from trl import SFTTrainer
import trl

import os
import shutil
import numpy as np

import transformers
#import llm_engine

from flask import request

from sklearn.model_selection import train_test_split
from src.scoring_models import Evaluator
from tqdm import tqdm

import pandas as pd


# Check if module exists before importing
try:
    import modules.music.train
except ImportError:
    print("modules.music.train module not found. Music evaluator training will be unavailable.")

try:
    import modules.images.train
except ImportError:
    print("modules.images.train module not found. Image evaluator training will be unavailable.")

try:
    import modules.train.universal_train
except ImportError:
    print("modules.train.universal_train module not found. Universal evaluator training will be unavailable.")

import time

def init_socket_events(socketio, cfg=None, app=None, data_folder='./project_data'):
    '''data = {
        'train_loss_hist': self.train_loss_hist,
        'eval_loss_hist': self.eval_loss_hist,
        'percent': percent * 100
    }
    socketio.emit("emit_display_loss_data", data) '''

    task_manager = app.task_manager

    def _make_train_callback(ctx, train_accuracy_hist, test_accuracy_hist):
        """Return a training progress callback that feeds both the Train page chart
        and the Task Manager modal."""
        last_emit = [0.0]

        def callback(status, percent, baseline_accuracy, train_accuracy=None, test_accuracy=None):
            if train_accuracy is not None and test_accuracy is not None:
                train_accuracy_hist.append(train_accuracy * 100)
                test_accuracy_hist.append(test_accuracy * 100)
            now = time.time()
            if now - last_emit[0] >= 1:
                last_emit[0] = now
                data = {
                    'status': status,
                    'percent': percent * 100,
                    'baseline_accuracy': baseline_accuracy * 100,
                    'train_accuracy_hist': list(train_accuracy_hist),
                    'test_accuracy_hist': list(test_accuracy_hist)
                }
                socketio.emit("emit_train_page_display_train_data", data)
            ctx.update(percent, status)

        return callback

    def _is_training_active():
        """Check if any training task is currently queued or running."""
        state = task_manager.get_state()
        all_tasks = ([state['active']] if state['active'] else []) + state['queued']
        return any(
            t and t.get('name', '').startswith('Train:')
            for t in all_tasks
        )

    @socketio.on("emit_train_page_get_training_status")
    def handle_emit_get_training_status(data=None):
        socketio.emit("emit_train_page_status", {"active": _is_training_active()}, room=request.sid)

    @socketio.on("emit_train_page_start_music_evaluator_training")
    def handle_emit_start_music_evaluator_training(data=None):
        if _is_training_active():
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return

        def _task(ctx):
            hist_train, hist_test = [], []
            cb = _make_train_callback(ctx, hist_train, hist_test)
            socketio.emit("emit_train_page_status", {"active": True})
            try:
                modules.music.train.train_music_evaluator(cfg, cb, socketio)
            finally:
                socketio.emit("emit_train_page_status", {"active": False})

        task_manager.submit('Train: music evaluator', _task)
        socketio.emit("emit_train_page_status", {"active": True})

    @socketio.on("emit_train_page_start_image_evaluator_training")
    def handle_emit_start_image_evaluator_training(data=None):
        if _is_training_active():
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return

        def _task(ctx):
            hist_train, hist_test = [], []
            cb = _make_train_callback(ctx, hist_train, hist_test)
            socketio.emit("emit_train_page_status", {"active": True})
            try:
                modules.images.train.train_image_evaluator(cfg, cb)
            finally:
                socketio.emit("emit_train_page_status", {"active": False})

        task_manager.submit('Train: image evaluator', _task)
        socketio.emit("emit_train_page_status", {"active": True})

    @socketio.on("emit_train_page_start_universal_evaluator_training")
    def handle_emit_start_universal_evaluator_training(data=None):
        if _is_training_active():
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return

        max_steps = data.get('max_steps', None) if data else None
        time_budget_seconds = data.get('time_budget_seconds', None) if data else None

        def _task(ctx):
            hist_train, hist_test = [], []
            cb = _make_train_callback(ctx, hist_train, hist_test)
            socketio.emit("emit_train_page_status", {"active": True})
            try:
                modules.train.universal_train.train_universal_evaluator(
                    cfg, cb,
                    max_steps=max_steps,
                    time_budget_seconds=time_budget_seconds,
                )
            finally:
                socketio.emit("emit_train_page_status", {"active": False})

        task_manager.submit('Train: universal evaluator', _task)
        socketio.emit("emit_train_page_status", {"active": True})