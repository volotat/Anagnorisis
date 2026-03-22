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

# Global, process-local flag. Simple and survives page refresh in the same server process.
TRAINING_ACTIVE = False

def init_socket_events(socketio, cfg=None, app=None, data_folder='./project_data'):
    '''data = {
        'train_loss_hist': self.train_loss_hist,
        'eval_loss_hist': self.eval_loss_hist,
        'percent': percent * 100
    }
    socketio.emit("emit_display_loss_data", data) '''

    train_accuracy_hist = []
    test_accuracy_hist = []
    last_emit_time = 0  # Initialize the last emit time

    def callback(status, percent, baseline_accuracy, train_accuracy = None, test_accuracy = None):
        nonlocal last_emit_time  # Access the outer scope variable
        current_time = time.time()  # Get the current time

        if train_accuracy is not None and test_accuracy is not None:
            train_accuracy_hist.append(train_accuracy * 100)
            test_accuracy_hist.append(test_accuracy * 100)

        if current_time - last_emit_time >= 1:
            data = {
                'status': status,
                'percent': percent * 100,
                'baseline_accuracy': baseline_accuracy * 100,
                'train_accuracy_hist': list(train_accuracy_hist),
                'test_accuracy_hist': list(test_accuracy_hist)
            }
        socketio.emit("emit_train_page_display_train_data", data)

    @socketio.on("emit_train_page_get_training_status")
    def handle_emit_get_training_status():
        # return current status to just this client
        socketio.emit("emit_train_page_status", {"active": TRAINING_ACTIVE}, room=request.sid)

    @socketio.on("emit_train_page_start_music_evaluator_training")
    def handle_emit_start_music_evaluator_training():
        global TRAINING_ACTIVE
        nonlocal train_accuracy_hist, test_accuracy_hist # Access the outer scope variables
        train_accuracy_hist = []
        test_accuracy_hist = []

        if TRAINING_ACTIVE:
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return

        TRAINING_ACTIVE = True
        socketio.emit("emit_train_page_status", {"active": True})
        try:
            modules.music.train.train_music_evaluator(cfg, callback, socketio)
        finally:
            TRAINING_ACTIVE = False
            socketio.emit("emit_train_page_status", {"active": False})
    
    @socketio.on("emit_train_page_start_image_evaluator_training")
    def handle_emit_start_image_evaluator_training():
        global TRAINING_ACTIVE
        nonlocal train_accuracy_hist, test_accuracy_hist # Access the outer scope variables
        train_accuracy_hist = []
        test_accuracy_hist = []

        if TRAINING_ACTIVE:
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return
        
        TRAINING_ACTIVE = True
        socketio.emit("emit_train_page_status", {"active": True})
        try:
            modules.images.train.train_image_evaluator(cfg, callback)
        finally:
            TRAINING_ACTIVE = False
            socketio.emit("emit_train_page_status", {"active": False})

    @socketio.on("emit_train_page_start_universal_evaluator_training")
    def handle_emit_start_universal_evaluator_training(data=None):
        global TRAINING_ACTIVE
        nonlocal train_accuracy_hist, test_accuracy_hist
        train_accuracy_hist = []
        test_accuracy_hist = []

        if TRAINING_ACTIVE:
            socketio.emit("emit_train_page_status", {"active": True}, room=request.sid)
            return

        max_steps = None
        time_budget_seconds = None
        if data:
            max_steps = data.get('max_steps', None)
            time_budget_seconds = data.get('time_budget_seconds', None)

        TRAINING_ACTIVE = True
        socketio.emit("emit_train_page_status", {"active": True})
        try:
            modules.train.universal_train.train_universal_evaluator(
                cfg, callback,
                max_steps=max_steps,
                time_budget_seconds=time_budget_seconds,
            )
        finally:
            TRAINING_ACTIVE = False
            socketio.emit("emit_train_page_status", {"active": False})