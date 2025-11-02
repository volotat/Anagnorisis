from datetime import datetime
import src.db_models
import datasets
import random
import json

import torch
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
import pages.music.train
import pages.images.train

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
            pages.music.train.train_music_evaluator(cfg, callback, socketio)
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
            pages.images.train.train_image_evaluator(cfg, callback)
        finally:
            TRAINING_ACTIVE = False
            socketio.emit("emit_train_page_status", {"active": False})

    '''@socketio.on("emit_train_page_export_audio_dataset")
    def handle_emit_export_audio_dataset():
        return None
    
        # Create dataset from DB, select only music with user rating
        music_library_entries = src.db_models.MusicLibrary.query.filter(src.db_models.MusicLibrary.user_rating != None).all()

        # filter all non-mp3 files
        music_library_entries = [entry for entry in music_library_entries if entry.file_path.endswith('.mp3')]

        music_files = [entry.file_path for entry in music_library_entries]
        music_scores = [entry.user_rating for entry in music_library_entries]

        # get embeddings for that music
        embedder = AudioEmbedder(audio_embedder_model_path = "./models/MERT-v1-95M")

        print('Embed music files...')
        data = []
        for ind, music_entry in enumerate(tqdm(music_library_entries)):
            file_path = music_files[ind]
            score = music_scores[ind]

            embedding = embedder.embed_audio(file_path).tolist()
            data.append({
                "artist": music_entry.artist, 
                "title": music_entry.title,
                "score": score,
                "embedding": embedding,
            })


        df = pd.DataFrame(data)
        df.to_csv("audio_scores_dataset.csv", index=False)
        print("Dataset has been generated and exported as 'audio_scores_dataset.csv'!")'''