
from flask import Flask, render_template, send_from_directory, Response, stream_with_context
import os
import sys
import glob
import subprocess
import hashlib
import numpy as np
from io import BytesIO
from PIL import Image
import math
from scipy.spatial import distance
import torch
import gc
import send2trash
import time
import datetime
from moviepy.editor import VideoFileClip

import threading
import uuid
import tempfile
import shutil

from pages.socket_events import CommonSocketEvents

import pages.file_manager as file_manager
import pages.videos.db_models as db_models
from pages.videos.engine import VideoSearch, VideoEvaluator
from pages.recommendation_engine import sort_files_by_recommendation
from pages.common_filters import CommonFilters

from pages.utils import convert_size, convert_length, time_difference

from src.metadata_search import MetadataSearch

# EVENTS:

# Incoming (handled with @socketio.on):

# emit_videos_page_get_files
# emit_videos_page_open_file_in_folder
# emit_videos_page_send_file_to_trash
# emit_videos_page_video_start_playing
# emit_videos_page_start_streaming
# emit_videos_page_stop_streaming

# Outgoing (emitted with socketio.emit):

# emit_videos_page_show_search_status
# emit_videos_page_show_files


def print_emb_extracting_status(num_extracted, num_total):
    if num_extracted % 100 == 0:
        print(f"Extracted embeddings for {num_extracted} out of {num_total} videos.")

def generate_preview(video_path, preview_path):
    """
    Generates a preview image for a given video file.

    Args:
    video_path (str): The path to the video file.
    preview_path (str): The path where the preview image will be saved.
    """
    try:
        # Load the video file
        clip = VideoFileClip(video_path)
        # Calculate the middle frame time
        middle_time = clip.duration / 2
        # Save the frame at the middle of the video as an image
        clip.save_frame(preview_path, t=middle_time)
    except Exception as e:
        print(f"Error generating preview for {video_path}: {e}")



def init_socket_events(socketio, app=None, cfg=None, data_folder='./project_data'):
    if cfg.videos.media_directory is None:
        print("Videos media folder is not set.")
        media_directory = None
    else:
        # media_directory = os.path.join(data_folder, cfg.videos.media_directory)
        media_directory = cfg.videos.media_directory

        # Check if media_directory exists, if not, print a warning and set to None
        if not os.path.isdir(media_directory):
            print(f"Warning: Videos media directory '{os.path.join(data_folder, cfg.videos.media_directory)}' does not exist. Setting media folder to None.")
            media_directory = None

    # necessary to allow web application access to music files
    @app.route('/video_files/<path:filename>')
    def serve_video_files(filename):
        nonlocal media_directory
        return send_from_directory(media_directory, filename)
  
    videos_search_engine = VideoSearch(cfg=cfg)
    videos_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path) # Needs actual models path

    common_socket_events = CommonSocketEvents(socketio)

    videos_file_manager = file_manager.FileManager(
            cfg=cfg,
            media_directory=media_directory,
            engine=videos_search_engine,
            module_name="videos",
            media_formats=cfg.videos.media_formats,
            socketio=socketio,
            db_schema=db_models.VideosLibrary,
        )
    
    # Create metadata search engine
    metadata_search_engine = MetadataSearch(engine=videos_search_engine)
  
    def update_model_ratings(files_list):
        print('update_model_ratings')

        # files_list_hash_map = {file_path: videos_search_engine.get_file_hash(file_path) for file_path in files_list}
        # hash_list = list(files_list_hash_map.values())

        # # Update file paths in the DB if none
        # for file_path, file_hash in files_list_hash_map.items():
        #     db_item = db_models.VideosLibrary.query.filter_by(hash=file_hash).first()
        #     if db_item is None or db_item.file_path is None:
        #         file_data = {
        #             "hash": file_hash,
        #             "hash_algorithm": videos_search_engine.get_hash_algorithm(),
        #             "file_path": os.path.relpath(file_path, media_directory),
        #             "model_rating": None,
        #             "model_hash": None,
        #         }
        #         new_item = db_models.VideosLibrary(**file_data)
        #         db_models.db.session.add(new_item)

        # db_models.db.session.commit()

        # ==== ????

        # # filter out files that already have a rating in the DB
        # files_list_hash_map = {file_path: videos_search_engine.get_file_hash(file_path) for file_path in files_list}
        # hash_list = list(files_list_hash_map.values())

        # # Fetch rated files from the database in a single query
        # rated_files_db_items = db_models.VideosLibrary.query.filter(
        #     db_models.VideosLibrary.hash.in_(hash_list),
        #     db_models.VideosLibrary.model_rating.isnot(None),
        #     db_models.VideosLibrary.model_hash.is_(videos_evaluator.hash)
        # ).all()

        # # Create a list of hashes for rated files
        # rated_files_hashes = {item.hash for item in rated_files_db_items}

        # # Filter out files that already have a rating in the database
        # filtered_files_list = [file_path for file_path, file_hash in files_list_hash_map.items() if file_hash not in rated_files_hashes]
        # if not filtered_files_list: return
        

        # # Rate all files in case they are not rated or model was updated
        # embeddings = videos_search_engine.process_files(filtered_files_list, callback=embedding_gathering_callback, media_folder=media_directory) #.cpu().detach().numpy() 
        # # model_ratings = text_evaluator.predict(embeddings)

        # # Update the model ratings in the database
        # common_socket_events.show_search_status(f"Updating model ratings of files...") 
        # new_items = []
        # update_items = []
        # last_shown_time = 0
        # for ind, full_path in enumerate(filtered_files_list):
        #     print(f"Updating model ratings for {ind+1}/{len(filtered_files_list)} files.")

        #     hash = files_list_hash_map[full_path]

        #     # print('model_ratings[ind]', model_ratings[ind])
        #     model_rating = None #model_ratings[ind].mean().item()

        #     music_db_item = db_models.VideosLibrary.query.filter_by(hash=hash).first()
        #     if music_db_item:
        #         music_db_item.model_rating = model_rating
        #         music_db_item.model_hash = videos_evaluator.hash
        #         update_items.append(music_db_item)
        #     else:
        #         file_data = {
        #                 "hash": hash,
        #                 "hash_algorithm": videos_search_engine.get_hash_algorithm(),
        #                 "file_path": os.path.relpath(full_path, media_directory),
        #                 "model_rating": model_rating,
        #                 "model_hash": videos_evaluator.hash
        #         }
        #         new_items.append(db_models.VideosLibrary(**file_data))

        #     current_time = time.time()
        #     if current_time - last_shown_time >= 1:
        #         common_socket_events.show_search_status(f"Updated model ratings for {ind+1}/{len(filtered_files_list)} files.")
        #         last_shown_time = current_time     

        # # Bulk update and insert
        # if update_items:
        #         db_models.db.session.bulk_save_objects(update_items)
        # if new_items:
        #         db_models.db.session.bulk_save_objects(new_items)

        # # Commit the transaction
        # db_models.db.session.commit()

    # Create common filters instance
    common_filters = CommonFilters(
        engine=videos_search_engine,
        metadata_engine=metadata_search_engine,
        common_socket_events=common_socket_events,
        media_directory=media_directory,
        db_schema=db_models.VideosLibrary,
        update_model_ratings_func=update_model_ratings
    )

    @socketio.on('emit_videos_page_get_folders')  
    def get_folders(data):
        path = data.get('path', '')
        return videos_file_manager.get_folders(path)

    @socketio.on('emit_videos_page_get_files')
    def get_files(input_data):
        def filter_by_recommendation(all_files, text_query):
            # For now, we only need basic info from DB for sorting.
            
            all_hashes = []
            for ind, file_path in enumerate(all_files):
                common_socket_events.show_search_status(f"Filtering by recommendation: computing files hashes {ind+1}/{len(all_files)}")
                file_hash = videos_search_engine.get_file_hash(file_path)
                all_hashes.append(file_hash)

            # Model ratings update logic can be added later if needed.
            common_socket_events.show_search_status("Filtering by recommendation: loading video data from DB")
            video_data = db_models.VideosLibrary.query.with_entities(
                db_models.VideosLibrary.hash,
                db_models.VideosLibrary.user_rating,
                db_models.VideosLibrary.model_rating,
                db_models.VideosLibrary.full_play_count,
                db_models.VideosLibrary.skip_count,
                db_models.VideosLibrary.last_played
            ).filter(db_models.VideosLibrary.hash.in_(all_hashes)).all()

            keys = ['hash', 'user_rating', 'model_rating', 'full_play_count', 'skip_count', 'last_played']
            video_data_dict = [dict(zip(keys, row)) for row in video_data]
            
            # Create a map from hash to db_data for quick lookup
            hash_to_db_data = {item['hash']: item for item in video_data_dict}
            
            # Create a list of dictionaries with all necessary fields for sort_files_by_recommendation
            common_socket_events.show_search_status("Filtering by recommendation: preparing data for sorting")
            full_video_data_for_sorting = []
            for file_path, file_hash in zip(all_files, all_hashes):
                full_video_data_for_sorting.append(hash_to_db_data.get(file_hash, {
                    'hash': file_hash, 'user_rating': None, 'model_rating': None, 'full_play_count': 0, 'skip_count': 0, 'last_played': None
                }))

            common_socket_events.show_search_status("Filtering by recommendation: sorting files")
            scores = sort_files_by_recommendation(all_files, full_video_data_for_sorting)

            return scores

        # path = data.get('path', '')
        # pagination = data.get('pagination', 0)
        # limit = data.get('limit', 100)
        # text_query = data.get('text_query', None)
        # seed = data.get('seed', None)

        filters = {
            # "by_file": filter_by_file, # special sorting case when file path used as query
            "by_text": common_filters.filter_by_text, # special sorting case when text used as query, i.e. all other cases wasn't triggered
            # "file size": filter_by_file_size,
            # "length": filter_by_length,
            # "similarity": filter_by_similarity, 
            "random": common_filters.filter_by_random, 
            # "rating": filter_by_rating, 
            "recommendation": filter_by_recommendation
        }

        def get_file_info(full_path, file_hash):
            basename = os.path.basename(full_path)
            preview_path = os.path.join(os.path.dirname(full_path), basename + ".preview.png")

            # Generate preview if it does not exist
            if not os.path.exists(preview_path):
                common_socket_events.show_search_status(f"Generating preview for {basename}...")
                try:
                    generate_preview(full_path, preview_path)
                    common_socket_events.show_search_status(f"Generated preview for {basename}.")
                except Exception as e:
                    print(f"Error generating preview for {basename}: {e}")
                    common_socket_events.show_search_status(f"Failed to generate preview for {basename}. Using placeholder.")

            file_size = os.path.getsize(full_path)

            db_item = db_models.VideosLibrary.query.filter_by(hash=file_hash).first()

            last_played = "Never"    
            user_rating = None
            model_rating = None
            if db_item:
                if db_item.last_played:
                    last_played_timestamp = db_item.last_played.timestamp()
                    last_played = time_difference(last_played_timestamp, datetime.datetime.now().timestamp())  

                user_rating = db_item.user_rating
                model_rating = db_item.model_rating
            else:
                pass
                # raise Exception(f"File '{full_path}' with hash '{file_hash}' not found in the database.")

            return {
                    #"full_path": full_path,
                    #"file_path": os.path.relpath(full_path, media_directory), # Relative path for serving
                    "preview_path": os.path.relpath(preview_path, media_directory), # Relative path for serving preview
                    "base_name": basename,
                    #"hash": file_hash,
                    "user_rating": user_rating,
                    "model_rating": model_rating,
                    "file_size": convert_size(file_size),
                    "resolution": "N/A", # Placeholder, implement proper metadata extraction later (e.g., using ffprobe)
                    "length": "N/A",     # Placeholder, implement proper metadata extraction later (e.g., using ffprobe)
                    "last_played": last_played,
                }
        
        #path, pagination, limit, text_query, seed, filters, get_file_info, update_model_ratings
        input_params = input_data.copy()
        input_params.update({
            "filters": filters,
            "get_file_info": get_file_info,
            "update_model_ratings": update_model_ratings,
        })
        return videos_file_manager.get_files(**input_params)

    @socketio.on('emit_videos_page_open_file_in_folder')
    def open_file_in_folder(file_path):
        file_path = os.path.normpath(file_path)
        print(f'Opening file with path: "{file_path}"')
        
        # Assuming file_path is the full path to the file
        folder_path = os.path.dirname(file_path)
        if os.path.isfile(file_path):
            if sys.platform == "win32":    # Windows
                subprocess.run(["explorer", "/select,", file_path], check=True)
            elif sys.platform == "darwin":    # macOS
                subprocess.run(["open", "-R", file_path], check=True)
            else:    # Linux and other Unix-like OS
                # Convert the file path to an absolute path
                abs_path = os.path.abspath(file_path)

                 # Check for the file manager and use the appropriate command on Linux
                if os.environ.get('XDG_CURRENT_DESKTOP') in ['GNOME', 'Unity']:
                    subprocess.run(['nautilus', '--no-desktop', abs_path])
                elif os.environ.get('XDG_CURRENT_DESKTOP') == 'KDE':
                    subprocess.run(['dolphin', '--select', abs_path])
                else:
                    print("Unsupported desktop environment. Please add support for your file manager.")
        else:
            print("Error: File does not exist.")

    @socketio.on('emit_videos_page_send_file_to_trash')
    def send_file_to_trash(file_path):
        if os.path.isfile(file_path):
            send2trash.send2trash(file_path)
            '''if sys.platform == "win32":    # Windows
                # Move the file to the recycle bin
                subprocess.run(["cmd", "/c", "del", "/q", "/f", file_path], check=True)
            elif sys.platform == "darwin":    # macOS
                # Move the file to the trash
                subprocess.run(["trash", file_path], check=True)
            else:    # Linux and other Unix-like OS
                # Move the file to the trash
                subprocess.run(["gio", "trash", file_path], check=True)'''
        else:
            print("Error: File does not exist.")    

    @socketio.on('emit_videos_page_video_start_playing')
    def video_start_playing(file_path):
        """
        Updates the last_played timestamp for a video in the database.
        """
        print(f"Updating last_played for video: {file_path}")
        nonlocal media_directory
        print(f"Video started playing: {file_path}")

        full_path = os.path.join(media_directory, file_path)
        video_hash = videos_search_engine.get_file_hash(full_path)
        if video_hash is None:
            raise Exception(f"Could not compute hash for file: {full_path}")

        video = db_models.VideosLibrary.query.filter_by(hash=video_hash).first()
        if video:
            video.last_played = datetime.datetime.now()
            # Also, increment full_play_count if needed, or handle skip here based on player events.
            # For now, just last_played as requested.
            db_models.db.session.commit()
        else:
            # Create a minimal record in the database if video not found
            print(f"Video with hash {video_hash} not found in DB, creating minimal record.")
            new_video = db_models.VideosLibrary(
                hash=video_hash,
                hash_algorithm=videos_search_engine.get_hash_algorithm(),
                file_path=os.path.relpath(full_path, media_directory),
                last_played=datetime.datetime.now()
            )
            db_models.db.session.add(new_video)
            db_models.db.session.commit()

    @socketio.on('emit_videos_page_get_external_metadata_file_content')
    def get_external_metadata_file_content(file_path):
        """
        Reads the content of the external .meta file associated with a file.
        If the file does not exist, returns an empty string.
        """
        nonlocal media_directory
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"

        content = ""
        try:
            if os.path.exists(metadata_file_path):
                with open(metadata_file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            print(f"Read external metadata for {file_path}")
        except Exception as e:
            print(f"Error reading external metadata for {file_path}: {e}")

        return {"content": content, "file_path": file_path}

    @socketio.on('emit_videos_page_save_external_metadata_file_content')
    def save_external_metadata_file_content(data):
        """
        Saves the provided content to the .meta file associated with a file.
        """
        nonlocal media_directory
        file_path = data['file_path']
        metadata_content = data['metadata_content']
        
        full_path = os.path.join(media_directory, file_path)
        metadata_file_path = full_path + ".meta"

        try:
            # Ensure the directory exists before writing the file
            os.makedirs(os.path.dirname(metadata_file_path), exist_ok=True)
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(metadata_content)
            print(f"Saved metadata for {file_path}")
        except Exception as e:
            print(f"Error saving metadata for {file_path}: {e}")

    # ----------------------------------------
    # HSL Streaming test
    # ----------------------------------------

    # This section requires a lot of refactoring and rethinking, for now it just works, somehow
    

    # Store active transcoding processes
    active_transcodings = {}

    @app.route('/stream/<stream_id>/master.m3u8')
    def stream_video(stream_id):
        """Stream a video with the given stream_id"""
        if stream_id not in active_transcodings:
            return "Stream not found", 404
            
        # If stream is already set up, just serve the playlist
        if 'temp_dir' in active_transcodings[stream_id]:
            master_playlist = os.path.join(active_transcodings[stream_id]['temp_dir'], "master.m3u8")
            if os.path.exists(master_playlist):
                return send_from_directory(active_transcodings[stream_id]['temp_dir'], "master.m3u8")
        
        # Otherwise, set up the stream
        video_path = active_transcodings[stream_id]['path']
        print(f"Starting stream for {stream_id} from {video_path}")
        
        # Create a temp directory for HLS files
        temp_dir = tempfile.mkdtemp()
        active_transcodings[stream_id]['temp_dir'] = temp_dir
        
        # Create master playlist path
        master_playlist = os.path.join(temp_dir, "master.m3u8")
        
        # Get video duration and other metadata
        try:
            metadata_cmd = [
                'ffprobe', 
                '-v', 'error', 
                '-show_entries', 'format=duration', 
                '-of', 'json', 
                video_path
            ]
            metadata_output = subprocess.check_output(metadata_cmd).decode('utf-8').strip()
            import json
            metadata = json.loads(metadata_output)
            duration = float(metadata['format']['duration'])
            print(f"Video duration: {duration} seconds")
            active_transcodings[stream_id]['duration'] = duration
        except Exception as e:
            print(f"Error getting video metadata: {e}")
            duration = 0
        
        # Build FFmpeg command for HLS with optimized settings
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', video_path,
            '-c:v', 'libx264',       # Video codec
            '-preset', 'ultrafast',  # Faster encoding
            '-tune', 'zerolatency',  # Reduce latency
            '-c:a', 'aac',           # Audio codec
            '-ar', '44100',          # Audio sample rate
            '-ac', '2',              # Stereo audio
            '-b:a', '128k',          # Audio bitrate
            '-f', 'hls',             # HLS format
            '-hls_time', '2',        # 2-second segments
            '-hls_list_size', '0',   # Keep all segments
            '-hls_flags', 'independent_segments+split_by_time+append_list', # append_list might be redundant with vod but often harmless
            '-hls_segment_type', 'mpegts',
            '-hls_playlist_type', 'event',
            '-force_key_frames', 'expr:gte(t,n_forced*2)', # Keyframe every 2s
            '-g', '48',              # GOP size
            '-start_number', '0',    # Start segment numbering at 0
            '-hls_segment_filename', os.path.join(temp_dir, f'segment_%03d.ts'),
            master_playlist
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE
        )
        
        # Store the process
        active_transcodings[stream_id]['process'] = process
        
        # Wait for the master playlist to be created (with a timeout)
        start_time = time.time()
        while not os.path.exists(master_playlist) and time.time() - start_time < 10:
            time.sleep(0.1)
            # Check if process died
            if process.poll() is not None:
                stderr_output = ""
                if process.stderr: # Check if stderr is available
                    try:
                        stderr_output = process.stderr.read().decode('utf-8', errors='ignore')
                    except Exception as e:
                        print(f"Error reading ffmpeg stderr: {e}")
                # You might want to log stderr_output here if it contains useful error info
                # print(f"FFmpeg stderr: {stderr_output}") 
                return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500
        
        if not os.path.exists(master_playlist):
            return "Timeout waiting for playlist to be created", 500
        
        # Wait for initial buffer segments to be created before serving
        segments_ready = 0
        segment_pattern = os.path.join(temp_dir, "segment_*.ts")
        while segments_ready < 3 and time.time() - start_time < 10:  # Wait for 3 segments or 10s max
            segments_ready = len(glob.glob(segment_pattern))
            if segments_ready >= 3:
                break
            time.sleep(0.2)
            if process.poll() is not None:
                stderr_output = ""
                if process.stderr: # Check if stderr is available
                    try:
                        stderr_output = process.stderr.read().decode('utf-8', errors='ignore') # Added errors='ignore'
                    except Exception as e:
                        print(f"Error reading ffmpeg stderr: {e}")
                # print(f"FFmpeg stderr: {stderr_output}")
                return f"Error starting stream: FFmpeg exited with code {process.returncode}", 500


        print(f"Starting playback with {segments_ready} segments ready")
        
        # Serve the m3u8 file
        return send_from_directory(temp_dir, "master.m3u8")

    @app.route('/stream/<stream_id>/<path:filename>')
    def stream_segment(stream_id, filename):
        """Serve HLS segment files"""
        if stream_id not in active_transcodings or 'temp_dir' not in active_transcodings[stream_id]:
            return "Stream not found", 404
        
        temp_dir = active_transcodings[stream_id]['temp_dir']
        return send_from_directory(temp_dir, filename)

    @socketio.on('emit_videos_page_start_streaming')
    def start_streaming(file_path):
        """Start streaming a video file"""
        # Make file_path absolute if it's not already
        if not os.path.isabs(file_path):
            file_path = os.path.join(media_directory, file_path)
        
        # Verify file exists
        if not os.path.isfile(file_path):
            print(f"Error: File not found at {file_path}")
            return {"error": "File not found"}
            
        # Generate unique stream ID
        stream_id = str(uuid.uuid4())
        
        # Store info about the stream
        active_transcodings[stream_id] = {
            'path': file_path,
            'process': None,
            'start_time': time.time()
        }
        
        # Return stream URL to client
        return {
            'stream_id': stream_id,
            'stream_url': f'/stream/{stream_id}/master.m3u8'
        }
    
    @socketio.on('emit_videos_page_stop_streaming')
    def stop_streaming(stream_id):
        """Stop streaming a video file and clean up resources"""
        if stream_id in active_transcodings:
            print(f"Stopping stream {stream_id}")
            
            # Clean up the resources
            cleanup_transcoding(stream_id)
            
            return {"status": "success", "message": "Stream stopped and resources cleaned up"}
        else:
            print(f"Stream {stream_id} not found or already stopped")
            return {"status": "error", "message": "Stream not found"}

    def cleanup_transcoding(stream_id):
        """Clean up temporary files and processes"""
        if stream_id in active_transcodings:
            print(f"Cleaning up transcoding for stream {stream_id}")
            
            # Terminate the process
            if 'process' in active_transcodings[stream_id] and active_transcodings[stream_id]['process']:
                try:
                    active_transcodings[stream_id]['process'].terminate()
                    print(f"Process for stream {stream_id} terminated")
                except Exception as e:
                    print(f"Error terminating process: {e}")
            
            # Close error log file
            if 'error_log' in active_transcodings[stream_id]:
                try:
                    active_transcodings[stream_id]['error_log'].close()
                except:
                    pass
            
            # Remove temp directory
            if 'temp_dir' in active_transcodings[stream_id]:
                try:
                    shutil.rmtree(active_transcodings[stream_id]['temp_dir'])
                    print(f"Removed temp directory for stream {stream_id}")
                except Exception as e:
                    print(f"Error removing temp directory: {e}")
            
            # Remove from active transcodings
            del active_transcodings[stream_id]
            print(f"Stream {stream_id} completely cleaned up")
  
    # Cleanup old transcodings periodically
    def cleanup_old_transcodings():
        """Periodically check for and clean up old transcoding processes"""
        while True:  # Continuous loop
            try:
                # Print some debug info
                #print(f"Cleanup thread running, active streams: {len(active_transcodings)}")
                
                current_time = time.time()
                for stream_id in list(active_transcodings.keys()):
                    # Check if the stream has been active for more than 1 hour
                    if current_time - active_transcodings[stream_id]['start_time'] > 3600:  # 1 hour timeout
                        print(f"Cleaning up expired stream: {stream_id}")
                        cleanup_transcoding(stream_id)
                        
                # Sleep for a specified interval before checking again
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
                time.sleep(60)  # If there's an error, wait a bit before retrying
                
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_transcodings)
    cleanup_thread.daemon = True
    cleanup_thread.start()