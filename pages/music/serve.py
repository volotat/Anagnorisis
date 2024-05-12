from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from pydub import AudioSegment
import base64

import os
#import eyed3
import hashlib

import datetime
from dateutil.relativedelta import relativedelta

from tqdm import tqdm
import numpy as np
import traceback
import pages.music.db_models as db_models

from tinytag import TinyTag
#import llm_engine
from TTS.api import TTS
from unidecode import unidecode
import scipy.io.wavfile
import io
import src.scoring_models

from omegaconf import OmegaConf

from mutagen.id3 import ID3, ID3NoHeaderError, USLT

# Function to calculate the SHA-256 hash of audio data
def calculate_audiodata_hash(mp3_file_path):
    # Load the MP3 file using pydub
    audio = AudioSegment.from_file(mp3_file_path)

    # Extract the raw audio data as bytes
    audio_data = audio.raw_data

    # Initialize the hash object
    audio_hash = hashlib.sha256()

    # Update the hash with the audio data
    audio_hash.update(audio_data)

    # Return the hexadecimal representation of the hash
    return audio_hash.hexdigest()

def get_audiofile_data(file_path, url_path):
  metadata = {
    'file_path': file_path,
    'url_path': url_path,
    'hash': calculate_audiodata_hash(file_path),
  }

  '''tag.album         # album as string
  tag.albumartist   # album artist as string
  tag.artist        # artist name as string
  tag.audio_offset  # number of bytes before audio data begins
  tag.bitdepth      # bit depth for lossless audio
  tag.bitrate       # bitrate in kBits/s
  tag.comment       # file comment as string
  tag.composer      # composer as string 
  tag.disc          # disc number
  tag.disc_total    # the total number of discs
  tag.duration      # duration of the song in seconds
  tag.filesize      # file size in bytes
  tag.genre         # genre as string
  tag.samplerate    # samples per second
  tag.title         # title of the song
  tag.track         # track number as string
  tag.track_total   # total number of tracks as string
  tag.year          # year or date as string'''

  #audiofile = eyed3.load(file_path)

  tag = TinyTag.get(file_path, image=True)

  metadata['title'] = tag.title or "N/A"
  metadata['artist'] = tag.artist or "N/A"
  metadata['album'] = tag.album or "N/A"
  metadata['track_num'] = tag.track if tag.track else "N/A"
  metadata['genre'] = tag.genre if tag.genre else "N/A"
  metadata['date'] = str(tag.year) if tag.year else "N/A"

  metadata['duration'] = tag.duration #(seconds)
  metadata['bitrate'] = tag.bitrate #(kbps)

  metadata['lyrics'] = tag.extra.get('lyrics', "")

  img = tag.get_image()
  if img is not None:
    base64_image = base64.b64encode(img).decode('utf-8')

    #buffer = io.BytesIO()
    #img.save(buffer, format='PNG')
    #buffer.seek(0)
    
    #data_uri = base64.b64encode(buffer.read()).decode('ascii')
    metadata['image'] = f"data:image/png;base64,{base64_image}"
  else:
    metadata['image'] = None

    # Get all available tags and their values as a dictionary
    #tag_dict = audiofile.tag.frame_set

    # If there are multiple artists, they will be stored in a list
    #if audiofile.tag.artist:
    #    print("Artists:", ", ".join(audiofile.tag.artist))

    # If there are multiple genres, they will be stored in a list
    #if audiofile.tag.genre:
    #    print("Genres:", ", ".join(audiofile.tag.genre))

    # You can access other tag fields in a similar way

    # To print all tags and their values, you can iterate through them
    #for tag in audiofile.tag.frame_set:
    #    print(tag, ":", audiofile.tag.frame_set[tag][0])

    # If you want to access additional metadata, you can use audiofile.tag.file_info
    #print("Sample Width (bits):", audiofile.tag.file_info.sample_width)
    #print("Channel Mode:", audiofile.tag.file_info.mode)

    # To print the entire tag as a dictionary
    #print("Tag Dictionary:", audiofile.tag.frame_set)

    #for frame in audiofile.tag.frameiter(["TXXX"]):
    #  print(f"{frame.description}: {frame.text}")
  
  return metadata

def get_music_list_metadata():
  return []
   
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def time_difference(timestamp1, timestamp2):
    # Convert timestamps to datetime
    dt1 = datetime.datetime.fromtimestamp(timestamp1)
    dt2 = datetime.datetime.fromtimestamp(timestamp2)

    # Calculate difference
    diff = relativedelta(dt2, dt1)

    # Prepare readable form
    readable_form = []
    if diff.years > 0:
        readable_form.append(f"{diff.years} years")
    if diff.months > 0:
        readable_form.append(f"{diff.months} months")
    if diff.days > 0:
        readable_form.append(f"{diff.days} days")
    if diff.hours > 0:
        readable_form.append(f"{diff.hours} hours")
    if diff.minutes > 0:
        readable_form.append(f"{diff.minutes} minutes")

    # Return difference in a readable form
    return " ".join(readable_form) + " ago"

def init_socket_events(socketio, app=None, cfg=None):
  # Determine the absolute path to the media file
  media_directory = cfg.music.media_directory
  music_list = []
  music_ratings = None
  play_history = ""
  AIDJ_history = ""
  AIDJ_tts = None # TTS(cfg.music.tts_model_path)
  AIDJ_tts_index = 0

  predictor = None

  audio_embedder = src.scoring_models.AudioEmbedder("./models/MERT-v1-95M")
  audio_evaluator = src.scoring_models.Evaluator(embedding_dim=audio_embedder.embedding_dim)
  audio_evaluator.load('./models/audio_evaluator.pt')
  

  def _music_list():
    nonlocal music_list
    if (music_list is not None) and (len(music_list)>0): return music_list
    
    music_list = db_models.MusicLibrary.query.all()
    music_list = [music.as_dict() for music in music_list]

    for music in music_list:
      music["last_played"] = music["last_played"].timestamp() if music["last_played"] is not None else None

  # necessary to allow web application access to music files
  @app.route('/media/<path:filename>')
  def serve_media(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_music_page_update_music_library')
  def update_music_library():
    print('Updating music library...')
    nonlocal media_directory
  
    music_files = []
    for root, dirs, files in os.walk(media_directory):
      for file in files:
        if file.lower().endswith(tuple(cfg.music.media_formats)):
          full_path = os.path.join(root, file)
          music_files.append(full_path)

    #music_list = []
    for ind, full_path in enumerate(tqdm(music_files)):   
      try: 
        url_path = os.path.join('media', full_path.replace(media_directory, ''))
        audiofile_data = get_audiofile_data(full_path, url_path)

        # Check if a row with the same primary key (hash) exists
        existing_music = db_models.MusicLibrary.query.get(audiofile_data['hash'])

        # Get only relevant music data
        new_music_data = {}
        for key, value in audiofile_data.items():
          # check if music object has attribute:
          if hasattr(db_models.MusicLibrary, key):
            new_music_data[key] = value

        # Rate music with model
        if full_path.endswith('.mp3'):
          embedding = audio_embedder.embed_audio(full_path)
          model_rating = audio_evaluator.predict([embedding])[0]
          new_music_data['model_rating'] = int(round(model_rating))

        if existing_music:
          # Update the existing row with the new data
          for key, value in new_music_data.items():
            setattr(existing_music, key, value)
        else:
          # Create a new row
          new_music = db_models.MusicLibrary(**new_music_data)
          db_models.db.session.add(new_music)

        # Commit the changes to the database
        db_models.db.session.commit()

        percent = (ind + 1) / len(music_files) * 100
        socketio.emit("emit_music_page_update_music_library_progress", percent) 
      except Exception as ex:
        print('Something went wrong with', full_path)
        print(traceback.format_exc())


  @socketio.on('emit_music_page_get_music_list')
  def get_music_list():

    #music_list = get_music_list_metadata()
    socketio.emit('emit_music_page_send_music_list', _music_list())  

  @socketio.on('emit_music_page_set_song_play_rate')
  def request_new_song(data):
    cur_song_hash = None
    song_score_change = None
    
    if len(data) > 0:
      cur_song_hash = data[0]
      skip_score_change = data[1]

    if cur_song_hash is not None:
      song = db_models.MusicLibrary.query.get(cur_song_hash)
      if skip_score_change == 1:
        song.full_play_count += 1
      if skip_score_change == -1:
        song.skip_count += 1
      
      #song.skip_score += skip_score_change
      db_models.db.session.commit()

  @socketio.on('emit_music_page_set_song_rating')
  def set_song_rating(data):
    song_hash = data['hash'] 
    song_score = data['score']

    print('Set song rating:', song_hash, song_score)

    song = db_models.MusicLibrary.query.get(song_hash)
    song.user_rating = int(song_score)
    db_models.db.session.commit()

    music = next((item for item in _music_list() if item['hash'] == song_hash), None)
    music['user_rating'] = song_score

  @socketio.on('emit_music_page_song_start_playing')
  def song_start_playing(song_hash):
    song = db_models.MusicLibrary.query.get(song_hash)
    song.last_played = datetime.datetime.now()
    db_models.db.session.commit()

  @socketio.on('emit_music_page_set_song_skip_score')
  def set_song_skip_score(data): 
    cur_song_hash = None
    song_score_change = None
    
    if len(data) > 0:
      cur_song_hash = data[0]
      skip_score_change = data[1]

    if cur_song_hash is not None:
      song = db_models.MusicLibrary.query.get(cur_song_hash)
      if skip_score_change == 1:
        song.full_play_count += 1
      if skip_score_change == -1:
        song.skip_count += 1
      
      #song.skip_score += skip_score_change
      db_models.db.session.commit()

  def select_random_music():
    # Convert the list to a NumPy array to work with numerical values
    not_none_scores = np.array([music['user_rating'] for music in _music_list() if music['user_rating'] is not None])
    print('Not none ratings:', list(not_none_scores))
    # Calculate the median of the existing values
    median_value = np.median(not_none_scores)
    mean_value = np.mean(not_none_scores)
    print('Median song rating:', median_value)

    # Get all the user_rating values from the music list as current scores
    scores = []
    for music in _music_list():
      if music['user_rating'] is not None:
        # Use the user-based rating if it exists
        scores.append(music['user_rating'])
      elif music['model_rating'] is not None:
        # Replace None values with the model-based rating if there exists one
        scores.append(music['model_rating'])
      else:
        # Replace None values with the calculated median
        #scores.append(median_value * 0.3)
        scores.append(mean_value)

    scores = np.array(scores)
    scores = np.maximum(0.1, scores) # we make a maximum small value to the rating so songs with 0 rating have some small chance to be played
    scores = (scores / 10) ** 2 # normalize scores and make songs with high rating much more likely to occur


    #skip_score = np.array([music['skip_score'] for music in _music_list()])
    full_play_count = np.array([music['full_play_count'] for music in _music_list()])
    skip_count = np.array([music['skip_count'] for music in _music_list()])

    #### Make select probability depended on when the song was last played
    ######################################################################
    
    sorted_last_played = sorted(_music_list(), key=lambda x: 0 if x['last_played'] is None else x['last_played']) # sort by last played date ignoring Nones
    # Note that oldest elements are come first in this list, we need to reverse it to get the newest elements first to decrease their probability multiplier 
    sorted_last_played = sorted_last_played[::-1]
    sorted_last_played_indices = [sorted_last_played.index(music) for music in _music_list()]
    last_played_score = np.array(sorted_last_played_indices) / len(sorted_last_played_indices) # normalize the score to adjust select probabilities with it
  
    skip_score = sigmoid((5 + full_play_count - skip_count) / 5) # meaningful rage for skip_score [-30, 30] that result in a value ~ [0, 1]
    scores = scores * skip_score * last_played_score

    #### Generate a random index based on the calculated weights
    ############################################################

    probs = scores / np.sum(scores)
    sampled_index = np.random.choice(len(scores), p=probs)

    return _music_list()[sampled_index], probs[sampled_index] * len(scores) # return the selected music item and its select probability normalized by the number of songs for better representation

  def rescore_music(music_item):
    # Rescore the songs with current evaluator model in case it was updated
    if music_item['file_path'].endswith('.mp3'):
      music_db_entity = db_models.MusicLibrary.query.get(music_item['hash'])
      embedding = audio_embedder.embed_audio(music_item['file_path'])
      model_rating = audio_evaluator.predict([embedding])[0]

      new_rating = int(round(model_rating))
      music_db_entity.model_rating = new_rating
      music_item['model_rating'] = new_rating
      db_models.db.session.commit()

  def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    # Format the string
    time_string = ""
    if hours > 0:
        time_string += f"{int(hours)} {'hour' if int(hours) == 1 else 'hours'} "
    if minutes > 0:
        time_string += f"{int(minutes)} {'minute' if int(minutes) == 1 else 'minutes'} "
    if seconds > 0:
        time_string += f"{int(seconds)} {'second' if int(seconds) == 1 else 'seconds'}"

    return time_string.strip()

  def edit_lyrics(file_path, new_lyrics):
    try:
        # Load the audio file
        audio = ID3(file_path)
    except ID3NoHeaderError:
        # If there's no existing metadata, create a new one
        audio = ID3()

    # Remove existing lyrics (if any)
    audio.delall('USLT')
    # Add new lyrics
    audio.add(USLT(text=new_lyrics))

    # Save changes
    audio.save()

  @socketio.on('emit_music_page_update_song_info')
  def update_song_info(data):
    print('update_song_info', data)
    edit_lyrics(data['file_path'], data['lyrics'])


  @socketio.on('emit_music_page_get_files')
  def get_files(path):
    print(path)
    
    nonlocal media_directory
    files_data = []
    for file_path in os.listdir(os.path.join(media_directory, path)):
      print(file_path)

      full_path = os.path.join(media_directory, path, file_path)
      basename = os.path.basename(file_path)

      file_type = "undefined"
      if os.path.isdir(full_path): file_type = "folder"
      if os.path.isfile(full_path): file_type = "file"

      if file_type == "folder" or basename.lower().endswith(tuple(cfg.music.media_formats)):
        data = {
          "type": file_type,
          "full_path": full_path,
          "file_path": os.path.join(path, file_path),
          "base_name": basename
        }
        files_data.append(data)
      
      #if file_path.lower().endswith(tuple(cfg.music.media_formats)):
        

    socketio.emit('emit_music_page_show_files', {"files_data":files_data, "folder_path": path}) 

  #### RADIO FUNCTIONALITY
  radio_state_history = []

  def add_radio_state(state_data):
    socketio.emit('emit_music_page_add_radio_state', state_data)
    radio_state_history.append(state_data) 

  @socketio.on('emit_music_page_get_radio_history')
  def get_radio_history():
    nonlocal radio_state_history
    socketio.emit('emit_music_page_show_radio_history', radio_state_history)

  @socketio.on('emit_music_page_radio_session_start')
  def radio_session_start(data):
    nonlocal predictor, AIDJ_history, AIDJ_tts, AIDJ_tts_index, play_history, radio_state_history

    prompt = data["prompt"]
    use_AIDJ = data["use_AIDJ"]

    if AIDJ_tts is None and use_AIDJ:
      add_radio_state({
        "hidden": False,
        "head": f"System:",
        "body": f"TTS initialization..."
      })

      AIDJ_tts = TTS(cfg.music.tts_model_path)

    #if predictor is None: 
    #  add_radio_state({
    #    "hidden": False,
    #    "head": f"System:",
    #    "body": f"Loading LLM to system memory..."
    #  })
    #  predictor = llm_engine.TextPredictor(socketio)

    _music_list() # Find a better way to initialize the list if it is not yet exist

    if len(_music_list())>0:
      for i in range(30):
        #if i==0:
        #  music_item = next((x for x in _music_list() if x['hash'] == '3166336a479d2e50a397269c31991cd1998dc61027c72fdcdbaa1afd92bbbb4d'), None)
        #else:
        music_item, select_prob = select_random_music()
          
        try: 
          rescore_music(music_item)

          audiofile_data = get_audiofile_data(music_item['file_path'], music_item['url_path'])
          # Add information from meta data of audio file
          music_item['lyrics'] = audiofile_data['lyrics']
          
          add_radio_state({
            "hidden": True,
            "head": f"Next song selected:",
            "body": f"{music_item['artist']} - {music_item['title']} | {music_item['album']}"
          })

          if use_AIDJ:
            current_time = datetime.datetime.now()
            current_time_str = current_time.strftime("%A, %B %d, %Y, %H:%M")

            AIDJ_history = AIDJ_history[-1000:]

            if AIDJ_history == "":
              AIDJ_history += f"### HUMAN:\n{cfg.music.aidj_first_prompt}"
            else:
              prompt = np.random.choice(cfg.music.aidj_consecutive_prompts)
              AIDJ_history += f"### HUMAN:\n{prompt}"

            #AIDJ_history += f" Do not use hackneyed phrases like 'So, sit back, relax, and enjoy..' and others like that."
            user_rating = 'Not rated yet' if music_item['user_rating'] is None else str(music_item['user_rating']) + '/10'
            model_rating = 'Not rated yet' if music_item['model_rating'] is None else str(music_item['model_rating']) + '/10'
            AIDJ_history += f'''\nCurrent time: {current_time_str};\n\nInformation about current song:\nBand/Artist: {music_item['artist']};\nSong title: {music_item['title']};\nAlbum: {music_item['album']};\nRelease year: {music_item['date']};\nLength: {seconds_to_hms(music_item['duration'])};'''
            AIDJ_history += f'''\nFull play count: {int(music_item['full_play_count'])};\nSkip count: {int(music_item['skip_count'])};\nUser rating: {user_rating};\nModel rating: {model_rating};'''

            if len(music_item['lyrics']) > 0: AIDJ_history += f"\nLyrics:\n{music_item['lyrics']}"


            AIDJ_history += f"\n### RESPONSE:\n"

            add_radio_state({
              "hidden": True,
              "head": f"LLM Prompt generated:",
              "body": AIDJ_history
            })

            add_radio_state({
              "hidden": True,
              "head": f"System:",
              "body": f"Running LLM..."
            })

            # Predict AI DJ remark before playing the song
            llm_text = predictor.predict_from_text(AIDJ_history, temperature = cfg.music.llm_temperature)
            AIDJ_history += llm_text

            add_radio_state({
              "hidden": True,
              "head": f"LLM output:",
              "body": llm_text
            })

            if len(llm_text.strip()) > 0:
              add_radio_state({
                "hidden": True,
                "head": f"System:",
                "body": f"Generating audio based on LLM output..."
              })
              
              # Use TTS to speak the text and save it to temporary file storage
              AIDJ_tts_filename = f"static/tmp/AIDJ_{AIDJ_tts_index:04d}.wav"
              AIDJ_tts_index += 1
              AIDJ_tts.tts_to_file(llm_text, file_path=AIDJ_tts_filename, speaker_wav=cfg.music.tts_model_speaker_sample, language=cfg.music.tts_model_language)

              # Icreasing the volume of TTS output
              # Load the audio file
              sound = AudioSegment.from_wav(AIDJ_tts_filename)
              # Increase the volume
              sound = sound + 3 # plus 10db
              # Save the modified audio to the same file
              sound.export(AIDJ_tts_filename, format="wav")

              add_radio_state({
                "hidden": False,
                "image": "/static/AI.jpg",
                "head": f"AI DJ:",
                "body": f"{llm_text}",
                "audio_element": AIDJ_tts_filename
              })

          user_rating_str = 'Not rated yet' if music_item['user_rating'] is None else '★' * music_item['user_rating'] + '☆' * (10 - music_item['user_rating'])
          model_rating_str = 'Not rated yet' if music_item['model_rating'] is None else '★' * music_item['model_rating'] + '☆' * (10 - music_item['model_rating'])
          skip_multiplier = sigmoid((10 + music_item['full_play_count'] - music_item['skip_count']) / 10)

          song_info = f"\n{music_item['artist']} - {music_item['title']} | {music_item['album']}"
          song_info += f"\nUser rating: {user_rating_str}"
          song_info += f"\nModel rating: {model_rating_str}"
          lyrics_stat = "Yes" if len(music_item['lyrics']) > 0 else "No"
          song_info += f"\nFull plays: {music_item['full_play_count']}"
          # convert datetime to string
          if music_item['last_played'] is not None:
            last_played = time_difference(music_item['last_played'], datetime.datetime.now().timestamp())
          else:
            last_played = "Never"
          song_info += f"\nSkips: {music_item['skip_count']}"
          song_info += f"\nSkip multiplier: {skip_multiplier:0.4f}"
          song_info += f"\nLast played: {last_played}"
          song_info += f"\nNormalized select probability: {select_prob / 100:0.4f}"
          song_info += f"\nLyrics: {lyrics_stat}"

          add_radio_state({
            "hidden": False,
            "image": audiofile_data['image'], #link to the cover or bit64 image?
            "head": f"Now playing:",
            "body": song_info,
            "audio_element": music_item
          })

        except Exception as error:
          add_radio_state({
            "hidden": False,
            "head": f"Error:",
            "body": f"{music_item['artist']} - {music_item['title']} | {music_item['album']}\n{str(error)}\n\n{traceback.format_exc()}",
          })

    #predictor.unload_model()

    socketio.emit('emit_music_page_radio_session_end')

  @socketio.on('emit_music_page_get_path_to_music_folder')
  def get_path_to_music_folder():
    nonlocal media_directory
    socketio.emit('emit_music_page_show_path_to_music_folder', media_directory)

  @socketio.on('emit_music_page_update_path_to_music_folder')
  def update_path_to_music_folder(new_path):
    nonlocal media_directory
    cfg.music.media_directory = new_path

    # Update the configuration file
    with open('config.yaml', 'w') as file:
      OmegaConf.save(cfg, file)

    media_directory = cfg.music.media_directory
    socketio.emit('emit_music_page_show_path_to_music_folder', media_directory)

    # Show files in new folder
    get_files("")

  