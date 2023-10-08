from flask import Flask, render_template, send_from_directory
from flask_socketio import SocketIO
from pydub import AudioSegment
import base64

import os
import eyed3
import hashlib

import datetime

from tqdm import tqdm
import numpy as np

import src.db_models


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

  audiofile = eyed3.load(file_path)
  #print('tag:', dir(file.tag))
  # Check if the file has tag information
  if audiofile.tag is not None:
    metadata['title'] = audiofile.tag.title or "N/A"
    metadata['artist'] = audiofile.tag.artist or "N/A"
    metadata['album'] = audiofile.tag.album or "N/A"
    metadata['track_num'] = audiofile.tag.track_num[0] if audiofile.tag.track_num else "N/A"
    metadata['genre'] = audiofile.tag.genre.name if audiofile.tag.genre else "N/A"
    metadata['date'] = str(audiofile.tag.getBestDate()) if audiofile.tag.getBestDate() else "N/A"
  else:
    metadata['title'] = "N/A"
    metadata['artist'] = "N/A"
    metadata['album'] = "N/A"
    metadata['track_num'] = "N/A"
    metadata['genre'] = "N/A"
    metadata['date'] = "N/A"

  metadata['duration'] = audiofile.info.time_secs #(seconds)
  metadata['bitrate'] = audiofile.info.bit_rate_str #(kbps)

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
  
  

def init_socket_events(socketio, predictor, app=None):
  # Determine the absolute path to the media file
  media_directory = '/media/alex/Expansion/Music'
  music_list = []
  music_ratings = None
  # necessary to allow web application access to music files
  @app.route('/media/<path:filename>')
  def serve_media(filename):
    nonlocal media_directory
    return send_from_directory(media_directory, filename)

  @socketio.on('emit_music_page_refresh_music_library')
  def refresh_music_library():
    #nonlocal music_list

    folder = "/media/alex/Expansion/Music/"
  
    music_files = []
    for root, dirs, files in os.walk(folder):
      for file in files:
        if file.endswith(".mp3"):
          full_path = os.path.join(root, file)
          music_files.append(full_path)

    #music_list = []
    for full_path in tqdm(music_files):   
      try: 
        url_path = 'media/' + full_path.replace(folder, '')
        audiofile_data = get_audiofile_data(full_path, url_path)

        # Check if a row with the same primary key (hash) exists
        existing_music = src.db_models.MusicLibrary.query.get(audiofile_data['hash'])

        if existing_music:
          # Update the existing row with the new data
          for key, value in audiofile_data.items():
            setattr(existing_music, key, value)
        else:
          # Create a new row
          new_music = src.db_models.MusicLibrary(**audiofile_data)
          src.db_models.db.session.add(new_music)

        # Commit the changes to the database
        src.db_models.db.session.commit()
      except Exception as ex:
        print('Something went wrong with', full_path)
        print(ex)


  @socketio.on('emit_music_page_get_music_list')
  def get_music_list():
    nonlocal music_list

    music_list = src.db_models.MusicLibrary.query.all()
    music_list = [music.as_dict() for music in music_list]
    print(music_list[0].keys())

    #music_list = get_music_list_metadata()
    socketio.emit('emit_music_page_send_music_list', music_list)  

  @socketio.on('emit_music_page_get_next_song')
  def request_new_song():
    if len(music_list)>0:
      # Convert the list to a NumPy array to work with numerical values
      not_none_scores = np.array([music['user_rating'] for music in music_list if music['user_rating'] is not None])
      print('Not none ratings:', list(not_none_scores))
      # Calculate the median of the existing values
      median_value = np.median(not_none_scores)
      print('Median song rating:', median_value)

      # Replace None values with the calculated median
      scores = np.array([median_value if music['user_rating'] is None else music['user_rating'] for music in music_list])
      scores = np.minimum(0.1, scores) # we make a minimum small value to the rating so songs with 0 rating have some small chance to be played
      scores = (scores / 10) ** 2 # normalize scores and make songs with high rating much more likely to occur

      # Generate a random index based on the weights
      sampled_index = np.random.choice(len(scores), p=scores/np.sum(scores))

      #ind = np.random.randint(len(music_list))
      socketio.emit('emit_music_page_send_next_song', music_list[sampled_index])  

  @socketio.on('emit_music_page_set_song_rating')
  def set_song_rating(data):
    song_hash = data['hash'] 
    song_score = data['score']

    print('Set song rating:', song_hash, song_score)

    song = src.db_models.MusicLibrary.query.get(song_hash)
    song.user_rating = int(song_score)
    src.db_models.db.session.commit()

    music = next((item for item in music_list if item['hash'] == song_hash), None)
    music['user_rating'] = song_score
  
if __name__ == "__main__":
  print('start')
  hash_1 = calculate_audiodata_hash('src/music_1.mp3')
  hash_2 = calculate_audiodata_hash('src/music_2.mp3')
  print(hash_1, hash_2, hash_1 == hash_2)
        

        
        