from PIL import Image # Still used for get_audiofile_data if it extracts cover art
import requests
from transformers import AutoFeatureExtractor, ClapConfig, ClapModel, ClapProcessor
import torch
import numpy as np
from tqdm import tqdm
import os
import pickle
import hashlib
import datetime
import io
import time
import cv2
import imageio
import torchaudio
from tinytag import TinyTag
import base64
# from huggingface_hub import snapshot_download # Moved to BaseSearchEngine

import src.scoring_models
import pages.music.db_models as db_models
import pages.file_manager as file_manager
from src.base_search_engine import BaseSearchEngine # Import the base class
from src.model_manager import ModelManager # Still needed for MusicEvaluator

def get_audiofile_data(file_path):
    """
    Extracts metadata from an audio file using TinyTag.
    Also attempts to extract album art and convert it to base64.
    """
    metadata = {}
    try:
        tag = TinyTag.get(file_path, image=True)

        metadata['title'] = tag.title or "N/A"
        metadata['artist'] = tag.artist or "N/A"
        metadata['album'] = tag.album or "N/A"
        metadata['track_num'] = tag.track if tag.track else "N/A"
        metadata['genre'] = tag.genre if tag.genre else "N/A"
        metadata['date'] = str(tag.year) if tag.year else "N/A"

        metadata['duration'] = tag.duration # seconds
        metadata['bitrate'] = tag.bitrate # kbps

        metadata['lyrics'] = tag.extra.get('lyrics', "")

        img = tag.get_image()
        if img is not None:
            base64_image = base64.b64encode(img).decode('utf-8')
            metadata['image'] = f"data:image/png;base64,{base64_image}" # Assuming PNG, adjust if needed
        else:
            metadata['image'] = None
    except Exception as e:
        print(f"Error extracting metadata from {file_path}: {e}")
        # Populate with N/A or None on error for consistency
        for key in ['title', 'artist', 'album', 'track_num', 'genre', 'date', 'duration', 'bitrate', 'lyrics', 'image']:
            if key not in metadata: metadata[key] = None
        if metadata['duration'] is None: metadata['duration'] = 0 # Ensure duration is numeric for sorting

    return metadata

class MusicSearch(BaseSearchEngine):
    def __init__(self, cfg=None):
        super().__init__(cfg) # Call base class __init__
        self.cfg = cfg # Store cfg for reading parameters
        self.tokenizer = None # Will be set in _load_model_and_processor

    @property
    def model_name(self) -> str:
        # Use cfg.music.embedding_model for dynamic model selection
        if self.cfg is None or not hasattr(self.cfg, 'music') or not hasattr(self.cfg.music, 'embedding_model'):
            raise ValueError("Music embedding model not specified in config.")
        return self.cfg.music.embedding_model # e.g., "laion/clap-htsat-fused"
    
    @property
    def cache_prefix(self) -> str:
        return 'music'
        
    def _get_metadata_function(self):
        return get_audiofile_data

    def _get_db_model_class(self):
        return db_models.MusicLibrary
    
    def _get_model_hash_postfix(self):
        return ""

    def _load_model_and_processor(self, local_model_path: str):
        """
        Loads the CLAP model and processor from the local path.
        Ensures the model is loaded to CPU before being wrapped by ModelManager.
        """
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"MusicSearch: Loading model to CPU first...")
            self.model = ClapModel.from_pretrained(local_model_path, local_files_only=True).cpu() # Load to CPU
            self.processor = ClapProcessor.from_pretrained(local_model_path, local_files_only=True)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(local_model_path, local_files_only=True)
            self.embedding_dim = self.model.config.text_config.projection_dim # CLAP's projection dim

        except Exception as e:
            print(f"ERROR: Failed to load CLAP model from '{local_model_path}'. The download might be incomplete or corrupted.")
            print(f"Error details: {e}")
            raise RuntimeError(f"Failed to load required model: {self.model_name}") from e

    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Reads and preprocesses a single audio file, then generates its embedding.
        """
        waveform, sample_rate = self._read_audio(file_path)
        if waveform is None: 
            raise Exception(f"Failed to read audio file: {file_path}")

        # Convert waveform to 48kHz if not (required by CLAP)
        if sample_rate != 48000:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=48000)(waveform)
            sample_rate = 48000

        # Convert to mono if not
        if waveform.shape[0] > 1: # If stereo or multi-channel
            waveform = waveform.mean(dim=0, keepdim=False)
        
        # Use the wrapped model (self.model is now ModelManager instance)
        model_instance = self.model
        
        inputs_audio = self.feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt").to(model_instance._device)
        
        with torch.no_grad():
            outputs = model_instance.get_audio_features(**inputs_audio)

        # CLAP audio features are typically normalized by default but can be re-normalized if needed
        audio_embeds = outputs # Assume CLAP returns normalized features or handle here if not
        
        return audio_embeds
  
    def _read_audio(self, audio_path: str):
        """Helper method to read audio files using torchaudio."""
        try:
            waveform, sample_rate = torchaudio.load(audio_path)
            return waveform, sample_rate
        except Exception as e:
            print(f"Error reading audio file {audio_path}: {e}")
            return None, None

    def process_audio(self, audio_files: list[str], callback=None, media_folder: str = None) -> torch.Tensor:
        """
        Public method for audio processing, now calls the base class's logic.
        Kept for backward compatibility.
        """
        return super().process_files(audio_files, callback=callback, media_folder=media_folder)
  
    def process_text(self, text: str) -> torch.Tensor:
        """
        Processes a text query to generate its embedding using the CLAP text encoder.
        """
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        model_instance = self.model # Triggers loading to GPU if needed
        
        inputs_text = self.processor(text=text, padding=True, return_tensors="pt").to(model_instance._device)

        with torch.no_grad():
            outputs = model_instance.get_text_features(**inputs_text)

        # CLAP text features are typically normalized by default but can be re-normalized if needed
        text_embeds = outputs # Assume CLAP returns normalized features or handle here if not
        
        # TODO: Find out why .squeeze(0) is needed here, it was working without it before refactoring
        return text_embeds.squeeze(0)  # Return as 1D tensor

  # The 'compare' method from the original MusicSearch is largely handled by BaseSearchEngine.compare now.
  # CLAP's compare has specific logit scales, so BaseSearchEngine.compare handles this.

# Create scoring model singleton class so it easily accessible from other modules
class MusicEvaluator(src.scoring_models.Evaluator):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(MusicEvaluator, cls).__new__(cls)
        return cls._instance

    def __init__(self, embedding_dim=768, rate_classes=11):
        if not hasattr(self, '_initialized'):
            super(MusicEvaluator, self).__init__(embedding_dim, rate_classes)
            self._initialized = True


# --- Testing Section ---
if __name__ == "__main__":
    from omegaconf import OmegaConf
    import tempfile
    import glob
    import colorama
    
    colorama.init()

    print("--- Running MusicSearch Engine Test ---")

    # Create a dummy config for testing
    dummy_cfg_dict = {
        'main': {
            'embedding_models_path': './models',
            'cache_path': './cache'
        },
        'music': {
            'embedding_model': "laion/clap-htsat-fused",
            'media_formats': ['.mp3', '.wav']
        }
    }
    cfg = OmegaConf.create(dummy_cfg_dict)

    os.makedirs(cfg.main.embedding_models_path, exist_ok=True)
    os.makedirs(cfg.main.cache_path, exist_ok=True)

    # Create dummy audio files for testing (requires pydub and tinytag)
    path = os.path.dirname(os.path.abspath(__file__))
    test_audio_dir = os.path.join(path, "engine_test_data")
    os.makedirs(test_audio_dir, exist_ok=True)
    
    # Generate a simple 5-second sine wave audio file
    import soundfile as sf
    samplerate_test = 48000 # CLAP prefers 48kHz
    duration_test = 5 # seconds
    frequency_test_1 = 440 # Hz
    frequency_test_2 = 880 # Hz
    
    t = np.linspace(0., duration_test, int(samplerate_test * duration_test), endpoint=False)
    waveform1 = 0.5 * np.sin(2. * np.pi * frequency_test_1 * t)
    waveform2 = 0.5 * np.sin(2. * np.pi * frequency_test_2 * t)
    
    dummy_audio_path1 = os.path.join(test_audio_dir, "262274__the_sound_side__abide-with-me-opera-vocals.wav")
    dummy_audio_path2 = os.path.join(test_audio_dir, "643360__duisterwho__she-has-lost-her-mind-vocal.wav")
    dummy_audio_path3 = os.path.join(test_audio_dir, "717297__curtiswcole__cardinal-white-noise.wav")
    dummy_audio_path4 = os.path.join(test_audio_dir, "803535__clod111__baby-daughter-crying.wav")

    # --- Initialize the model ---
    try:
        print("\nInitializing MusicSearch engine...")
        music_search_engine = MusicSearch(cfg=cfg)
        music_search_engine.initiate(models_folder=cfg.main.embedding_models_path, cache_folder=cfg.main.cache_path)
        print(f"MusicSearch engine initialized. Model hash: {music_search_engine.model_hash}")
        print(f"Model on device: {music_search_engine.model.device}")
    except Exception as e:
        print(f"FATAL: MusicSearch engine initiation failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

    # --- Test file processing ---
    print("\n--- Test file processing (embedding generation and caching) ---")
    test_files = [dummy_audio_path1, dummy_audio_path2, dummy_audio_path3, dummy_audio_path4]
    
    def test_callback(num_processed, num_total):
        print(f"Processed {num_processed}/{num_total} files...")

    try:
        print(f"{colorama.Fore.CYAN}Processing files for the first time (should generate embeddings):{colorama.Style.RESET_ALL}")
        embeddings = music_search_engine.process_audio(
            test_files, 
            callback=test_callback, 
            media_folder=test_audio_dir
        )
        print(f"Generated embeddings shape: {embeddings.shape}")
        assert embeddings.shape[0] == 4, f"Expected 3 embeddings, got {embeddings.shape[0]}"
        assert embeddings.shape[1] == music_search_engine.embedding_dim, f"Expected embedding dim {music_search_engine.embedding_dim}, got {embeddings.shape[1]}"
        print(f"{colorama.Fore.GREEN}First processing successful.{colorama.Style.RESET_ALL}")

        print(f"\n{colorama.Fore.CYAN}Processing files again (should use cache):{colorama.Style.RESET_ALL}")
        music_search_engine._fast_cache = {} 
        embeddings_cached = music_search_engine.process_audio(
            test_files, 
            callback=test_callback, 
            media_folder=test_audio_dir
        )
        print(f"Generated embeddings shape (cached): {embeddings_cached.shape}")
        assert torch.allclose(embeddings, embeddings_cached), "Cached embeddings do not match original"
        print(f"{colorama.Fore.GREEN}Cached processing successful.{colorama.Style.RESET_ALL}")

    except Exception as e:
        print(f"{colorama.Fore.RED}File processing test FAILED: {e}{colorama.Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

    # --- Test text processing and comparison ---
    print("\n--- Test text processing and comparison ---")
    try:
        # Define realistic search queries
        search_queries = [
            "woman singing a song acapella",
            "baby crying",
            "white noise",
            #"radio host talking clearly",
            "she lost her mind",
        ]

        for query in search_queries:
            print(f"\nProcessing query: '{query}'")
            query_embedding = music_search_engine.process_text(query)
            print(f"Query embedding shape: {query_embedding.shape}")
            
            scores_data = music_search_engine.compare(embeddings, query_embedding)
            
            print(f"Scores for '{query}'")

            # Combine file paths with their scores
            results = []
            for i, file_path in enumerate(test_files):
                file_name = os.path.basename(file_path)
                score = scores_data[i] if i < len(scores_data) else 0.0
                results.append((file_name, score))

            # Sort results by score in descending order
            results.sort(key=lambda item: item[1], reverse=True)

            # Print sorted results
            for file_name, score in results:
                print(f"  {file_name}: {score:.4f}")

    except Exception as e:
        print(f"{colorama.Fore.RED}Text processing or comparison test FAILED: {e}{colorama.Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

    # --- Test ModelManager lazy loading/unloading ---
    print("\n--- Testing ModelManager lazy loading ---")
    print(f"Model manager loaded: {music_search_engine.model._loaded}")
    print(f"Model device: {music_search_engine.model._model.device}")
    assert music_search_engine.model._loaded and music_search_engine.model._model.device.type == 'cuda', "Model not loaded to GPU after use."
    
    print("Waiting for idle timeout (120 seconds + 30s cleanup check)...")
    time.sleep(160)
    
    print(f"Model manager loaded after idle: {music_search_engine.model._loaded}")
    assert not music_search_engine.model._loaded, "Model was not unloaded after idle period."
    print(f"{colorama.Fore.GREEN}ModelManager unloading test successful.{colorama.Style.RESET_ALL}")

    print("\nRe-using model to trigger reload...")
    dummy_text = "This is a test text to trigger model reloading."
    query_embedding = music_search_engine.process_text(dummy_text)

    print(f"Model manager loaded after re-use: {music_search_engine.model._loaded}")
    print(f"Model device after re-use: {music_search_engine.model._model.device}")
    assert music_search_engine.model._loaded and music_search_engine.model._model.device.type == 'cuda', "Model did not reload to GPU on re-use."
    print(f"{colorama.Fore.GREEN}ModelManager reloading test successful.{colorama.Style.RESET_ALL}")

    # Shut down ModelManager gracefully at the end of tests
    ModelManager.shutdown()
    print("\n--- MusicSearch Engine Test Completed ---")