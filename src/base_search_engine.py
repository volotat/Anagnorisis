# src/base_search_engine.py
import os
import torch
import pickle
import hashlib
import flask
import io
from abc import ABC, abstractmethod
from huggingface_hub import snapshot_download
from tqdm import tqdm
import pages.images.db_models as db_models
import pages.file_manager as file_manager
import numpy as np
from src.model_manager import ModelManager # Assuming ModelManager is already implemented and works

class BaseSearchEngine(ABC):
    """
    Base class for all search engines (Image, Text, Music).
    Encapsulates common functionality like model loading, caching, and file processing.
    """
    
    _instance = None # For singleton pattern implementation in subclasses if desired

    def __new__(cls, *args, **kwargs):
        # Implement a basic singleton pattern here, allowing only one instance per subclass
        if cls._instance is None:
            cls._instance = super(BaseSearchEngine, cls).__new__(cls)
            cls._instance._initialized = False # Flag to control __init__ calls
        return cls._instance

    def __init__(self, cfg=None): # cfg might be needed by subclasses in their _load_model_and_processor
        if self._initialized:
            return

        self.device = None # Will be set during model loading
        self._model_manager = None # Will wrap the actual model
        self.processor = None
        self.is_busy = False # Flag to prevent concurrent processing
        self.model_hash = None
        self.embedding_dim = None

        # Caching mechanisms
        self.cached_file_list = None
        self.cached_file_hash = None
        self.cached_metadata = None
        self._fast_cache = {} # In-memory cache for embeddings

        self._initialized = True # Mark as initialized

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Abstract property: Returns the Hugging Face model name/ID (e.g., "google/siglip-base-patch16-224")."""
        pass
    
    @property
    @abstractmethod
    def cache_prefix(self) -> str:
        """Abstract property: Returns a unique prefix for cache files (e.g., 'images', 'text', 'music')."""
        pass
    
    @abstractmethod
    def _load_model_and_processor(self, local_model_path: str):
        """
        Abstract method: Loads the specific model and processor from the local path.
        Sets self.model (the raw model object), self.processor, and self.device.
        Important: Load the raw model to CPU before wrapping with ModelManager if needed.
        """
        pass
    
    @abstractmethod
    def _get_metadata_function(self):
        """
        Abstract method: Returns the function responsible for extracting metadata
        specific to this modality (e.g., get_image_metadata, get_audiofile_data).
        """
        pass
    
    @abstractmethod
    def _process_single_file(self, file_path: str, **kwargs) -> torch.Tensor:
        """
        Abstract method: Processes a single file to generate its raw embeddings.
        This method should encapsulate modality-specific file reading and model inference.
        Returns a torch.Tensor of embeddings (e.g., [1, embedding_dim]).
        """
        pass
    
    @abstractmethod
    def _get_db_model_class(self):
        """
        Abstract method: Returns the SQLAlchemy database model class
        associated with this modality (e.g., ImagesLibrary, MusicLibrary, TextLibrary).
        """
        pass

    @abstractmethod
    def _get_model_hash_postfix(self) -> str:
        """
        Returns a short hash postfix based on the model name and version.
        Used for versioning embeddings.
        """
        return ""

    def initiate(self, models_folder: str, cache_folder: str, **kwargs):
        """
        Initializes the search engine. This is the primary entry point for setup.
        Ensures model is downloaded, loads it, and sets up caching.
        """
        # Model is already initiated via ModelManager's lazy loading
        if self._model_manager is not None:
             return
        
        # Setup model path
        if self.model_name is None:
            model_folder_name = 'dummy_model' # Fallback if model_name is not set
        else:
            model_folder_name = self.model_name.replace('/', '__') # Sanitized name for local folder

        local_model_path = os.path.join(models_folder, model_folder_name)
        
        # Ensure directories exist
        os.makedirs(models_folder, exist_ok=True)
        
        # Download model if needed
        self._ensure_model_downloaded(local_model_path)
        
        # Load the actual model and processor (this method should put model on CPU initially)
        self._load_model_and_processor(local_model_path)
        
        # Wrap the model with ModelManager for lazy GPU loading/unloading
        # Assuming _load_model_and_processor set self.model and self.device
        self._model_manager = ModelManager(self.model, device=self.device)
        self.model = self._model_manager # Override self.model with the manager instance
        self.model_hash = self._get_model_hash_from_instance() # Get hash from the *actual* loaded model
        
        # Setup caches
        self._setup_caches(cache_folder)
        
        print(f"{self.__class__.__name__} initiated successfully.")
    
    def _ensure_model_downloaded(self, local_model_path: str):
        """
        Checks if the model exists locally; if not, downloads it from Hugging Face.
        """
        # Check for a common file like 'config.json' to indicate a potentially complete download
        # Different models might have different definitive files. 'config.json' is a good general one.
        config_file_path = os.path.join(local_model_path, 'config.json') 
        if not os.path.exists(config_file_path):
            print(f"Model '{self.model_name}' not found locally or incomplete at '{local_model_path}'. Downloading...")
            if self.model_name is None:
                print("ERROR: Model name is not set. Cannot download model.")
            else:    
                try:
                    snapshot_download(
                        repo_id=self.model_name,
                        local_dir=local_model_path,
                        local_dir_use_symlinks=False # Download actual files
                    )
                    print(f"Model '{self.model_name}' downloaded successfully.")
                except Exception as e:
                    print(f"ERROR: Failed to download model '{self.model_name}'. Please check your internet connection and permissions.")
                    print(f"Error details: {e}")
                    #raise RuntimeError(f"Failed to download required model: {self.model_name}") from e
        else:
            print(f"Found existing model '{self.model_name}' at '{local_model_path}'.")
    
    def _setup_caches(self, cache_folder: str):
        """Sets up the file list, file hash, and metadata caching mechanisms."""
        os.makedirs(cache_folder, exist_ok=True)
        self.cached_file_list = file_manager.CachedFileList(
            os.path.join(cache_folder, f'{self.cache_prefix}_file_list.pkl')
        )
        self.cached_file_hash = file_manager.CachedFileHash(
            os.path.join(cache_folder, f'{self.cache_prefix}_file_hash.pkl')
        )
        self.cached_metadata = file_manager.CachedMetadata(
            os.path.join(cache_folder, f'{self.cache_prefix}_metadata.pkl'),
            self._get_metadata_function() # Pass the modality-specific metadata function
        )
    
    def _get_model_hash_from_instance(self) -> str:
        """
        Calculates a hash of the currently loaded model's state dictionary.
        Used for versioning embeddings.
        """
        if self._model_manager is None: 
            raise Exception("Model manager is not initialized. Call initiate() first.")
        
        # Access the underlying model's state_dict via ModelManager
        state_dict = self._model_manager.state_dict()
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        buffer.seek(0)
        model_hash = hashlib.md5(buffer.read()).hexdigest() + self._get_model_hash_postfix()
        return model_hash
    
    def process_files(self, file_paths: list[str], callback=None, media_folder: str = None, **kwargs) -> torch.Tensor:
        """
        Processes a list of files to generate their embeddings.
        Handles caching and database interaction.
        """
        # Ensure the engine is initialized and not busy
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        if self.is_busy: 
            raise Exception(f"{self.__class__.__name__} is busy processing another request.")
        
        self.is_busy = True
        
        try:
            all_embeddings = []
            new_db_entries = [] # List to accumulate new database model instances
            db_model_class = self._get_db_model_class()
            
            for ind, file_path in enumerate(tqdm(file_paths, desc=f"Processing {self.cache_prefix} files")):
                current_file_embeddings = None
                try:
                    # Get file hash for caching
                    file_hash = self.cached_file_hash.get_file_hash(file_path)
                    
                    # 1. Check in-memory fast cache first
                    if file_hash in self._fast_cache:
                        current_file_embeddings = self._fast_cache[file_hash]
                    else:
                        if not flask.has_app_context():
                            current_file_embeddings = self._process_single_file(file_path)
                            self._fast_cache[file_hash] = current_file_embeddings
                        else:
                            # 2. Check database for existing embedding
                            db_record = db_model_class.query.filter_by(hash=file_hash).first()

                            if db_record and db_record.embedding and db_record.embedder_hash == self.model_hash:
                                current_file_embeddings = pickle.loads(db_record.embedding)
                                self._fast_cache[file_hash] = current_file_embeddings # Add to fast cache
                            else:
                                # 3. If not in cache or DB, process the file to get embeddings
                                current_file_embeddings = self._process_single_file(file_path, **kwargs)
                                
                                # Ensure embeddings are on CPU for pickling, will be moved to GPU by ModelManager when used
                                if isinstance(current_file_embeddings, torch.Tensor):
                                    current_file_embeddings = current_file_embeddings.cpu()

                                # 4. Prepare for database update/insert
                                if db_record:
                                    db_record.embedding = pickle.dumps(current_file_embeddings)
                                    db_record.embedder_hash = self.model_hash
                                    # No need to add to session for updates managed by SQLAlchemy
                                else:
                                    # Create new record if none exists
                                    new_entry_data = {
                                        'hash': file_hash,
                                        'file_path': os.path.relpath(file_path, media_folder) if media_folder else file_path,
                                        'embedding': pickle.dumps(current_file_embeddings),
                                        'embedder_hash': self.model_hash
                                    }
                                    new_db_entries.append(db_model_class(**new_entry_data))
                                
                                self._fast_cache[file_hash] = current_file_embeddings # Add to fast cache
                
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
                    # Return a zero vector on error to maintain batch shape
                    if self.embedding_dim:
                        current_file_embeddings = torch.zeros(1, self.embedding_dim).cpu()
                    else: # Fallback if embedding_dim not determined yet
                        current_file_embeddings = None # Handle this case in calling function
                
                if current_file_embeddings is not None:
                    all_embeddings.append(current_file_embeddings)
                
                if callback:
                    callback(ind + 1, len(file_paths))
            
            # Commit new database entries and updated ones (if any were modified directly)
            if new_db_entries and flask.has_app_context():
                db_models.db.session.bulk_save_objects(new_db_entries)
                db_models.db.session.commit()

            # Move all embeddings to the same device
            all_embeddings = [embed.to(self._model_manager._device) for embed in all_embeddings if embed is not None]

            # Concatenate all embeddings into a single tensor, move to active device
            if all_embeddings:
                return torch.cat(all_embeddings, dim=0).to(self._model_manager._device)
            else:
                return torch.empty(0, self.embedding_dim).to(self._model_manager._device)
            
        finally:
            self.is_busy = False # Reset busy flag
            self.cached_file_hash.save_hash_cache()
            self.cached_metadata.save_metadata_cache()

    def process_text(self, text: str, **kwargs) -> torch.Tensor:
        """
        Generic method for processing a text query (e.g., for semantic search).
        This will typically just use the model's text encoder.
        """
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")
        
        # Ensure model is loaded to device for inference
        model_instance = self._model_manager # This triggers loading to GPU if needed
        
        # Use the specific model's text processing capabilities
        # This part still needs to be implemented in subclasses if the text processing
        # is significantly different (e.g., tokenizer usage, specific model methods)
        raise NotImplementedError("Text processing is not implemented in BaseSearchEngine directly. "
                                  "Subclasses should implement their own 'process_text' if applicable.")
    
    def compare(self, embeds_target: torch.Tensor, embeds_query: torch.Tensor) -> np.ndarray:
        """
        Generic method for comparing two sets of embeddings (e.g., image vs text).
        Performs dot product and applies sigmoid/softmax as appropriate.
        Subclasses can override if their comparison logic is more complex.
        """
        if self._model_manager is None:
            raise RuntimeError(f"{self.__class__.__name__} not initialized. Call initiate() first.")

        # Ensure embeddings are on the correct device
        embeds_target = embeds_target.to(self._model_manager._device)
        embeds_query = embeds_query.to(self._model_manager._device)

        # Normalize features
        embeds_target = embeds_target / embeds_target.norm(p=2, dim=-1, keepdim=True)
        embeds_query = embeds_query / embeds_query.norm(p=2, dim=-1, keepdim=True)

        # Dot product for similarity
        logits = torch.matmul(embeds_query, embeds_target.t())

        # If model has logit_scale (like CLIP/CLAP), apply it
        if hasattr(self._model_manager, 'logit_scale'):
            logits = logits * self._model_manager.logit_scale.exp()
        elif hasattr(self._model_manager, 'logit_scale_t') and hasattr(self._model_manager, 'logit_scale_a'): # For CLAP
            # Assuming embeds_query is text and embeds_target is audio/image
            logits_per_text = logits * self._model_manager.logit_scale_t.exp()
            logits_per_audio = torch.matmul(embeds_target, embeds_query.t()) * self._model_manager.logit_scale_a.exp()
            logits = (logits_per_text + logits_per_audio.t()) / 2.0
            
        probs = torch.sigmoid(logits) # Sigmoid for similarity score (0-1)

        return probs.cpu().detach().numpy()