import threading
import time
import weakref
import torch
import traceback
from types import MethodType

# TODO: The CPU cleanup has been disabled for now, as it is not working properly.
# I need to work on this to ensure that the CPU memory is also freed when the model is unloaded and everything is handled correctly.

class ModelManager:
    """
    A proxy class that wraps PyTorch models to:
    1. Load them to GPU only when methods are called
    2. Unload them after a period of inactivity (2 minutes by default)
    """
    # Class-level tracking of all models
    _models = {}
    _lock = threading.RLock()
    _cleanup_thread = None
    _shutdown = False

    def __init__(self, model, device=None, idle_timeout=120):
        """
        Initialize the model manager for a specific model.
        
        Args:
            model: The model object to manage
            device: The device to load the model to (default: current model device)
            idle_timeout: Time in seconds after which to unload idle models (default: 120s)
        """
        # Store the model and its original state
        self._model = model
        self._model_id = id(model)
        self._device = device or (model.device if hasattr(model, 'device') else 'cuda')
        self._loaded = False
        self._last_used = time.time()
        self._idle_timeout = idle_timeout
        self._busy = False 
        self._busy_lock = threading.Lock() 
        
        # Store the model's original methods and attributes
        self._original_methods = {}
        self._attribute_cache = {}
        
        # Store original state_dict for reloading
        self._state_dict = model.state_dict()
        self._model_class = model.__class__
        
        # Register this model in the class-level registry
        with ModelManager._lock:
            ModelManager._models[self._model_id] = self
            self._ensure_cleanup_thread()

    def __call__(self, *args, **kwargs):
        """
        Handle method calls, loading the model if needed.
        This is called when the model instance is called like a function.
        """
        with ModelManager._lock:
            self._last_used = time.time()
            
            # If model is not loaded, load it
            if not self._loaded:
                self._load_model()
            
            with self._busy_lock:
                self._busy = True
            try:
                # Call the method on the model
                return self._model(*args, **kwargs)
            finally:
                with self._busy_lock:
                    self._busy = False
        
        
    def __getattr__(self, name):
        """
        Handle attribute access, loading the model if needed.
        This is called when Python can't find the attribute through normal means.
        """
        with ModelManager._lock:
            self._last_used = time.time()
            
            # If model is not loaded, load it
            if not self._loaded:
                self._load_model()
                
            # Try to get the attribute from the model
            if hasattr(self._model, name):
                attr = getattr(self._model, name)
                
                # If it's a method, wrap it to update last_used timestamp
                if callable(attr) and not name.startswith('__'):
                    def wrapped_method(*args, **kwargs):
                        with ModelManager._lock:
                            self._last_used = time.time()
                        
                        with self._busy_lock:
                            self._busy = True
                        try:
                            return attr(*args, **kwargs)
                        finally:
                            with self._busy_lock:
                                self._busy = False
                    return wrapped_method
                return attr
                
            # If the attribute doesn't exist, raise an AttributeError
            raise AttributeError(f"'{self._model.__class__.__name__}' object has no attribute '{name}'")
    
    def __dir__(self):
        """Return all attributes of the wrapped model plus our own"""
        own_attrs = dir(type(self))
        if self._loaded:
            model_attrs = dir(self._model)
        else:
            # When model is unloaded, use cached attributes
            model_attrs = list(self._attribute_cache.keys())
        return list(set(own_attrs + model_attrs))
    
    def _load_model(self):
        """Load the model back to the device"""
        if not self._loaded:
            try:
                print(f"Loading model {self._model.__class__.__name__} to {self._device}...")
                # Create a new instance of the model
                #self._model = self._model_class()
                # Load the state dict
                #self._model.load_state_dict(self._state_dict)
                # Move to device
                self._model = self._model.to(self._device)
                self._loaded = True
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")
                traceback.print_exc()
    
    def _unload_model(self):
        """Unload the model from GPU memory"""
        if self._loaded:
            try:
                print(f"Unloading model {self._model.__class__.__name__} from GPU (idle for {time.time() - self._last_used:.1f}s)...")
                # Cache important attributes before unloading
                self._cache_attributes()
                # Move to CPU first to properly free CUDA memory
                self._model = self._model.cpu()
                # Delete the model to free memory
                #del self._model
                # Set a placeholder to avoid errors
                #self._model = None
                # Trigger garbage collection to free memory
                torch.cuda.empty_cache()
                self._loaded = False
            except Exception as e:
                print(f"Error unloading model: {e}")
    
    def _cache_attributes(self):
        """Cache important non-method attributes of the model"""
        for attr_name in dir(self._model):
            if not attr_name.startswith('__') and not callable(getattr(self._model, attr_name)):
                try:
                    self._attribute_cache[attr_name] = getattr(self._model, attr_name)
                except:
                    pass  # Skip attributes that can't be accessed
    
    def state_dict(self):
        """Return the model's state dictionary"""
        with ModelManager._lock:
            self._last_used = time.time()
            if not self._loaded:
                self._load_model()
            return self._model.state_dict()

    def to(self, device):
        """Move the model to the specified device"""
        with ModelManager._lock:
            self._last_used = time.time()
            if not self._loaded:
                self._load_model()
            self._device = device
            self._model = self._model.to(device)
            return self
    
    # def config(self):
    #     """Access the model's configuration"""
    #     with ModelManager._lock:
    #         self._last_used = time.time()
    #         if hasattr(self._model, 'config'):
    #             return self._model.config
    #         return None
    
    @classmethod
    def _cleanup_idle_models(cls):
        """Check and unload models that have been idle for too long"""
        current_time = time.time()
        with cls._lock:
            for model_id, manager in list(cls._models.items()):
                with manager._busy_lock:
                    is_busy = manager._busy

                if (not is_busy and manager._loaded and 
                    current_time - manager._last_used > manager._idle_timeout):
                    manager._unload_model()
    
    @classmethod
    def _cleanup_thread_func(cls):
        """Background thread that periodically checks for idle models"""
        print("Model cleanup thread started")
        while not cls._shutdown:
            try:
                time.sleep(30)  # Check every 30 seconds
                if not cls._shutdown:
                    cls._cleanup_idle_models()
            except Exception as e:
                print(f"Error in cleanup thread: {e}")
        print("Model cleanup thread stopped")
    
    @classmethod
    def _ensure_cleanup_thread(cls):
        """Start the cleanup thread if it's not already running"""
        if cls._cleanup_thread is None or not cls._cleanup_thread.is_alive():
            cls._shutdown = False
            cls._cleanup_thread = threading.Thread(
                target=cls._cleanup_thread_func, 
                daemon=True
            )
            cls._cleanup_thread.start()
    
    @classmethod
    def shutdown(cls):
        """Properly shutdown the manager and cleanup thread"""
        cls._shutdown = True
        if cls._cleanup_thread and cls._cleanup_thread.is_alive():
            cls._cleanup_thread.join(timeout=1.0)
        print(f"Model manager shutdown, active threads: {threading.active_count()}")