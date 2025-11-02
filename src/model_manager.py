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
        # Unwrap if someone accidentally passes a ModelManager
        if isinstance(model, ModelManager):
            model = model._model


        # Store the model and its original state
        self._model = model
        self._model_id = id(model)

        # Choose a sensible default device
        if device is not None:
            self._device = device
        elif hasattr(model, 'device'):
            self._device = model.device
        else:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
       
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

        # Methods that should not trigger loading
        self._no_load_calls = {
            'state_dict',
            'load_state_dict',
            'parameters',
            'named_parameters',
            'buffers',
            'named_buffers',
            'eval',
            'train',
        }

    def __call__(self, *args, **kwargs):
        """
        Handle direct calls to the model (e.g., forward). Load lazily here.
        """
        with ModelManager._lock:
            self._last_used = time.time()
            if not self._loaded:
                self._load_model()
            target = self._model  # bound __call__/forward
        with self._busy_lock:
            self._busy = True
        try:
            return target(*args, **kwargs)
        finally:
            with self._busy_lock:
                self._busy = False
        
        
    def __getattr__(self, name):
        """
        Lazily resolve attributes. For callables, return a wrapper that ensures the
        model is (re)loaded right before invocation. Do NOT load just to read attributes.
        """
        with ModelManager._lock:
            self._last_used = time.time()

            if hasattr(self._model, name):
                attr = getattr(self._model, name)
            else:
                raise AttributeError(f"'{self._model.__class__.__name__}' object has no attribute '{name}'")
            
            if callable(attr) and not name.startswith('__'):
                # Pass-through for CPU-safe methods that shouldn't trigger GPU load
                if name in self._no_load_calls:
                    def passthrough(*args, **kwargs):
                        with self._busy_lock:
                            self._busy = True
                        try:
                            return getattr(self._model, name)(*args, **kwargs)
                        finally:
                            with self._busy_lock:
                                self._busy = False
                    return passthrough

                def wrapped_method(*args, **kwargs):
                    with ModelManager._lock:
                        self._last_used = time.time()
                        if not self._loaded:
                            self._load_model()
                        target = getattr(self._model, name)
                    with self._busy_lock:
                        self._busy = True
                    try:
                        return target(*args, **kwargs)
                    finally:
                        with self._busy_lock:
                            self._busy = False
                return wrapped_method

            # Non-callables: return without forcing a load
            return attr
    
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
        """Load the model back to the desired device (lazy)."""
        if not self._loaded:
            try:
                name = getattr(self._model, "_name", self._model.__class__.__name__)
                print(f"Loading model {name} to {self._device}...")
                self._model = self._model.to(self._device)
                self._loaded = True
            except Exception as e:
                print(f"ERROR: Failed to load model: {e}")
                traceback.print_exc()
    
    def _unload_model(self):
        """Unload the model from GPU memory"""
        if self._loaded:
            try:
                name = getattr(self._model, "_name", self._model.__class__.__name__)
                print(f"Unloading model {name} from {self._device} (idle for {time.time() - self._last_used:.1f}s)...")
                self._cache_attributes()
                self._model = self._model.cpu()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self._loaded = False
            except Exception as e:
                print(f"Error unloading model: {e}")
    
    def unload_model(self):
        """Public API to unload the wrapped model from GPU."""
        with ModelManager._lock:
            self._unload_model()
        return self
    
    def _cache_attributes(self):
        """Cache important non-method attributes of the model"""
        for attr_name in dir(self._model):
            if not attr_name.startswith('__') and not callable(getattr(self._model, attr_name)):
                try:
                    self._attribute_cache[attr_name] = getattr(self._model, attr_name)
                except:
                    pass  # Skip attributes that can't be accessed
    
    # def state_dict(self):
    #     """Return the model's state dictionary"""
    #     with ModelManager._lock:
    #         self._last_used = time.time()
    #         if not self._loaded:
    #             self._load_model()
    #         return self._model.state_dict()

    def to(self, device):
        """
        Set the desired device. If already loaded, move now; otherwise defer until first use.
        """
        with ModelManager._lock:
            self._last_used = time.time()
            self._device = device
            if self._loaded:
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