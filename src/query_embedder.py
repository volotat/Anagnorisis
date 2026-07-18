"""
query_embedder.py — In-process CPU query embedder for search-time use.

Loaded once on first call, cached in RAM, used for one query at a time.
This avoids the subprocess + CUDA initialisation cost of the main
ImageEmbedder / AudioEmbedder / TextEmbedder for a single forward pass.

No GPU is ever touched. Background tasks (which DO use the GPU via the
subprocess embedders) cannot be starved by a search.
"""
import os
import threading

import numpy as np
import torch
from typing import Optional


class QueryEmbedder:
    """Lazy-loaded CPU query embedder for search-time use.

    Singleton per (model_name, model_type) pair. Model is loaded on first
    embed_text() call, cached in memory until process exit or unload().
    """

    # Per-(model, type) singletons so we never load the same model twice
    _instances: dict = {}
    _lock = threading.Lock()

    def __init__(self, model_name: str, models_folder: str, model_type: str):
        # model_type: 'image' (SigLIP) | 'audio' (CLAP) | 'text' (SentenceTransformer)
        self.model_name = model_name
        self.models_folder = models_folder
        self.model_type = model_type
        self._model = None
        self._processor = None  # None for SentenceTransformer
        self._load_lock = threading.Lock()
        self._load_attempted = False
        self.embedding_dim: Optional[int] = None
        # For sentence_transformers: truncate_dim applied at encode() time
        self._truncate_dim: Optional[int] = None

    @classmethod
    def get_instance(cls, model_name, models_folder, model_type, truncate_dim=None):
        key = (model_name, model_type, models_folder)
        with cls._lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(model_name, models_folder, model_type)
                if truncate_dim is not None:
                    inst._truncate_dim = truncate_dim
                cls._instances[key] = inst
            return inst

    # ── public API ────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> Optional[np.ndarray]:
        """Embed a single text query on CPU. Returns None if model unavailable."""
        if not self._ensure_loaded() or self._model is None:
            return None
        try:
            with torch.no_grad():
                if self.model_type == 'text':
                    # sentence-transformers API (uses retrieval.query prompt for queries)
                    emb = self._model.encode(
                        text, task='retrieval.query', convert_to_tensor=False,
                        device='cpu', truncate_dim=self._truncate_dim,
                    )
                    return np.asarray(emb, dtype=np.float32)

                # transformers API (SigLIP / CLAP) — defensive unwrap for >=5.x
                inputs = self._processor(text=text, padding='max_length', return_tensors='pt')
                inputs = {k: (v.to('cpu') if isinstance(v, torch.Tensor) else v)
                          for k, v in inputs.items()}
                out = self._model.get_text_features(**inputs)
                if isinstance(out, torch.Tensor):
                    features = out
                elif hasattr(out, 'pooler_output') and out.pooler_output is not None:
                    features = out.pooler_output
                elif hasattr(out, 'text_embeds') and out.text_embeds is not None:
                    features = out.text_embeds
                elif hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
                    features = out.last_hidden_state[:, 0]
                else:
                    raise RuntimeError(f'Unexpected output type: {type(out).__name__}')
                # L2-normalise to match the existing subprocess output convention
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                return features.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            print(f'[QueryEmbedder] embed_text failed for {self.model_name}: {exc}')
            return None

    def unload(self):
        """Free model memory. Safe to call multiple times."""
        with self._load_lock:
            self._model = None
            self._processor = None
            import gc
            gc.collect()

    # ── internals ────────────────────────────────────────────────────────

    def _local_path(self) -> str:
        return os.path.join(self.models_folder, self.model_name.replace('/', '__'))

    def _ensure_loaded(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        with self._load_lock:
            if self._load_attempted:
                return self._model is not None
            self._load_attempted = True
            return self._load_model()

    def _load_model(self) -> bool:
        local = self._local_path()
        if not os.path.exists(os.path.join(local, 'config.json')):
            return False  # model not downloaded yet
        try:
            if self.model_type == 'image':
                from transformers import AutoModel, AutoProcessor
                self._model = AutoModel.from_pretrained(
                    local, local_files_only=True, torch_dtype=torch.float32,
                )
                self._processor = AutoProcessor.from_pretrained(
                    local, local_files_only=True,
                )
            elif self.model_type == 'audio':
                from transformers import ClapModel, ClapProcessor
                self._model = ClapModel.from_pretrained(local, local_files_only=True)
                self._processor = ClapProcessor.from_pretrained(local, local_files_only=True)
            elif self.model_type == 'text':
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    local, device='cpu', trust_remote_code=True,
                    processor_kwargs={'fix_mistral_regex': True},
                )
            else:
                return False
            self._model = self._model.to('cpu')
            if hasattr(self._model, 'eval'):
                self._model.eval()
            return True
        except Exception as exc:
            print(f'[QueryEmbedder] Failed to load {self.model_name} on CPU: {exc}')
            self._model = None
            self._processor = None
            return False