"""omni_embedder.py — the single embedding model for every kind of content.

One multimodal model (`jina-embeddings-v5-omni`) embeds text, images, audio and
video into one shared vector space, which is what lets a natural-language query
rank a photo, a song and a `.meta` description against each other in the same
list. It replaces the three separate subprocess embedders this project used to
carry (CLAP for audio, SigLIP for images, Qwen3 for text), each of which owned
its own vector space and its own copy of the same subprocess plumbing.

Runs the model in a spawned subprocess, exactly like its predecessors did: the
GPU context stays out of the Flask process, and the worker is terminated after
an idle period so background tasks and searches never hold VRAM they are not
using. ``model_hash`` survives an unload, so cache keys stay stable across a
restart of the worker.

Retrieval is asymmetric — a query and a document are encoded differently, and
that applies to *every* modality, not just text. Use ``embed_query`` for what
the user typed and ``embed_document`` for what is being searched.
"""

import multiprocessing
import os
import queue
import sys
import threading
import time
import traceback
from typing import List, Optional, Sequence, Union

import numpy as np
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.models.model_identity import fingerprint_model_dir

# Anything sentence-transformers accepts directly as one item to encode.
EmbedInput = Union[str, Sequence[str]]


def compute_model_hash(cfg, models_folder: str) -> str:
    """The embedder's identity, computed from the weights on disk.

    Deliberately needs no loaded model. That is what lets a *search* build the
    same cache keys the indexer used without waking the GPU: searching must never
    load a model, and a search that cannot name the model cannot find anything it
    indexed. Both the worker and the proxy call this, so the two can never
    disagree about what a cache key means.
    """
    model_name = cfg.embedder.model_name
    local_path = os.path.join(models_folder, model_name.replace('/', '__'))
    dim = getattr(cfg.embedder, 'embedding_dimension', None)
    return fingerprint_model_dir(
        local_path,
        model_name,
        getattr(cfg.embedder, 'task', 'retrieval'),
        int(dim) if dim else None,
        # Images and video frames are reduced to this long side before they
        # reach the model, and the processor is capped to match, so it changes
        # the vector exactly the way the truncation dimension does.
        int(getattr(cfg.embedder, 'video_frame_max_size', 512) or 512),
    )


def is_tensor(value) -> bool:
    """True if *value* is a torch tensor, without importing torch to find out.

    Importing torch costs about 1.1 seconds, and every command pays it if any
    module on the import graph asks for it at the top. Nothing in this process
    can be holding a tensor unless torch is already loaded, so when it is not,
    the answer is no and there is nothing to pay for.
    """
    torch = sys.modules.get('torch')
    return torch is not None and isinstance(value, torch.Tensor)


def _block_url_fetching() -> None:
    """Make the model treat a URL as the text it is, never as a file to fetch.

    The model ships its own ``custom_st.py``. Because it takes text and media
    as the same type — plain strings — it cannot tell them apart by signature,
    so it guesses: it hands anything starting with ``http://`` or ``https://``
    to ``urllib.request.urlretrieve`` and sniffs whatever comes back. It runs
    that guess over every string in every batch, *before* embedding anything.

    A note beginning with a link therefore became an outbound request, and the
    fetched bytes could be embedded in place of the text. Worse, the media
    branch is chosen for the whole batch if any one string trips it, so a
    single link changes how everything alongside it is encoded.

    Reading a file must read the file and nothing else, and a local index must
    not make requests because of what someone wrote in their notes.

    ``_resolve_input`` is the one choke point: both the media *detection* path
    and the two real encoding paths funnel through it. Short-circuiting URLs
    there skips the download, the existence check and the content sniff in one
    move, and leaves local media entirely to the model's own code. We only ever
    pass local paths anyway — remote files are copied locally first.
    """
    import sys

    patched = []
    for name, module in list(sys.modules.items()):
        # Only look inside the model's own vendored code. Reaching into every
        # loaded module would mean calling getattr on things like torchaudio,
        # whose lazy attributes have side effects of their own.
        if module is None or 'transformers_modules' not in name:
            continue
        resolve = getattr(module, '_resolve_input', None)
        if not callable(resolve) or not hasattr(module, '_is_media_string'):
            continue

        if not getattr(resolve, '_url_blocked', False):
            def _resolve_local_only(x, _original=resolve):
                if isinstance(x, str) and x.startswith(('http://', 'https://')):
                    return ('text', x)
                return _original(x)

            _resolve_local_only._url_blocked = True
            module._resolve_input = _resolve_local_only
            # Unreachable for URLs now, but it is module-level and cheap to
            # make harmless in case a future version calls it elsewhere.
            module._download_if_url = lambda x: x
        patched.append(name)

    if not patched:
        # The model's internals changed shape. Say so loudly: the silent
        # failure mode is the network access quietly coming back.
        print("[OmniEmbedder] WARNING: could not disable the model's URL "
              "fetching — it may reach the network for text that looks like a "
              "link. Check custom_st.py for '_resolve_input'.")


def _cap_audio_decode(seconds: float = 30.0) -> None:
    """Decode only the audio the model can actually look at.

    The model's ``_load_audio_array`` calls ``librosa.load`` with no bound and
    hands the result to a ``WhisperFeatureExtractor`` built with
    ``padding="max_length"``, whose default ``truncation=True`` cuts the input
    at ``n_samples`` — exactly 30 seconds at 16 kHz. Everything decoded past
    that point is resampled and then discarded. Verified against the installed
    transformers: a 240s input and a 30s input produce identical
    ``(1, 128, 3000)`` features, so this is a pure saving and not a trade.

    Measured on real tracks with a warm page cache: 0.391s per file to decode
    the whole thing against 0.054s for the first 30 seconds.
    """
    patched = []
    for name, module in list(sys.modules.items()):
        if module is None or 'transformers_modules' not in name:
            continue
        loader = getattr(module, '_load_audio_array', None)
        if not callable(loader):
            continue

        if not getattr(loader, '_duration_capped', False):
            def _load_capped(audio_input, _original=loader, _secs=seconds):
                if isinstance(audio_input, str) and os.path.isfile(audio_input):
                    import librosa
                    audio, sr = librosa.load(audio_input, sr=16000, duration=_secs)
                    return audio.astype(np.float32), sr
                return _original(audio_input)

            _load_capped._duration_capped = True
            module._load_audio_array = _load_capped

        patched.append(name)

    if not patched:
        print("[OmniEmbedder] WARNING: could not cap audio decoding — long "
              "tracks will be decoded in full and then truncated to 30s. "
              "Check custom_st.py for '_load_audio_array'.")


def _cap_visual_pixels(model, max_side: int = 512) -> None:
    """Stop the processor from scaling images and frames back up.

    Video frames are already downscaled to a 512px long side before they are
    handed over, and then ``_align_eval_processor`` sets a *minimum* pixel count
    of 262144 — so Qwen's ``smart_resize`` upscales a 512x288 frame back to
    roughly 672x384, undoing the downscale and paying for the extra visual
    tokens. Images go in at full size and land anywhere up to 1145 square.

    Capping the maximum at ``max_side`` squared and dropping the minimum to the
    model's own floor means nothing is ever enlarged and nothing exceeds the
    512px long side. Both the module constants and the live processor objects
    have to be set: the constants are re-applied per call, and the processors
    were built before this runs.
    """
    import sys

    cap = max_side * max_side
    floor = 4 * 28 * 28  # Qwen's own minimum patch budget.

    for name, module in list(sys.modules.items()):
        if module is None or 'transformers_modules' not in name:
            continue
        if not hasattr(module, 'EVAL_IMAGE_MIN_PIXELS'):
            continue
        module.EVAL_IMAGE_MIN_PIXELS = floor
        module.EVAL_IMAGE_MAX_PIXELS = cap
        module.EVAL_VIDEO_MAX_PIXELS = cap

    for st_module in model:
        processor = getattr(st_module, 'processor', None)
        if processor is None:
            continue
        for attr in ('image_processor', 'video_processor'):
            sub = getattr(processor, attr, None)
            if sub is None:
                continue
            if hasattr(sub, 'min_pixels'):
                sub.min_pixels = floor
            if hasattr(sub, 'max_pixels'):
                sub.max_pixels = cap
            size = getattr(sub, 'size', None)
            if size is None:
                continue
            try:
                sub.size = type(size)(**{**dict(size),
                                         'longest_edge': cap,
                                         'shortest_edge': floor})
            except Exception:
                # A shape we do not recognise: leave it rather than break it.
                pass


def cosine_similarity(embeddings, query_embedding) -> List[float]:
    """Cosine similarity of each row against the query.

    Deliberately a plain function rather than a method on the embedder: it is
    pure arithmetic, and routing it through the GPU worker would wake the model
    just to do a matrix multiply — on the search path, of all places.
    """
    if embeddings is None or query_embedding is None:
        return [0.0]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    query_embedding = np.asarray(query_embedding, dtype=np.float32).ravel()
    if embeddings.size == 0 or query_embedding.size == 0:
        return [0.0]
    rows = embeddings.reshape(-1, query_embedding.shape[0])
    rows = rows / np.clip(np.linalg.norm(rows, axis=-1, keepdims=True), 1e-12, None)
    query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-12)
    return (rows @ query_embedding).astype(np.float32).tolist()


# ---------------------------------------------------------------------------
# Worker implementation — runs inside the subprocess
# ---------------------------------------------------------------------------

class _OmniEmbedderImpl:
    """Holds the model and the CUDA context inside the worker process."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.embedding_dim = None
        self.model_hash = None
        import torch

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_seq_length = None
        self._video_extensions = None
        self._image_extensions = None

    # -- setup ----------------------------------------------------------

    def initiate(self, models_folder: str):
        if self.model is not None:
            return self._state()

        model_name = self.cfg.embedder.model_name
        local_path = os.path.join(models_folder, model_name.replace('/', '__'))
        _ensure_model_downloaded(model_name, local_path)

        from sentence_transformers import SentenceTransformer

        task = getattr(self.cfg.embedder, 'task', 'retrieval')
        self.model = SentenceTransformer(
            local_path,
            trust_remote_code=True,
            model_kwargs={'default_task': task},
            device=str(self.device),
        )
        self.model.eval()
        _block_url_fetching()
        _cap_audio_decode(float(getattr(self.cfg.embedder, 'audio_seconds', 30.0) or 30.0))
        _cap_visual_pixels(self.model, self._visual_max_side())

        self.max_seq_length = int(getattr(self.model, 'max_seq_length', 0) or 0)
        self.model_hash = self._calculate_model_hash(local_path)
        probe = self.model.encode_document('probe', truncate_dim=self._truncate_dim())
        self.embedding_dim = int(np.asarray(probe).ravel().shape[0])

        print(f"OmniEmbedder (Worker): Initiated '{model_name}' on {self.device}. "
              f"dim={self.embedding_dim} max_seq_length={self.max_seq_length}")
        return self._state()

    def _state(self):
        return {
            'embedding_dim': self.embedding_dim,
            'device_type': self.device.type,
            'model_hash': self.model_hash,
            'max_seq_length': self.max_seq_length,
        }

    def _truncate_dim(self) -> Optional[int]:
        dim = getattr(self.cfg.embedder, 'embedding_dimension', None)
        return int(dim) if dim else None

    def _calculate_model_hash(self, local_path: str) -> str:
        """Fingerprint what determines a vector: the weights plus the settings
        that change how they are applied.

        ``task`` selects a different LoRA and the truncation dim changes the
        vector's length, so both alter the output while leaving the files
        untouched — see :mod:`anagnorisis_core.models.model_identity` for why this reads the files
        rather than the loaded model.
        """
        return compute_model_hash(self.cfg, os.path.dirname(local_path))

    # -- embedding ------------------------------------------------------

    def embed_query(self, item: EmbedInput) -> np.ndarray:
        """Encode what the user is searching *for* (any modality)."""
        item = self._prepare(item)
        return self._to_numpy(self.model.encode_query(item, truncate_dim=self._truncate_dim()))

    def embed_document(self, item: EmbedInput) -> np.ndarray:
        """Encode a thing being searched *over* (any modality)."""
        item = self._prepare(item)
        return self._to_numpy(self.model.encode_document(item, truncate_dim=self._truncate_dim()))

    # -- video -----------------------------------------------------------

    def _visual_max_side(self) -> int:
        """Long side, in pixels, that any image or frame is reduced to first."""
        return int(getattr(self.cfg.embedder, 'video_frame_max_size', 512) or 512)

    def _prepare(self, item):
        """Hand media in as decoded arrays rather than as a path.

        Given a video path, transformers decodes the *entire* file into memory
        before sampling it down, which kills the worker on clips as small as
        24 MB. (Its preferred decoder, torchcodec, avoids that but ships
        CUDA-version specific binaries.) Sampling here keeps the cost
        proportional to the number of frames we actually want, not to the
        length of the file.

        Images are resolved here for a different reason. Handed a path, the
        model decodes the file *twice*: once inside ``_is_media_string`` purely
        to learn that it is an image, throwing the pixels away, and again when
        it actually encodes it. Passing a decoded image takes the non-string
        branch and costs one decode, and it is also where the 512px long side
        gets enforced rather than left to the processor.
        """
        if not isinstance(item, str):
            return item
        if self._is_video(item):
            frames = self._sample_video_frames(item)
            if frames is None:
                # Do NOT fall back to handing over the path: that is the
                # full-decode route that kills the worker. An unreadable video
                # is a failed file, which the caller records and moves past.
                raise ValueError(f"Could not sample frames from video: {item}")
            return frames
        if self._is_image(item):
            image = self._load_image(item)
            if image is not None:
                return image
        return item

    def _is_image(self, path: str) -> bool:
        if self._image_extensions is None:
            from omegaconf import OmegaConf
            exts = OmegaConf.select(self.cfg, 'media_types.images.extensions', default=None) or []
            self._image_extensions = {str(e).lower() for e in exts}
        return os.path.splitext(path)[1].lower() in self._image_extensions

    def _load_image(self, path: str):
        """Decode an image once, no larger than the configured long side.

        ``draft()`` lets the JPEG decoder do the downscale while decoding rather
        than after, so a 12 MP photo never becomes a 12 MP array in the first
        place. It is a no-op for formats that do not support it.
        """
        try:
            from PIL import Image as PILImage

            max_side = self._visual_max_side()
            image = PILImage.open(path)
            try:
                image.draft('RGB', (max_side, max_side))
            except Exception:
                pass
            image = image.convert('RGB')
            if max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize(
                    (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
                    resample=PILImage.BICUBIC,
                )
            return image
        except Exception as exc:
            print(f"OmniEmbedder (Worker): image decode failed for {path}: {exc}")
            return None

    def _is_video(self, path: str) -> bool:
        if self._video_extensions is None:
            from omegaconf import OmegaConf
            exts = OmegaConf.select(self.cfg, 'media_types.videos.extensions', default=None) or []
            self._video_extensions = {str(e).lower() for e in exts}
        return os.path.splitext(path)[1].lower() in self._video_extensions

    def _sample_video_frames(self, path: str) -> Optional[np.ndarray]:
        """Evenly spaced frames as (T, H, W, 3) uint8, downscaled. None on failure.

        This is where video indexing actually spends its time: measured 4.63s
        per file to pull the frames against 0.13s to embed them, so the decoder
        is the whole cost and the GPU is idle waiting for it.

        One sequential ffmpeg pass beats seeking to sixteen positions, because
        every `CAP_PROP_POS_FRAMES` on long-GOP H.264 re-seeks to a keyframe and
        decodes forward again. Measured over five demo videos: 2.65s against
        4.63s, with the scaling done during decode rather than after. OpenCV
        stays as the fallback for whatever ffmpeg will not open.
        """
        frames = self._sample_video_frames_ffmpeg(path)
        if frames is not None:
            return frames
        return self._sample_video_frames_cv2(path)

    def _video_duration(self, path: str) -> Optional[float]:
        import json
        import subprocess
        try:
            out = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'json', path],
                capture_output=True, timeout=60)
            if out.returncode != 0:
                return None
            value = json.loads(out.stdout)['format']['duration']
            duration = float(value)
            return duration if duration > 0 else None
        except Exception:
            return None

    def _sample_video_frames_ffmpeg(self, path: str) -> Optional[np.ndarray]:
        """*count* evenly spaced frames from one decode pass, or None."""
        import io
        import subprocess

        count = int(getattr(self.cfg.embedder, 'video_frames', 16) or 16)
        max_side = self._visual_max_side()
        duration = self._video_duration(path)
        if not duration or count <= 0:
            return None

        try:
            from PIL import Image as PILImage
        except ImportError:
            return None

        # An fps low enough to yield `count` frames across the whole file.
        rate = count / duration
        scale = (f"scale='if(gt(iw,ih),{max_side},-1)'"
                 f":'if(gt(ih,iw),{max_side},-1)'")
        cmd = ['ffmpeg', '-loglevel', 'error', '-hide_banner',
               '-err_detect', 'ignore_err', '-i', path,
               '-vf', f'fps={rate:.6f},{scale}',
               '-vsync', '0', '-frames:v', str(count),
               '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1']
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=600)
        except Exception:
            return None
        if proc.returncode != 0 or not proc.stdout:
            return None

        from anagnorisis_core.models.media_io import split_png_stream

        frames = []
        for chunk in split_png_stream(proc.stdout):
            try:
                frames.append(np.asarray(
                    PILImage.open(io.BytesIO(chunk)).convert('RGB'), dtype=np.uint8))
            except Exception:
                break
        if not frames:
            return None
        # Frames can differ by a pixel after rounding; trim to a common size.
        h = min(f.shape[0] for f in frames)
        w = min(f.shape[1] for f in frames)
        return np.stack([f[:h, :w] for f in frames])

    def _sample_video_frames_cv2(self, path: str) -> Optional[np.ndarray]:
        """Fallback frame sampler for files ffmpeg will not read."""
        try:
            import cv2
        except ImportError:
            return None

        count = int(getattr(self.cfg.embedder, 'video_frames', 16) or 16)
        max_side = int(getattr(self.cfg.embedder, 'video_frame_max_size', 512) or 512)

        capture = cv2.VideoCapture(path)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                return None
            positions = (np.linspace(0, total - 1, min(count, total))
                         .astype(int).tolist())
            frames = []
            for position in positions:
                capture.set(cv2.CAP_PROP_POS_FRAMES, position)
                ok, frame = capture.read()
                if not ok:
                    continue
                height, width = frame.shape[:2]
                scale = min(1.0, max_side / max(height, width))
                if scale < 1.0:
                    frame = cv2.resize(frame, (int(width * scale), int(height * scale)),
                                       interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not frames:
                return None
            # Frames can differ by a pixel after rounding; trim to a common size.
            h = min(f.shape[0] for f in frames)
            w = min(f.shape[1] for f in frames)
            return np.stack([f[:h, :w] for f in frames]).astype(np.uint8)
        except Exception as exc:
            print(f"OmniEmbedder (Worker): frame sampling failed for {path}: {exc}")
            return None
        finally:
            capture.release()

    def embed_documents(self, items: List[EmbedInput]) -> np.ndarray:
        """Batch form of embed_document. Returns [N, D]."""
        if not items:
            return np.zeros((0, self.embedding_dim or 0), dtype=np.float32)
        batch_size = int(getattr(self.cfg.embedder, 'batch_size', 8) or 8)
        out = self.model.encode_document(
            list(items), truncate_dim=self._truncate_dim(), batch_size=batch_size
        )
        return np.asarray(out, dtype=np.float32).reshape(len(items), -1)

    def embed_long_text(self, long_text: str) -> np.ndarray:
        """Encode text of any length. Returns [n_chunks, D].

        Text that fits the context window becomes ONE vector from a single
        forward pass — a genuine contextual embedding of the whole document, not
        an average of pieces. Only text that exceeds the window is split, and
        then with an overlap so a passage spanning a boundary is still covered
        by one chunk in full.
        """
        chunks = self._split_text(long_text)
        if not chunks:
            return np.zeros((0, self.embedding_dim or 0), dtype=np.float32)
        return self.embed_documents(chunks)

    def _split_text(self, long_text: str) -> List[str]:
        if not long_text:
            return []

        limit = int(getattr(self.cfg.embedder, 'chunk_size', 0) or 0)
        if limit <= 0:
            limit = self.max_seq_length or 8192
        # Leave room for the "Document: " prefix and any special tokens.
        limit = max(64, limit - 16)

        # Tokenizing the whole document with an offset map, only to find it fits
        # and hand the original string back for sentence-transformers to
        # tokenize again, is two full passes over as much as a megabyte of text.
        # No tokenizer produces more tokens than the string has characters, so
        # anything shorter than the limit certainly fits and needs no splitting.
        if len(long_text) <= limit:
            return [long_text]

        tokenizer = self.model.tokenizer
        tokens = tokenizer(long_text, add_special_tokens=False,
                           truncation=False, return_offsets_mapping=True)
        ids = tokens['input_ids']
        offsets = tokens['offset_mapping']

        if len(ids) <= limit:
            return [long_text]

        overlap = int(getattr(self.cfg.embedder, 'chunk_overlap', 0) or 0)
        overlap = max(0, min(overlap, limit // 2))
        step = max(1, limit - overlap)

        chunks, start = [], 0
        while start < len(ids):
            end = min(start + limit, len(ids))
            chunks.append(long_text[offsets[start][0]:offsets[end - 1][1]])
            if end == len(ids):
                break
            start += step
        return chunks

    @staticmethod
    def _to_numpy(value) -> np.ndarray:
        if is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).ravel()


def _ensure_model_downloaded(model_name: str, local_path: str):
    """Fetch the model on first use, so a fresh install just works."""
    if os.path.exists(os.path.join(local_path, 'config.json')):
        return
    print(f"OmniEmbedder: Downloading '{model_name}' to '{local_path}'...")
    snapshot_download(
        repo_id=model_name,
        local_dir=local_path,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"OmniEmbedder: Model '{model_name}' downloaded.")


def _worker_loop(input_queue, output_queue, cfg):
    """Command dispatch loop for the subprocess."""
    import setproctitle
    setproctitle.setproctitle("Anagnorisis-OmniEmbedder")
    from anagnorisis_core.config import ensure_writable_cache_dirs
    ensure_writable_cache_dirs(cfg, 'OmniEmbedder')

    try:
        embedder = _OmniEmbedderImpl(cfg)
        while True:
            task = input_queue.get()
            if task is None:
                break
            command, args, kwargs = task
            try:
                if not hasattr(embedder, command) or command.startswith('_'):
                    raise ValueError(f"Unknown command: {command}")
                output_queue.put(('success', getattr(embedder, command)(*args, **kwargs)))
            except Exception as exc:
                traceback.print_exc()
                output_queue.put(('error', exc))
    except Exception as exc:
        print(f"Critical error in OmniEmbedder worker process: {exc}")
        traceback.print_exc()


# ---------------------------------------------------------------------------
# Proxy — runs in the main process
# ---------------------------------------------------------------------------

class OmniEmbedder:
    """Process-wide handle to the embedding model.

    Singleton: constructing it is free and always returns the same instance, so
    any part of the app can ask for the embedder without threading it through.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OmniEmbedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return
        if cfg is None:
            raise ValueError("OmniEmbedder requires a configuration object (cfg) on first initialization.")

        self.cfg = cfg
        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._lock = threading.RLock()

        # Mirrored worker state — survives an unload so cache keys stay stable.
        self.embedding_dim = None
        self._model_hash = None
        self.max_seq_length = None
        # A plain string, not a torch.device: this is the main process, and
        # naming a device must not be what drags torch into it. Nothing outside
        # the worker reads this beyond reporting it.
        self.device = 'cpu'
        self._models_folder = None

        self._last_used_time = 0.0
        self._idle_timeout = float(getattr(cfg.embedder, 'idle_timeout_seconds', 120) or 120)
        self._shutdown_event = threading.Event()
        threading.Thread(target=self._monitor_idle, daemon=True,
                         name="OmniEmbedder-idle").start()

        self._initialized = True

    # -- public API -----------------------------------------------------

    def initiate(self, models_folder: str):
        """Load the model (downloading it first if needed) and mirror its state.

        Releases the worker again before returning: starting the app should
        learn the model's identity, not occupy the GPU with it. The next embed
        call respawns transparently.
        """
        self._models_folder = models_folder
        state = self._execute('initiate', models_folder)
        self._absorb(state)
        self.unload()
        return state

    @property
    def model_hash(self) -> Optional[str]:
        """Identity of the embedding model, whether or not it is loaded.

        Reported by the worker once it has run, and otherwise computed from the
        weight files on disk. The fallback is the important half: a search needs
        this to build cache keys, and loading a model to answer a search is
        exactly what this project does not do. Without it, a fresh process finds
        nothing it previously indexed.
        """
        if self._model_hash:
            return self._model_hash
        folder = self._models_folder or OmegaConf.select(
            self.cfg, 'main.embedding_models_path', default=None)
        if not folder:
            return None
        try:
            self._model_hash = compute_model_hash(self.cfg, folder)
        except Exception as exc:
            print(f'OmniEmbedder: could not fingerprint the model on disk ({exc}).')
            return None
        return self._model_hash

    @model_hash.setter
    def model_hash(self, value):
        self._model_hash = value

    def ensure_ready(self) -> Optional[str]:
        """Make sure the model has been loaded at least once; return its hash."""
        if not self.model_hash:
            self.initiate(self.cfg.main.embedding_models_path)
        return self.model_hash

    def embed_query(self, item: EmbedInput) -> np.ndarray:
        return self._execute('embed_query', item)

    def embed_document(self, item: EmbedInput) -> np.ndarray:
        return self._execute('embed_document', item)

    def embed_documents(self, items: List[EmbedInput]) -> np.ndarray:
        return self._execute('embed_documents', items)

    def embed_long_text(self, long_text: str) -> np.ndarray:
        return self._execute('embed_long_text', long_text)

    # Text of any length, as [n_chunks, D]. Named for the contract the training
    # pipeline and the module train.py files already speak.
    embed_text = embed_long_text

    def embed_file(self, file_path: str) -> np.ndarray:
        """Embed a media file by path.

        Remote files are streamed to a temporary local copy first, because the
        model's decoders need a real file. Callers that must not download remote
        content should check ``vfs.is_local_url`` before calling.
        """
        local_path, temp = vfs.resolve_to_local_path(file_path)
        try:
            return self._execute('embed_document', local_path)
        finally:
            if temp:
                try:
                    os.remove(temp)
                except OSError:
                    pass

    def unload(self):
        """Terminate the worker to free VRAM. Model state is preserved."""
        with self._lock:
            self._terminate_process()
        print("OmniEmbedder: Unloaded subprocess (model_hash preserved for restart).")

    # -- internals ------------------------------------------------------

    def _absorb(self, state: dict):
        if not state:
            return
        self.embedding_dim = state.get('embedding_dim', self.embedding_dim)
        self._model_hash = state.get('model_hash', self._model_hash)
        self.max_seq_length = state.get('max_seq_length', self.max_seq_length)
        device_type = state.get('device_type')
        if device_type:
            self.device = str(device_type)

    def _monitor_idle(self):
        while not self._shutdown_event.is_set():
            time.sleep(5)
            with self._lock:
                if self._process is not None and self._process.is_alive():
                    if self._last_used_time > 0 and time.time() - self._last_used_time > self._idle_timeout:
                        print(f"OmniEmbedder: Idle for {self._idle_timeout:.0f}s. "
                              f"Terminating subprocess to free GPU.")
                        self._terminate_process()

    def _terminate_process(self):
        if not self._process:
            return
        try:
            self._input_queue.put(None)
            self._process.join(timeout=1)
        except Exception:
            pass
        if self._process.is_alive():
            print("OmniEmbedder: Force killing subprocess...")
            self._process.terminate()
            self._process.join()
        self._process = None
        self._input_queue = None
        self._output_queue = None
        import gc
        gc.collect()

    def _ensure_process_running(self):
        """Start the worker if it is not running. Must hold self._lock."""
        if self._process is not None and self._process.is_alive():
            return
        print("OmniEmbedder: Starting worker subprocess...")
        ctx = multiprocessing.get_context('spawn')
        self._input_queue = ctx.Queue()
        self._output_queue = ctx.Queue()
        # daemon=True so an abandoned worker can never hold the process
        # open: multiprocessing joins non-daemon children at interpreter
        # exit, so a worker left running after a failed initiate hung the
        # CLI indefinitely. Safe here — this worker starts no
        # multiprocessing children of its own.
        self._process = ctx.Process(
            target=_worker_loop,
            args=(self._input_queue, self._output_queue, self.cfg),
            name="Anagnorisis-OmniEmbedder",
            daemon=True,
        )
        self._process.start()

        # A fresh worker holds no model, and every embed command needs one. Fall
        # back to the configured models directory when nobody has called
        # initiate() — otherwise on-demand loading silently does not happen and
        # the first embed fails with 'NoneType has no attribute encode_document'.
        folder = self._models_folder or OmegaConf.select(
            self.cfg, 'main.embedding_models_path', default=None)
        if folder:
            print("OmniEmbedder: Loading model in new subprocess...")
            self._models_folder = folder
            self._absorb(self._send('initiate', (folder,), {}))
        else:
            print("OmniEmbedder: WARNING no models directory configured; "
                  "the worker has no model to load.")

    def _send(self, command, args, kwargs):
        """Send one command and wait for its result. Must hold self._lock."""
        self._input_queue.put((command, args, kwargs))
        # Poll rather than block forever, so a worker that dies (OOM kill, CUDA
        # fault) surfaces as an error instead of hanging the caller.
        while True:
            try:
                status, result = self._output_queue.get(timeout=5)
                break
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    exit_code = self._process.exitcode if self._process else None
                    self._terminate_process()
                    raise RuntimeError(
                        f"OmniEmbedder subprocess died unexpectedly during "
                        f"'{command}' (exit code: {exit_code})."
                    )
        if status == 'error':
            raise result
        return result

    def _execute(self, command, *args, **kwargs):
        with self._lock:
            self._ensure_process_running()
            result = self._send(command, args, kwargs)
            self._last_used_time = time.time()
            return result

    def __del__(self):
        self._shutdown_event.set()
        try:
            self._terminate_process()
        except Exception:
            pass


def get_omni_embedder(cfg) -> OmniEmbedder:
    """Return the process-wide embedder."""
    return OmniEmbedder(cfg)


# ---------------------------------------------------------------------------
# Query side — CPU only, in-process
# ---------------------------------------------------------------------------

class QueryEmbedder:
    """Embeds search queries on the CPU, in this process. Never touches the GPU.

    Searching must stay responsive no matter what the background tasks are
    doing, and the GPU belongs to those tasks — they are the ones the user can
    see and pause in the Task Manager. A search that queued behind them, or
    stole VRAM from them, would be neither predictable nor visible.

    Loads the whole model by default (``embedder.query_modality: omni``) so a
    query can be a *file* as well as a phrase — drop an image, a clip or a song
    into the search box and find things like it, the way a visual search works.
    Every tower is frozen and shared with the GPU worker's copy, so a query
    vector lands in exactly the same space as the indexed file embeddings.

    Set ``query_modality: text`` to load only the text tower (a third of the
    weights) on a RAM-constrained machine — text queries still work, but
    searching by an image or a clip does not.
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self, cfg):
        self.cfg = cfg
        self.modality = str(getattr(cfg.embedder, 'query_modality', 'omni') or 'omni')
        self._model = None
        self._load_lock = threading.Lock()
        self._load_attempted = False

    @classmethod
    def get_instance(cls, cfg) -> "QueryEmbedder":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(cfg)
            return cls._instance

    # -- public API -----------------------------------------------------

    def embed_query(self, item) -> Optional[np.ndarray]:
        """Embed a search query — a phrase, or a path to an image/audio/video."""
        return self._encode(item, query_side=True)

    def embed_query_file(self, file_path: str) -> Optional[np.ndarray]:
        """Embed a file the user is searching *with*, on the CPU.

        This is the "find things like this one" path, so it runs on demand
        rather than in a background pass — it is one file, triggered by the
        user, and it must not queue behind GPU work.
        """
        if self.modality == 'text':
            print("[QueryEmbedder] query_modality is 'text'; cannot embed a media file.")
            return None
        local_path, temp = vfs.resolve_to_local_path(file_path)
        try:
            return self._encode(local_path, query_side=True)
        finally:
            if temp:
                try:
                    os.remove(temp)
                except OSError:
                    pass

    def embed_document(self, text: str) -> Optional[np.ndarray]:
        """Embed text as a document — used for tag vocabularies."""
        return self._encode(text, query_side=False)

    def unload(self):
        with self._load_lock:
            self._model = None
            self._load_attempted = False
            import gc
            gc.collect()

    # -- internals ------------------------------------------------------

    def _truncate_dim(self) -> Optional[int]:
        dim = getattr(self.cfg.embedder, 'embedding_dimension', None)
        return int(dim) if dim else None

    def _encode(self, text: str, query_side: bool) -> Optional[np.ndarray]:
        if not self._ensure_loaded():
            return None
        try:
            import torch as _torch
            with _torch.no_grad():
                encode = self._model.encode_query if query_side else self._model.encode_document
                return np.asarray(
                    encode(text, truncate_dim=self._truncate_dim()), dtype=np.float32
                ).ravel()
        except Exception as exc:
            print(f"[QueryEmbedder] Failed to embed on CPU: {exc}")
            return None

    def _ensure_loaded(self) -> bool:
        if self._load_attempted:
            return self._model is not None
        with self._load_lock:
            if self._load_attempted:
                return self._model is not None
            model_name = self.cfg.embedder.model_name
            local_path = os.path.join(
                self.cfg.main.embedding_models_path, model_name.replace('/', '__')
            )
            if not os.path.exists(os.path.join(local_path, 'config.json')):
                # The background worker downloads the model; until it has, search
                # simply has no query vector rather than blocking on a download.
                # Not marked as attempted: the background worker may still be
                # downloading, and the next search should try again.
                print(f"[QueryEmbedder] Model not present at {local_path} yet.")
                return False
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(
                    local_path,
                    trust_remote_code=True,
                    device='cpu',
                    model_kwargs={
                        'default_task': getattr(self.cfg.embedder, 'task', 'retrieval'),
                        'modality': self.modality,
                    },
                )
                self._model.eval()
                # The search path is the more exposed one: a query pasted into
                # the search bar reaches this directly.
                _block_url_fetching()
                self._load_attempted = True
                print(f"[QueryEmbedder] Loaded on CPU for search queries "
                      f"(modality={self.modality!r}).")
                return True
            except Exception as exc:
                print(f"[QueryEmbedder] Failed to load the text tower on CPU: {exc}")
                self._model = None
                return False


def get_query_embedder(cfg) -> QueryEmbedder:
    """Return the process-wide CPU query embedder."""
    return QueryEmbedder.get_instance(cfg)
