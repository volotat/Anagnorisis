"""Turns a file into a paragraph of English.

That paragraph is the project's universal currency. Metadata search indexes it,
the evaluator rates from it, and it is the only thing a remote server publishes
about a file (in `{filename}.meta`) — embeddings are not portable between
machines running different models, but a sentence is. So this module is what
lets an image, a song and a video be searched in the same list as a text file.

Descriptions are expensive — seconds per file, against milliseconds for an
embedding — so everything here is built around not doing the work twice. The
model lives in a subprocess that is killed after five idle minutes, and results
are cached under a key containing the model's identity (see
:mod:`anagnorisis_core.models.model_identity`).

The model is `gemma-4-E2B-it`: natively multimodal, natively supported by
transformers. Its predecessor here shipped its own modelling code through
`trust_remote_code`, which stopped loading entirely when transformers 5 renamed
an attribute the vendored code depended on. Nothing in this file should ever
depend on a third party's code tracking a transformers release again.

Gemma accepts at most 30 seconds of audio and 60 seconds of video per turn, so
long files are sampled: several short windows spread across the timeline, each
described, then synthesised into one summary. A two-hour film is therefore
described by what happens throughout it, not by its first minute.
"""
import os
import re
import time
import traceback
import threading
import multiprocessing
import queue
from typing import Optional, List, Dict, Sequence

import numpy as np
import torch
import setproctitle

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.models.model_identity import fingerprint_model_dir

# Gemma 4's audio input is a *fixed* window, not a maximum: the processor emits
# `audio_seq_length` = 750 soft tokens at `audio_ms_per_token` = 40 ms, which is
# exactly 30 seconds. A shorter clip produces proportionally fewer tokens — a 10
# second window gives 250 — and the model then does not recognise them as audio
# at all. It answers "Please provide the audio you would like me to describe",
# fluently and with no error anywhere.
#
# That single wrong assumption ("keep windows comfortably below the maximum")
# cost every audio description in the library. Measured on one track:
#
#     10s slice  -> 250 tokens -> "Please provide the audio..."
#     30s slice  -> 750 tokens -> "...a solo performance, the main instrument
#                                  featured is a violin, carrying the melody..."
#     10s padded -> 750 tokens -> a real description
#
# So every clip is padded to exactly this length before it reaches the model.
AUDIO_WINDOW_SECONDS = 30.0
AUDIO_SAMPLE_RATE = 16000       # everything here resamples to this
AUDIO_WINDOW_SAMPLES = int(AUDIO_WINDOW_SECONDS * AUDIO_SAMPLE_RATE)

# Kept as the old name so callers and configuration read the same as before.
MAX_AUDIO_SECONDS = AUDIO_WINDOW_SECONDS


def fit_audio_window(clip):
    """Pad or trim one clip to the exact window the model expects.

    Padding is silence at the end, which the model treats as silence rather than
    as content — a 5 second clip padded to 30 describes those 5 seconds. Trimming
    only bites if a caller hands over more than the window holds; the extra would
    be dropped by the processor anyway, less predictably.
    """
    import numpy as np

    clip = np.asarray(clip, dtype=np.float32).reshape(-1)
    if len(clip) > AUDIO_WINDOW_SAMPLES:
        return clip[:AUDIO_WINDOW_SAMPLES]
    if len(clip) < AUDIO_WINDOW_SAMPLES:
        return np.pad(clip, (0, AUDIO_WINDOW_SAMPLES - len(clip)))
    return clip


# --- The Worker Implementation (Runs in separate process) ---

# A description the model produced without ever receiving the media. It happens
# often enough to matter — roughly a third of one audio library — and the result
# is worse than a failure, because it is fluent, plausible and permanent once it
# is in a .meta file. Every pattern below is taken from descriptions actually
# written into sidecars, not imagined:
#
#   "The audio consists entirely of requests for an audio file or link so that
#    a description can be provided."
#   "Please provide the audio so I can describe it for you."
#   "No audio was provided, so I cannot write a description."
#   "The provided text consists entirely of the letter "A" repeated many times"
#
# The last is the model reading its own placeholder tokens as content. The
# middle ones are single-turn refusals; the first is what _synthesise produces
# when it is handed five of those and asked to summarise them.
#
# These are deliberately narrow. "The provided text is a complex mathematical
# expression" and "The white areas provide high contrast" are real descriptions
# from the same library and must not match — which is why the patterns anchor on
# *asking to be given* the media rather than on the word "provide".
_REFUSALS = [
    re.compile(p, re.IGNORECASE) for p in (
        r'\b(?:please\s+)?provide\s+(?:me\s+)?(?:with\s+)?(?:the|an|a)\s+'
        r'(?:audio|image|video|media|file|link|text)\b',
        r'\brequests?\s+for\s+(?:an?\s+)?(?:audio|image|video|media)?\s*'
        r'(?:files?|links?|input)\b',
        r'\bno\s+(?:audio|image|video|media|file|text)\b[^.]{0,40}\b'
        r'(?:was|is|were)?\s*(?:provided|given|attached|supplied)\b',
        r'\bno\s+description\b[^.]{0,60}\bcan\s+be\s+(?:provided|written|made)\b',
        r'\bI\s+(?:need|require)\s+(?:an?|the)\s+'
        r'(?:audio|image|video|media|file|text)\b',
        r'\b(?:cannot|can\'t|could\s+not|unable\s+to)\s+'
        r'(?:hear|see|listen|view|access|open|process|analyse|analyze)\b',
        # "consists entirely of the letter "A" repeated many times" and
        # "a sequence of the letter "A" repeated many times" — the shared part is
        # the repetition, not how the sentence opens.
        r'\bletter\s+"?\w"?\s+repeated\b',
        r'\bas\s+an\s+AI\b[^.]{0,60}\b(?:cannot|unable|don\'t\s+have)\b',
    )
]


def looks_like_a_refusal(text: str) -> bool:
    """True if the model answered without having received the media.

    Kept as a function rather than a method so the patterns can be tested
    directly against the strings that prompted them.
    """
    return bool(text) and any(p.search(text) for p in _REFUSALS)


class _OmniDescriptorImpl:
    """
    The actual implementation that runs inside the subprocess.

    Holds the model and the CUDA context, so that terminating the process is
    enough to give the GPU back — there is no way to fully release a CUDA
    context from inside a live Python process.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.processor = None
        self.model_name = cfg.omni.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_hash = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initiate(self, models_folder: str):
        if self.model is not None:
            return

        local_path = os.path.join(models_folder, self.model_name.replace('/', '__'))
        _ensure_model_downloaded(self.model_name, local_path)

        # Hash the files, not the loaded model — cheap, and it means a failed
        # load still cannot poison the cache with a different key.
        self.model_hash = self._calculate_model_hash(local_path)

        self._check_free_vram()
        self._load_model(local_path)

        print(f"OmniDescriptor (Worker): Initiated '{self.model_name}' on {self.device}.")

    def _check_free_vram(self):
        """Refuse to load when the GPU is too full to hold the model.

        The embedder and this descriptor cannot both be resident on an 8 GB
        card. Failing here is deliberate and safe: the caller is a background
        scheduler that will simply try again on its next tick, whereas an
        out-of-memory abort takes the whole worker down mid-batch.
        """
        required_mb = self.cfg.omni.get('min_free_vram_to_run', None)
        if not required_mb or self.device.type != 'cuda':
            return
        free_bytes, _total = torch.cuda.mem_get_info()
        free_mb = free_bytes / (1024 * 1024)
        if free_mb < required_mb:
            raise RuntimeError(
                f"OmniDescriptor: only {free_mb:.0f} MB of VRAM free, need "
                f"{required_mb} MB. Not loading — another model is probably "
                f"resident. This pass will be retried later."
            )

    # The lookup table that makes a 5.5 B model "effectively 2 B". It is 2.35 B
    # of those parameters — 4.7 GB — and bitsandbytes quantises Linear layers
    # but not embeddings, so it stays full size no matter the quantization.
    _PER_LAYER_EMBEDDING = 'model.language_model.embed_tokens_per_layer'

    def _load_model(self, local_path: str):
        """Load Gemma 4, quantised and split to fit on a consumer card."""
        from transformers import (AutoProcessor, AutoModelForMultimodalLM,
                                  BitsAndBytesConfig)

        load_kwargs = dict(dtype=torch.bfloat16, device_map='auto')

        if self.cfg.omni.get('load_in_4bit', True):
            print("OmniDescriptor (Worker): Loading with 4-bit NF4 quantization...")
            # The compute dtype must match the dtype above: the vision and audio
            # towers are not quantised, and handing a bfloat16 conv weight a
            # float16 input raises a dtype mismatch deep in the forward pass.
            load_kwargs['quantization_config'] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                # Required before accelerate will place any layer off the GPU.
                llm_int8_enable_fp32_cpu_offload=True,
            )
            if self.device.type == 'cuda':
                # accelerate's 'auto' refuses this model outright: at 4-bit it
                # still wants ~6.9 GB and 'auto' will not leave a card that full.
                # Pin everything to the GPU ourselves instead.
                load_kwargs['device_map'] = {'': 0}
                if self.cfg.omni.get('cpu_offload_embeddings', False):
                    # Last resort for cards under ~8 GB: park the embedding
                    # table in system RAM, taking the GPU side to ~2.2 GB.
                    # Measured cost is roughly 10x slower generation — the table
                    # is read once per generated token, so every token pays a
                    # round trip — plus ~5 GB of host RAM. Only worth it when
                    # the model otherwise does not fit at all.
                    load_kwargs['device_map'][self._PER_LAYER_EMBEDDING] = 'cpu'
        else:
            print("OmniDescriptor (Worker): Loading in bfloat16 (unquantized)...")

        try:
            self.processor = AutoProcessor.from_pretrained(local_path)
            self.model = AutoModelForMultimodalLM.from_pretrained(local_path, **load_kwargs)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to load OmniDescriptor model from '{local_path}': {e}"
            ) from e

    def _calculate_model_hash(self, local_path: str) -> str:
        """Identity of this descriptor, for the description cache key.

        The generation settings and the prompts are part of it: changing a
        prompt changes every description it would produce, while leaving the
        weights untouched. A hash that ignored them would keep serving
        descriptions written to the old prompt forever.
        """
        omni = self.cfg.omni
        return fingerprint_model_dir(
            local_path,
            self.model_name,
            omni.get('load_in_4bit', True),
            omni.get('max_new_tokens', 512),
            omni.get('do_sample', False),
            omni.get('temperature', 0.3),
            omni.get('image_prompt', ''),
            omni.get('audio_prompt', ''),
            omni.get('video_prompt', ''),
            omni.get('text_prompt', ''),
        )

    # ------------------------------------------------------------------
    # Generation — the single place this file talks to the model
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        images: Optional[Sequence] = None,
        audio: Optional[Sequence] = None,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Run one multimodal turn and return the generated text.

        Media is passed to the processor directly rather than as URLs in the
        message, because everything here is already in memory: PIL frames
        pulled out of a video, waveforms sliced out of a track.
        """
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        images = list(images) if images else []
        # Every clip, whatever produced it — a sampled window, a whole short
        # file, or the soundtrack under a video segment — has to arrive at the
        # exact window length. Done here because this is the one place they all
        # pass through, and getting it wrong is silent.
        audio = [fit_audio_window(clip) for clip in audio] if audio else []

        # The template emits one placeholder token per media item, in order,
        # and the processor expands each into the right number of soft tokens.
        content = ([{"type": "image", "image": img} for img in images]
                   + [{"type": "audio", "audio": clip} for clip in audio]
                   + [{"type": "text", "text": prompt}])
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.processor(
            text=[text],
            images=images or None,
            audio=audio or None,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs = {
            'max_new_tokens': max_new_tokens or self.cfg.omni.get('max_new_tokens', 512),
            'do_sample': self.cfg.omni.get('do_sample', False),
        }
        # Passing temperature alongside greedy decoding is a warning in
        # transformers 5 and does nothing, so only send it when it applies.
        if gen_kwargs['do_sample']:
            gen_kwargs['temperature'] = self.cfg.omni.get('temperature', 0.3)

        try:
            with torch.no_grad():
                out = self.model.generate(**inputs, **gen_kwargs)
            # Decode only what was generated; the prompt is echoed back
            # otherwise, and it would end up stored as the file's description.
            generated = out[0][inputs['input_ids'].shape[-1]:]
            text = self.processor.decode(generated, skip_special_tokens=True).strip()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Vetted here rather than at each caller, because this is the one place
        # every description passes through — a segment, a synthesis, and a
        # single-turn description alike. Returning nothing makes the caller
        # treat it as a failure, which is what it is: describe() then writes no
        # .meta, and the file is picked up again on a later run instead of being
        # stuck with a fluent description of nothing.
        if looks_like_a_refusal(text):
            print(f"  Discarded a description the model wrote without the media: "
                  f"{text[:80]}…")
            return ''
        return text

    # ------------------------------------------------------------------
    # Shared sampling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _segment_starts(total_s: float, n_samples: int, duration_s: float) -> List[float]:
        """Start times of evenly-spaced windows across a file.

        Keeps a small margin away from both ends, where intros, silence and
        credits tend to say little about the file as a whole.
        """
        if n_samples <= 1:
            return [max(0.0, (total_s / 2) - (duration_s / 2))]
        margin = min(duration_s / 2, total_s * 0.05)
        lo = margin
        hi = max(lo, total_s - duration_s - margin)
        step = (hi - lo) / max(n_samples - 1, 1)
        return [lo + i * step for i in range(n_samples)]

    def _synthesise(self, descriptions: List[str], total_s: float, media: str) -> str:
        """Fold per-segment descriptions into one description of the whole file."""
        if not descriptions:
            # Every segment failed or was vetted away. Raising rather than
            # returning a thin description means describe() records a failure and
            # writes nothing, so the file is tried again another day.
            raise RuntimeError(f"Failed to describe any {media} segments.")
        if len(descriptions) == 1:
            # Nothing to reconcile; drop the timestamp prefix.
            return descriptions[0].split(": ", 1)[-1]

        joined = "\n".join(descriptions)
        # The result is stored and indexed verbatim, so ask for the description
        # alone. Left to itself the model opens with "Based on the provided
        # samples…", which is preamble about the method rather than anything
        # searchable about the file.
        return self._generate(
            f"The following are descriptions of {len(descriptions)} sampled "
            f"segments from a {media} recording that is {total_s:.0f} seconds "
            f"long:\n\n{joined}\n\n"
            f"Write a concise description of the {media} as a whole. Write only "
            f"the description itself — no preamble, and no mention of samples, "
            f"segments or timestamps."
        )

    def _describe_segments(self, segments, media: str, prompt: str, total_s: float) -> str:
        """Describe each segment, tolerate individual failures, then synthesise.

        A single unreadable segment should cost that segment, not the file: a
        video whose middle is corrupt is still worth describing from the rest.
        """
        descriptions: List[str] = []
        for idx, (start_s, end_s, images, audio) in enumerate(segments):
            print(f"OmniDescriptor (Worker): {media} segment {idx + 1} "
                  f"[{start_s:.1f}s – {end_s:.1f}s] …")
            try:
                desc = self._generate(prompt, images=images, audio=audio)
                if not desc:
                    # Vetted away by _generate. Keeping it would be worse than
                    # dropping it: _synthesise summarises whatever it is handed,
                    # so five refusals become "the audio consists entirely of
                    # requests for an audio file" — fluent, wrong, and written
                    # to a .meta that nothing ever overwrites.
                    print(f"  Segment {idx + 1} produced no usable description.")
                    continue
                descriptions.append(f"[{start_s:.0f}s–{end_s:.0f}s]: {desc[:1024].strip()}")
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on segment {idx + 1} — skipping.")
                torch.cuda.empty_cache()
            except Exception as exc:
                print(f"  Error on segment {idx + 1}: {exc}")
        return self._synthesise(descriptions, total_s, media)

    def _prompt_for(self, kind: str) -> str:
        """The configured prompt for a media kind."""
        key = f'{kind}_prompt'
        prompt = self.cfg.omni.get(key, None)
        if not prompt:
            raise ValueError(f"Prompt not specified in config (cfg.omni.{key}).")
        return prompt

    # ------------------------------------------------------------------
    # Description entry points
    # ------------------------------------------------------------------

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description of an image."""
        from PIL import Image

        prompt = prompt or self._prompt_for('image')
        # PIL cannot open 'osfs://...'; resolve (and download, if remote) first.
        local_path, temp_to_cleanup = vfs.resolve_to_local_path(image_path)
        if not os.path.exists(local_path):
            raise ValueError(f"[OmniDescriptor] [Error: file not found — {image_path}]")

        try:
            image = Image.open(local_path).convert("RGB")
            max_size = self.cfg.omni.get('image_max_size', 512)
            if max(image.size) > max_size:
                scale = max_size / max(image.size)
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    resample=Image.BICUBIC,
                )
            return self._generate(prompt, images=[image])
        finally:
            _cleanup_temp(temp_to_cleanup)

    def describe_audio(self, audio_path: str, prompt: Optional[str] = None) -> str:
        """Describe an audio file short enough to be heard in one turn."""
        import librosa

        prompt = prompt or self._prompt_for('audio')
        local_path, temp_to_cleanup = vfs.resolve_to_local_path(audio_path)
        if not os.path.exists(local_path):
            raise ValueError(f"[OmniDescriptor] [Error: file not found — {audio_path}]")

        try:
            waveform, _ = librosa.load(local_path, sr=16000, mono=True,
                                       duration=MAX_AUDIO_SECONDS)
            return self._generate(prompt, audio=[waveform])
        finally:
            _cleanup_temp(temp_to_cleanup)

    def describe_audio_sampled(
        self,
        audio_path: str,
        n_samples: int = 3,
        sample_duration_s: float = AUDIO_WINDOW_SECONDS,
        prompt: Optional[str] = None,
    ) -> str:
        """Describe audio from three windows spread across it.

        Each window is a continuous 30 seconds — the whole of the model's audio
        input — described on its own, and the three descriptions are folded into
        one. Four generations rather than the six that five windows cost, and it
        still hears three places in the recording rather than one.

        **The windows have to be continuous.** Joining several short excerpts
        into a single window to save a pass does not work, and the failure is
        worth recording because it looks like it should. The model hears the cuts
        and describes the collage instead of the music — measured on one track,
        with the excerpts explicitly explained in the prompt:

            1 x 30s  "classical music, a solo performance of a piano piece,
                      moderate and flowing, reflective"
            2 x 15s  "a collection of short, fragmented musical excerpts"
            3 x 10s  "a compilation ... Without listening"
            5 x  6s  "a collection of short, distinct musical excerpts"

        Two different tracks produced near-identical text that way, which is
        useless for search — the point of the description.

        So covering a long recording means describing each window separately and
        folding the results together. Raise *n_samples* for more of the recording
        at one more generation each; drop it to 1 to skip the synthesis pass
        entirely and describe a single window.
        """
        import librosa

        prompt = prompt or self._prompt_for('audio')
        sample_duration_s = min(sample_duration_s, AUDIO_WINDOW_SECONDS)

        # librosa can't open 'osfs://...' URLs; resolve once at the top.
        local_path, temp_to_cleanup = vfs.resolve_to_local_path(audio_path)
        if not os.path.exists(local_path):
            raise ValueError(f"[OmniDescriptor] [Error: file not found — {audio_path}]")

        try:
            waveform, sr = librosa.load(local_path, sr=16000, mono=True)
            total_s = len(waveform) / sr

            # Short enough to hear whole — no need to sample it at all.
            if total_s <= AUDIO_WINDOW_SECONDS:
                return self._generate(prompt, audio=[waveform])

            width = int(sample_duration_s * sr)
            segments = []
            for start_s in self._segment_starts(total_s, n_samples, sample_duration_s):
                start = int(start_s * sr)
                end = min(start + width, len(waveform))
                segments.append((start / sr, end / sr, None, [waveform[start:end]]))

            # One window needs no synthesis pass: describe it and be done.
            if len(segments) == 1:
                return self._generate(prompt, audio=segments[0][3])

            return self._describe_segments(segments, 'audio', prompt, total_s)
        finally:
            _cleanup_temp(temp_to_cleanup)

    def describe_video_sampled(
        self,
        video_path: str,
        n_samples: int = 3,
        sample_duration_s: float = 10.0,
        frames_per_segment: int = 4,
        prompt: Optional[str] = None,
    ) -> str:
        """Describe a video from three windows of stills and sound.

        Three windows spread across the file, each described on its own and
        folded together — the same shape as audio, and for the same reason: it
        sees three places rather than one, for four generations rather than six.

        Stills and sound come from the same stretch of the video, because they
        have to be describing the same moment. Spreading the stills across the
        whole file and pairing them with one window of sound was tried and does
        two things wrong: it describes pictures and sound from different parts of
        the video as though they belonged together, and it does not fit.

        Four stills per window, because that is where the measurements land.
        Peak VRAM alongside a full audio window, on an 8 GB card:

            4 stills   7.47 GB   ~310 MB spare
            6 stills   7.76 GB   ~20 MB spare — and the same description, word
                                 for word, as four
            8 stills   out of memory

        So six is the ceiling and four is the useful maximum: the two extra
        stills bought nothing but the last of the headroom, and headroom is what
        stops a background pass failing when anything else touches the card.
        Raise `omni.video_frames_per_window` on a larger card.
        """
        import cv2

        prompt = prompt or self._prompt_for('video')
        sample_duration_s = min(sample_duration_s, AUDIO_WINDOW_SECONDS)
        frames_per_segment = int(
            self.cfg.omni.get('video_frames_per_window', frames_per_segment))

        # cv2/ffmpeg can't open 'osfs://...' URLs — resolve up front so every
        # subprocess call below gets a real path.
        local_path, temp_to_cleanup = vfs.resolve_to_local_path(video_path)
        if not os.path.exists(local_path):
            raise ValueError(f"[OmniDescriptor] [Error: file not found — {video_path}]")

        try:
            cap = cv2.VideoCapture(local_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {local_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
            cap.release()
            total_s = total_frames / fps if fps else 0.0
            if total_s <= 0:
                raise RuntimeError(f"Could not determine duration of: {video_path}")

            segments = []
            for start_s in self._segment_starts(total_s, n_samples, sample_duration_s):
                end_s = min(start_s + sample_duration_s, total_s)
                frames = self._extract_frames(local_path, start_s, end_s, frames_per_segment)
                if not frames:
                    print(f"  No frames extracted at {start_s:.1f}s — skipping segment.")
                    continue
                audio = self._extract_audio(local_path, start_s, end_s)
                segments.append((start_s, end_s, frames, audio))

            if not segments:
                raise RuntimeError(f"Could not extract any frames from: {video_path}")

            if len(segments) == 1:
                _, _, frames, audio = segments[0]
                return self._generate(prompt, images=frames, audio=audio)

            return self._describe_segments(segments, 'video', prompt, total_s)
        finally:
            _cleanup_temp(temp_to_cleanup)

    def describe_text(self, text: str, prompt: Optional[str] = None) -> str:
        """Generate a summary/description of a long text."""
        prompt = prompt or self._prompt_for('text')
        return self._generate(f"{prompt}\n\n{text}")

    # ------------------------------------------------------------------
    # Media extraction
    # ------------------------------------------------------------------

    def _extract_frames(self, local_path: str, start_s: float, end_s: float,
                        count: int) -> List:
        """Evenly-spaced frames from one window, as PIL images.

        ffmpeg first: it scales during decode and stays quiet about pixel
        formats. cv2 is the fallback for the files ffmpeg refuses.
        """
        import io
        import subprocess
        import cv2
        from PIL import Image as PILImage

        if count <= 0:
            return []

        step = (end_s - start_s) / count
        timestamps = [start_s + (i + 0.5) * step for i in range(count)]

        max_size = self.cfg.omni.get('video_sample_max_size', None)
        scale_filter = (
            f"scale='if(gt(iw,ih),{max_size},-1)':'if(gt(ih,iw),{max_size},-1)'"
            if max_size else None
        )

        frames = []
        for t in timestamps:
            # Option order is load-bearing: -ss and -hwaccel configure the
            # input and must precede -i, while -vf describes the output and
            # must follow it. Putting the filter first makes ffmpeg reject the
            # whole command, which silently pushed every extraction onto the
            # OpenCV fallback — and OpenCV cannot decode AV1 at all.
            cmd = ['ffmpeg', '-loglevel', 'error', '-hide_banner',
                   '-hwaccel', 'none', '-err_detect', 'ignore_err',
                   '-ss', str(t), '-i', local_path]
            if scale_filter:
                cmd += ['-vf', scale_filter]
            cmd += ['-vframes', '1', '-pix_fmt', 'rgb24',
                    '-f', 'image2pipe', '-vcodec', 'png', 'pipe:1']
            try:
                proc = subprocess.run(cmd, capture_output=True, timeout=300)
                if proc.returncode == 0 and proc.stdout:
                    frames.append(PILImage.open(io.BytesIO(proc.stdout)).convert('RGB'))
                    continue
            except Exception:
                pass

            # ffmpeg failed on this frame — fall back to OpenCV.
            cap = cv2.VideoCapture(local_path)
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ok, frame = cap.read()
            cap.release()
            if ok:
                image = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_size and max(image.size) > max_size:
                    scale = max_size / max(image.size)
                    image = image.resize(
                        (int(image.width * scale), int(image.height * scale)),
                        resample=PILImage.BICUBIC,
                    )
                frames.append(image)
        return frames

    def _extract_audio(self, local_path: str, start_s: float, end_s: float) -> Optional[List]:
        """The soundtrack under one window, or None if there isn't one.

        Silent videos are ordinary, so a missing audio track is not an error —
        the segment is simply described from its frames.
        """
        import io
        import subprocess
        import librosa

        try:
            proc = subprocess.run(
                ['ffmpeg', '-y', '-loglevel', 'error',
                 '-ss', str(start_s), '-t', str(end_s - start_s),
                 '-i', local_path, '-ac', '1', '-ar', '16000', '-f', 'wav', 'pipe:1'],
                capture_output=True, timeout=60,
            )
            if proc.returncode == 0 and proc.stdout:
                waveform, _ = librosa.load(io.BytesIO(proc.stdout), sr=16000, mono=True)
                if waveform.size:
                    return [waveform]
        except Exception as exc:
            print(f"  Audio extraction failed at {start_s:.1f}s: {exc}")
        return None


def _cleanup_temp(path: Optional[str]):
    """Remove a temp file left behind by resolving a remote path."""
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def _ensure_model_downloaded(model_name: str, local_path: str):
    """Fetch the model on first use, so a fresh install just works."""
    if os.path.exists(os.path.join(local_path, 'config.json')):
        return
    from huggingface_hub import snapshot_download
    print(f"OmniDescriptor: Downloading '{model_name}' to '{local_path}'...")
    snapshot_download(repo_id=model_name, local_dir=local_path)
    print(f"OmniDescriptor: Downloaded '{model_name}'.")


def _worker_loop(input_queue, output_queue, cfg):
    """The loop running in the separate process."""
    setproctitle.setproctitle("Anagnorisis-OmniDescriptor")
    # Before anything imports librosa (lazily, inside the audio and video paths)
    # or touches CUDA: both want a writable place for compiled artefacts and a
    # non-root container user has neither.
    from anagnorisis_core.config import ensure_writable_cache_dirs
    ensure_writable_cache_dirs(cfg, 'OmniDescriptor')

    try:
        descriptor = _OmniDescriptorImpl(cfg)

        while True:
            try:
                task = input_queue.get()
                if task is None:  # Sentinel to stop
                    break

                command, args, kwargs = task

                if command == 'initiate':
                    descriptor.initiate(*args, **kwargs)
                    output_queue.put(('success', {
                        'device_type': descriptor.device.type,
                        'model_hash': descriptor.model_hash,
                    }))

                elif hasattr(descriptor, command):
                    method = getattr(descriptor, command)
                    output_queue.put(('success', method(*args, **kwargs)))
                else:
                    output_queue.put(('error', ValueError(f"Unknown command: {command}")))

            except Exception as e:
                traceback.print_exc()
                output_queue.put(('error', e))

    except Exception as e:
        print(f"Critical error in OmniDescriptor worker process: {e}")
        traceback.print_exc()


# --- The Proxy Class (Runs in main process) ---

class OmniDescriptor:
    """
    A singleton proxy class that manages a subprocess for omni-modal description.
    It ensures the subprocess is terminated after a period of inactivity.
    Converts images, audio, video and text into text descriptions.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OmniDescriptor, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, cfg=None):
        if self._initialized:
            return

        if cfg is None:
            raise ValueError("OmniDescriptor requires a configuration object (cfg) on first initialization.")

        self.cfg = cfg
        self._process = None
        self._input_queue = None
        self._output_queue = None
        self._lock = threading.Lock()

        # State mirroring
        self.device = torch.device('cpu')  # Updated to actual device after initiate()
        self.model = "ProxyModel"  # Dummy to satisfy checks
        self._models_folder = None
        self.model_hash = None

        # Idle management
        self._last_used_time = 0
        self._idle_timeout = 300  # 5 minutes (model is large)
        self._shutdown_event = threading.Event()
        self._monitor_thread = threading.Thread(target=self._monitor_idle, daemon=True)
        self._monitor_thread.start()

        self._initialized = True

    def _monitor_idle(self):
        """Background thread to kill the process when idle."""
        while not self._shutdown_event.is_set():
            time.sleep(5)
            with self._lock:
                if self._process is not None and self._process.is_alive():
                    if self._last_used_time > 0 and time.time() - self._last_used_time > self._idle_timeout:
                        print(f"OmniDescriptor: Idle for {self._idle_timeout}s. Terminating subprocess to free GPU.")
                        self._terminate_process()

    def _terminate_process(self):
        """Terminates the worker process immediately."""
        if self._process:
            try:
                self._input_queue.put(None)
                self._process.join(timeout=1)
            except Exception:
                pass

            if self._process.is_alive():
                print("OmniDescriptor: Force killing subprocess...")
                self._process.terminate()
                self._process.join()

            self._process = None
            self._input_queue = None
            self._output_queue = None

            import gc
            gc.collect()

    def unload(self):
        """
        Immediately terminate the worker subprocess to free GPU/CPU memory.
        model_hash and _models_folder are preserved so the process restarts
        transparently on the next call.
        """
        with self._lock:
            self._terminate_process()
        print("OmniDescriptor: Unloaded subprocess (model_hash preserved for restart).")

    def _ensure_process_running(self):
        """Starts the process if it's not running. Must be called within self._lock."""
        if self._process is None or not self._process.is_alive():
            print("OmniDescriptor: Starting worker subprocess...")
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
                name="Anagnorisis-OmniDescriptor",
                daemon=True,
            )
            self._process.start()

            # Re-initiate if previously loaded
            if self._models_folder:
                print("OmniDescriptor: Re-initiating model in new subprocess...")
                self._send_command_internal('initiate', (self._models_folder,), {})

    def _send_command_internal(self, command, args, kwargs):
        """Helper to send command and wait for result. Assumes lock is held."""
        self._input_queue.put((command, args, kwargs))
        while True:
            try:
                status, result = self._output_queue.get(timeout=5)
                break
            except queue.Empty:
                if self._process is None or not self._process.is_alive():
                    exit_code = self._process.exitcode if self._process else None
                    self._terminate_process()
                    raise RuntimeError(
                        f"OmniDescriptor subprocess died unexpectedly during "
                        f"'{command}' (exit code: {exit_code})."
                    )
                # Still alive — keep waiting.

        if status == 'error':
            raise result
        return result

    def _execute(self, command, *args, **kwargs):
        """Public wrapper to execute commands safely."""
        with self._lock:
            self._ensure_process_running()
            result = self._send_command_internal(command, args, kwargs)
            self._last_used_time = time.time()
            return result

    # --- Public Interface ---

    def initiate(self, models_folder: str):
        """Initialize the model. Downloads if necessary."""
        self._models_folder = models_folder
        res = self._execute('initiate', models_folder)
        self.model_hash = res.get('model_hash', 'unknown_hash')

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description of an image file."""
        return self._execute('describe_image', image_path, prompt)

    def describe_audio(self, audio_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description/transcription of a short audio file."""
        return self._execute('describe_audio', audio_path, prompt)

    def describe_audio_sampled(
        self,
        audio_path: str,
        n_samples: int = 3,
        sample_duration_s: float = AUDIO_WINDOW_SECONDS,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe audio from one 30-second window, or several if asked.

        Three windows by default: each is described separately and the results
        folded together, costing a generation per window plus one more. Raise it
        for more coverage, or drop it to 1 to describe a single window with no
        synthesis pass.

        These defaults are the ones that count: they are passed explicitly to the
        worker, so they override the implementation's own rather than deferring
        to them.
        """
        return self._execute(
            'describe_audio_sampled', audio_path, n_samples, sample_duration_s, prompt
        )

    def describe_video_sampled(
        self,
        video_path: str,
        n_samples: int = 3,
        sample_duration_s: float = 10.0,
        frames_per_segment: int = 4,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe a video from one window of stills and sound, or several.

        Each window contributes ``frames_per_segment`` stills and the sound
        underneath them. Three windows by default, described separately and
        folded together; drop to 1 for a single window and no synthesis pass.
        """
        return self._execute(
            'describe_video_sampled',
            video_path, n_samples, sample_duration_s, frames_per_segment, prompt
        )

    def describe_text(self, text: str, prompt: Optional[str] = None) -> str:
        """Generate a summary/description of text content."""
        return self._execute('describe_text', text, prompt)

    def __del__(self):
        try:
            self._shutdown_event.set()
            with self._lock:
                self._terminate_process()
        except Exception:
            pass


if __name__ == '__main__':
    # Tier-2 smoke test: needs a GPU and the model on disk.
    #   python3 -m anagnorisis_core.models.descriptor
    import sys
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('config.yaml')
    models_folder = cfg.main.get('embedding_models_path', './models')

    descriptor = OmniDescriptor(cfg)
    descriptor.initiate(models_folder)
    print(f"model_hash: {descriptor.model_hash}")

    for arg in sys.argv[1:]:
        ext = os.path.splitext(arg)[1].lower()
        if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.gif'}:
            print(f"\n--- {arg}\n{descriptor.describe_image(arg)}")
        elif ext in {'.mp3', '.wav', '.flac', '.ogg', '.m4a'}:
            print(f"\n--- {arg}\n{descriptor.describe_audio_sampled(arg)}")
        elif ext in {'.mp4', '.mkv', '.avi', '.webm', '.mov'}:
            print(f"\n--- {arg}\n{descriptor.describe_video_sampled(arg)}")
        else:
            with open(arg, 'r', errors='replace') as fh:
                print(f"\n--- {arg}\n{descriptor.describe_text(fh.read()[:8000])}")

    descriptor.unload()
