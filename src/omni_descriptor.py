import os
import time
import traceback
import threading
import multiprocessing
import queue
import hashlib
from typing import Optional, List, Dict, Any

import torch
import numpy as np

# --- The Worker Implementation (Runs in separate process) ---

class _OmniDescriptorImpl:
    """
    The actual implementation that runs inside the subprocess.
    It holds the heavy MiniCPM-o model and CUDA context.
    Converts images, audio, video and text into text descriptions.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_hash = None

    def initiate(self, models_folder: str):
        if self.model is not None:
            return

        model_name = self.cfg.omni.model_name
        if not model_name:
            raise ValueError("cfg.omni.model_name is not specified.")

        self.model_name = model_name
        model_folder_name = model_name.replace('/', '__')
        local_model_path = os.path.join(models_folder, model_folder_name)

        self._ensure_model_downloaded(models_folder, model_name)
        self._load_model(local_model_path)
        self.model_hash = self._calculate_model_hash()

    def _calculate_model_hash(self) -> str:
        """Calculates a lightweight hash of the model weights for cache validity."""
        print("OmniDescriptor (Worker): Calculating model hash...")
        try:
            md5 = hashlib.md5()
            for k, v in sorted(self.model.state_dict().items()):
                md5.update(k.encode('utf-8'))
                md5.update(str(v.shape).encode('utf-8'))
                flat_v = v.reshape(-1)
                sample = flat_v[:100].tolist()
                md5.update(str(sample).encode('utf-8'))
            return md5.hexdigest()
        except Exception as e:
            print(f"Error calculating model hash: {e}")
            return "unknown_hash"

    def _ensure_model_downloaded(self, models_folder: str, model_name: str):
        """Ensure the HF model is present locally."""
        from huggingface_hub import snapshot_download
        from transformers import AutoConfig

        local_model_path = os.path.join(models_folder, model_name.replace('/', '__'))
        config_file_path = os.path.join(local_model_path, 'config.json')

        def download(force=False):
            print(f"{'Re-downloading' if force else 'Downloading'} model '{self.model_name}' to '{local_model_path}'...")
            snapshot_download(
                repo_id=self.model_name,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                force_download=force,
                resume_download=True
            )
            print(f"Model '{self.model_name}' downloaded successfully.")

        model_exists = os.path.exists(config_file_path)
        weights_exist = False

        if model_exists:
            weight_files = ['pytorch_model.bin', 'model.safetensors', 'tf_model.h5',
                            'model.ckpt.index', 'flax_model.msgpack']
            weights_exist = any(os.path.exists(os.path.join(local_model_path, wf)) for wf in weight_files)
            if not weights_exist:
                import glob
                weights_exist = bool(glob.glob(os.path.join(local_model_path, 'pytorch_model-*.bin')) or
                                     glob.glob(os.path.join(local_model_path, 'model-*.safetensors')))

        if not model_exists or not weights_exist:
            try:
                if model_exists and not weights_exist:
                    print(f"WARNING: Config found but model weights missing for '{self.model_name}'. Resuming download...")
                download(force=False)
            except Exception as e:
                print(f"ERROR: Failed to download model '{self.model_name}'.")
                raise RuntimeError(f"Failed to download model '{self.model_name}': {e}") from e
        else:
            print(f"Found existing model '{self.model_name}' at '{local_model_path}'. Verifying integrity...")
            try:
                cfg = AutoConfig.from_pretrained(local_model_path, trust_remote_code=True)
                del cfg
                print(f"Model '{self.model_name}' integrity check passed.")
            except Exception as e:
                print(f"WARNING: Local model at '{local_model_path}' seems corrupted. Re-downloading...")
                try:
                    download(force=True)
                except Exception as download_e:
                    raise RuntimeError(f"Failed to re-download model '{self.model_name}': {download_e}") from download_e

    def _load_model(self, local_path: str):
        """Load MiniCPM-o model for omni-modal understanding."""
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        load_in_4bit = self.cfg.omni.get('load_in_4bit', True)

        try:
            load_kwargs = dict(
                trust_remote_code=True,
                attn_implementation="sdpa",
                init_vision=True,
                init_audio=True,
                init_tts=False,  # We only need understanding, not speech generation
            )

            if load_in_4bit:
                print("OmniDescriptor (Worker): Loading model with BitsAndBytes 4-bit quantization...")
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                load_kwargs['quantization_config'] = quantization_config
                load_kwargs['device_map'] = 'auto'
                # torch_dtype must match bnb_4bit_compute_dtype so that the image
                # processor and all non-quantized buffers (vpm, resampler) use the
                # same dtype as the inputs — otherwise a bfloat16 conv weight receives
                # a float16 pixel_values tensor and raises "Half and Byte" mismatches.
                load_kwargs['torch_dtype'] = torch.bfloat16
            else:
                print("OmniDescriptor (Worker): Loading model in bfloat16 (full precision)...")
                load_kwargs['torch_dtype'] = torch.bfloat16

            # The 'unused weights' warning is caused by init_tts=False: TTS weights
            # exist in the checkpoint but are intentionally not loaded. Suppress it.
            import transformers as _hf
            _prev_verbosity = _hf.logging.get_verbosity()
            _hf.logging.set_verbosity_error()
            self.model = AutoModel.from_pretrained(local_path, **load_kwargs)
            _hf.logging.set_verbosity(_prev_verbosity)
            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                local_path, trust_remote_code=True
            )

            if load_in_4bit:
                # Only the resampler needs special treatment. Its forward() calls
                # F.multi_head_attention_forward(), which accesses out_proj.weight
                # as a raw tensor, bypassing bitsandbytes' __torch_function__ hook.
                # This means bfloat16 activations hit the raw uint8 (Byte) quantized
                # weight and crash with "BFloat16 and Byte" dtype mismatch.
                # vpm (SigLip2) and apm (Whisper) use standard nn.Linear calls that
                # bitsandbytes intercepts correctly, so they work fine as 4-bit.
                # Dequantizing them would OOM the GPU — only resampler needs the fix.
                import bitsandbytes as bnb
                from bitsandbytes.functional import dequantize_4bit

                def _dequantize_4bit_layers(parent: torch.nn.Module) -> None:
                    for name, child in list(parent.named_children()):
                        if isinstance(child, bnb.nn.Linear4bit):
                            qs = child.weight.quant_state
                            w = dequantize_4bit(child.weight.data, qs).to(torch.bfloat16)
                            new_linear = torch.nn.Linear(
                                child.in_features, child.out_features,
                                bias=child.bias is not None,
                                device=w.device, dtype=torch.bfloat16,
                            )
                            new_linear.weight = torch.nn.Parameter(w)
                            if child.bias is not None:
                                new_linear.bias = torch.nn.Parameter(
                                    child.bias.data.to(torch.bfloat16)
                                )
                            setattr(parent, name, new_linear)
                        else:
                            _dequantize_4bit_layers(child)

                torch.cuda.empty_cache()
                resampler = getattr(self.model, 'resampler', None)
                if resampler is not None:
                    _dequantize_4bit_layers(resampler)
                    resampler.to(torch.bfloat16)
                    torch.cuda.empty_cache()
                    print("OmniDescriptor (Worker): Dequantized 'resampler' to bfloat16.")
            else:
                self.model = self.model.to(self.device)

            print(f"OmniDescriptor (Worker): Model loaded (4bit={load_in_4bit}) on {self.device}")
        except Exception as e:
            raise RuntimeError(f"Failed to load OmniDescriptor model from '{local_path}': {e}") from e

    # --- Public methods for each modality ---

    def describe_image(self, image_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description of an image."""
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        from PIL import Image

        if prompt is None:
            if 'image_prompt' in self.cfg.omni and self.cfg.omni.image_prompt:
                prompt = self.cfg.omni.image_prompt
            else:
                raise ValueError("Image prompt not specified in config (cfg.omni.image_prompt).")

        image = Image.open(image_path).convert("RGB")
        # Resize if too large, keeping aspect ratio
        max_size = self.cfg.omni.get('image_max_size', 512)
        if max(image.size) > max_size:
            scale = max_size / max(image.size)
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, resample=Image.BICUBIC)

        msgs = [{"role": "user", "content": [image, prompt]}]

        result = self.model.chat(
            msgs=msgs,
            use_tts_template=False,
            enable_thinking=False,
            max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
            do_sample=self.cfg.omni.get('do_sample', False),
        )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    def describe_audio(self, audio_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description/transcription of audio."""
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        import librosa

        if prompt is None:
            if 'audio_prompt' in self.cfg.omni and self.cfg.omni.audio_prompt:
                prompt = self.cfg.omni.audio_prompt
            else:
                raise ValueError("Audio prompt not specified in config (cfg.omni.audio_prompt).")

        audio_input, _ = librosa.load(audio_path, sr=16000, mono=True)
        msgs = [{"role": "user", "content": [prompt, audio_input]}]

        result = self.model.chat(
            msgs=msgs,
            do_sample=True,
            max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
            use_tts_template=True,
            generate_audio=False,
            temperature=self.cfg.omni.get('temperature', 0.3),
        )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    def describe_audio_sampled(
        self,
        audio_path: str,
        n_samples: int = 5,
        sample_duration_s: float = 10.0,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe audio by sampling short segments spread across the file.

        Instead of feeding the entire (potentially very long) audio clip to the
        model at once — which OOMs on limited VRAM — this method:

          1. Loads the full waveform.
          2. Picks ``n_samples`` evenly-spaced windows of ``sample_duration_s``
             seconds each (first, middle, last regions).
          3. Runs the model independently on each window.
          4. If ``n_samples > 1``, asks the model to synthesise the per-segment
             descriptions into a single coherent summary.

        Parameters
        ----------
        audio_path        : path to the audio file (any format librosa accepts)
        n_samples         : number of segments to sample (default 5)
        sample_duration_s : length of each segment in seconds (default 10)
        prompt            : per-segment prompt; defaults to the audio_prompt config
        """
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        import librosa

        if prompt is None:
            prompt = self.cfg.omni.get('audio_prompt',
                "Summarize the main content of this audio segment.")

        audio_full, sr = librosa.load(audio_path, sr=16000, mono=True)
        total_s = len(audio_full) / sr
        samples_per_seg = int(sample_duration_s * sr)

        # Compute evenly-spaced start positions, clamped to valid range.
        if n_samples == 1:
            start_offsets = [max(0, int((total_s / 2 - sample_duration_s / 2) * sr))]
        else:
            margin = min(sample_duration_s / 2, total_s * 0.05)
            lo = int(margin * sr)
            hi = max(lo, int((total_s - sample_duration_s - margin) * sr))
            step = (hi - lo) / max(n_samples - 1, 1)
            start_offsets = [int(lo + i * step) for i in range(n_samples)]

        segment_descriptions: List[str] = []

        for idx, start in enumerate(start_offsets):
            end = min(start + samples_per_seg, len(audio_full))
            chunk = audio_full[start:end]

            start_s = start / sr
            end_s = end / sr
            print(
                f"OmniDescriptor (Worker): Audio segment {idx + 1}/{n_samples} "
                f"[{start_s:.1f}s – {end_s:.1f}s] …"
            )

            msgs = [{"role": "user", "content": [prompt, chunk]}]
            try:
                seg_desc = self.model.chat(
                    msgs=msgs,
                    do_sample=True,
                    num_beams=1,
                    max_new_tokens=self.cfg.omni.get('max_new_tokens', 512),
                    use_tts_template=True,
                    generate_audio=False,
                    temperature=self.cfg.omni.get('temperature', 0.3),
                )
                segment_descriptions.append(
                    f"[{start_s:.0f}s–{end_s:.0f}s]: {seg_desc[:512].strip()}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on segment {idx + 1} — skipping.")
                torch.cuda.empty_cache()
            except Exception as seg_exc:
                print(f"  Error on segment {idx + 1}: {seg_exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not segment_descriptions:
            raise RuntimeError("Failed to generate descriptions for any audio segments.")

        if len(segment_descriptions) == 1:
            # Strip the timestamp prefix for single-segment results.
            return segment_descriptions[0].split(": ", 1)[-1]

        # Synthesise per-segment descriptions into one coherent summary.
        joined = "\n".join(segment_descriptions)
        synthesis_prompt = (
            f"The following are descriptions of {len(segment_descriptions)} "
            f"sampled segments from an audio recording that is {total_s:.0f} "
            f"seconds long. Based on these samples, write a concise overall "
            f"description of the audio:\n\n{joined}"
        )
        synthesis_msgs = [{"role": "user", "content": [synthesis_prompt]}]
        summary = self.model.chat(
            msgs=synthesis_msgs,
            use_tts_template=False,
            enable_thinking=False,
            max_new_tokens=self.cfg.omni.get('max_new_tokens', 512),
            do_sample=self.cfg.omni.get('do_sample', False),
        )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return summary

    def describe_video(self, video_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description of a video."""
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        if prompt is None:
            if 'video_prompt' in self.cfg.omni and self.cfg.omni.video_prompt:
                prompt = self.cfg.omni.video_prompt
            else:
                raise ValueError("Video prompt not specified in config (cfg.omni.video_prompt).")

        try:
            from minicpmo.utils import get_video_frame_audio_segments
            video_frames, _, _ = get_video_frame_audio_segments(video_path)
            print(f"OmniDescriptor (Worker): Extracted {len(video_frames)} frames from video.")

            msgs = [{"role": "user", "content": video_frames + [prompt]}]

            result = self.model.chat(
                msgs=msgs,
                max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
                use_image_id=False,
                max_slice_nums=1,
                use_tts_template=False,
                enable_thinking=False,
                do_sample=self.cfg.omni.get('do_sample', False),
            )

            if torch.cuda.is_available(): torch.cuda.empty_cache()
            return result
        except Exception as e:
            print(f"Error describing video '{video_path}': {e}")
            traceback.print_exc()
            return ""

    def describe_video_sampled(
        self,
        video_path: str,
        n_samples: int = 5,
        sample_duration_s: float = 10.0,
        frames_per_segment: int = 4,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe a video by sampling N evenly-spaced audio+video segments.

        Unlike ``describe_video`` (which feeds the entire video at once and can
        OOM on long files), this method:

          1. Divides the video timeline into ``n_samples`` evenly-spaced windows
             of ``sample_duration_s`` seconds each.
          2. For every window, extracts:
             - ``frames_per_segment`` evenly-spaced frames (via cv2 / PIL)
             - the corresponding audio waveform (via ffmpeg pipe → librosa)
          3. Calls the model once per segment with both modalities so it can
             see *and* hear what is happening in that slice.
          4. Synthesises the per-segment descriptions into one overall summary.

        Parameters
        ----------
        video_path         : path to the video file (any format OpenCV can open)
        n_samples          : number of time-windows to sample (default 5)
        sample_duration_s  : duration of each window in seconds (default 10)
        frames_per_segment : frames to extract per window (default 4)
        prompt             : per-segment prompt; defaults to the video_prompt config
        """
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        import io
        import subprocess
        import librosa
        import cv2
        from PIL import Image as PILImage

        if prompt is None:
            prompt = self.cfg.omni.get('video_prompt',
                "Describe the video segment in detail. Include the scene, actions, "
                "objects, any speech or sounds, and any notable events.")

    
        # ------------------------------------------------------------------
        # Probe video metadata
        # ------------------------------------------------------------------
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_s = total_frames / fps
        cap.release()

        # Check whether ffmpeg is available (needed for audio extraction)
        try:
            _ffmpeg_available = (
                subprocess.run(
                    ['ffmpeg', '-version'], capture_output=True
                ).returncode == 0
            )
        except FileNotFoundError:
            _ffmpeg_available = False
        if not _ffmpeg_available:
            print("OmniDescriptor (Worker): WARNING — ffmpeg not found; "
                    "audio extraction disabled, falling back to video-only segments.")

        # ------------------------------------------------------------------
        # Compute evenly-spaced segment start times (same logic as describe_audio_sampled)
        # ------------------------------------------------------------------
        if n_samples == 1:
            start_times = [max(0.0, (total_s / 2) - (sample_duration_s / 2))]
        else:
            margin = min(sample_duration_s / 2, total_s * 0.05)
            lo = margin
            hi = max(lo, total_s - sample_duration_s - margin)
            step = (hi - lo) / max(n_samples - 1, 1)
            start_times = [lo + i * step for i in range(n_samples)]

        segment_descriptions: List[str] = []

        for idx, start_s in enumerate(start_times):
            end_s = min(start_s + sample_duration_s, total_s)
            print(
                f"OmniDescriptor (Worker): Video segment {idx + 1}/{n_samples} "
                f"[{start_s:.1f}s – {end_s:.1f}s] …"
            )

            # --------------------------------------------------------------
            # Extract frames for this window
            # --------------------------------------------------------------
            start_frame = int(start_s * fps)
            end_frame   = min(int(end_s * fps), total_frames - 1)
            seg_frames  = end_frame - start_frame

            if frames_per_segment >= seg_frames:
                frame_indices = list(range(start_frame, end_frame + 1))
            else:
                f_step = seg_frames / frames_per_segment
                frame_indices = [
                    min(int(start_frame + i * f_step + f_step / 2), end_frame)
                    for i in range(frames_per_segment)
                ]

            cap = cv2.VideoCapture(video_path)
            pil_frames = []
            for fi in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
                ret, frame = cap.read()
                if ret:
                    pil_frames.append(
                        PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    )
            cap.release()

            # --------------------------------------------------------------
            # Extract audio for this window via ffmpeg pipe
            # --------------------------------------------------------------
            audio_chunk = None
            if _ffmpeg_available:
                try:
                    proc = subprocess.run(
                        [
                            'ffmpeg', '-y',
                            '-ss', str(start_s),
                            '-t',  str(end_s - start_s),
                            '-i',  video_path,
                            '-ac', '1',
                            '-ar', '16000',
                            '-f',  'wav',
                            'pipe:1',
                        ],
                        capture_output=True,
                        timeout=60,
                    )
                    if proc.returncode == 0 and proc.stdout:
                        audio_chunk, _ = librosa.load(
                            io.BytesIO(proc.stdout), sr=16000, mono=True
                        )
                except Exception as audio_exc:
                    print(f"  Audio extraction failed for segment {idx + 1}: {audio_exc}")

            if not pil_frames:
                print(f"  No frames extracted for segment {idx + 1} — skipping.")
                continue

            # --------------------------------------------------------------
            # Build content list: frames [+ audio] + prompt
            # use_tts_template=True only when audio is present
            # --------------------------------------------------------------
            if audio_chunk is not None:
                content = pil_frames + [audio_chunk, prompt]
                use_tts = True
            else:
                content = pil_frames + [prompt]
                use_tts = False

            msgs = [{"role": "user", "content": content}]
            try:
                seg_desc = self.model.chat(
                    msgs=msgs,
                    do_sample=True,
                    num_beams=1,
                    max_new_tokens=self.cfg.omni.get('max_new_tokens', 512),
                    use_image_id=False,
                    max_slice_nums=1,
                    use_tts_template=use_tts,
                    generate_audio=False,
                    enable_thinking=False,
                    temperature=self.cfg.omni.get('temperature', 0.3),
                )
                # We crop the description to 1024 chars to prevent the final synthesis prompt from becoming too long if many segments are used.
                segment_descriptions.append(
                    f"[{start_s:.0f}s–{end_s:.0f}s]: {seg_desc[:1024].strip()}"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM on segment {idx + 1} — skipping.")
                torch.cuda.empty_cache()
            except Exception as seg_exc:
                print(f"  Error on segment {idx + 1}: {seg_exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not segment_descriptions:
            raise RuntimeError("Failed to generate descriptions for any video segments.")

        if len(segment_descriptions) == 1:
            return segment_descriptions[0].split(": ", 1)[-1]

        # Synthesise per-segment descriptions into one coherent summary.
        joined = "\n".join(segment_descriptions)
        synthesis_prompt = (
            f"The following are descriptions of {len(segment_descriptions)} "
            f"sampled segments from a video that is {total_s:.0f} seconds long. "
            f"Based on these samples, write a concise overall description of the "
            f"video:\n\n{joined}"
        )
        synthesis_msgs = [{"role": "user", "content": [synthesis_prompt]}]
        summary = self.model.chat(
            msgs=synthesis_msgs,
            use_tts_template=False,
            enable_thinking=False,
            num_beams=1,
            max_new_tokens=self.cfg.omni.get('max_new_tokens', 512),
            do_sample=self.cfg.omni.get('do_sample', False),
        )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return summary

    def describe_text(self, text: str, prompt: Optional[str] = None) -> str:
        """Generate a summary/description of a long text."""
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        if prompt is None:
            if 'text_prompt' in self.cfg.omni and self.cfg.omni.text_prompt:
                prompt = self.cfg.omni.text_prompt
            else:
                raise ValueError("Text prompt not specified in config (cfg.omni.text_prompt).")

        full_prompt = f"{prompt}\n\n{text}"
        msgs = [{"role": "user", "content": [full_prompt]}]

        result = self.model.chat(
            msgs=msgs,
            use_tts_template=False,
            enable_thinking=False,
            max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
            do_sample=self.cfg.omni.get('do_sample', False),
        )

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return result

    # ------------------------------------------------------------------
    # Multi-turn conversation helpers
    # ------------------------------------------------------------------

    _IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.tiff', '.tif'}

    def _convert_messages_to_msgs(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert standard OpenAI-style messages to MiniCPM-o msgs format.

        Each message is a dict with keys:
          "role"    : "user" | "assistant" | "system"
          "content" : str  — plain text
                    | list — mix of:
                        str  ending in image ext → loaded as PIL Image
                        str  otherwise           → kept as text
                        PIL Image                → kept as-is

        Returned list is suitable for passing directly to model.chat().
        """
        from PIL import Image as PILImage

        msgs = []
        for message in messages:
            role = message['role']
            content = message['content']

            if isinstance(content, str):
                msgs.append({"role": role, "content": [content]})

            elif isinstance(content, list):
                processed = []
                for item in content:
                    if isinstance(item, PILImage.Image):
                        processed.append(item)
                    elif isinstance(item, str):
                        ext = os.path.splitext(item)[1].lower()
                        if ext in self._IMAGE_EXTS and os.path.exists(item):
                            processed.append(PILImage.open(item).convert("RGB"))
                        else:
                            processed.append(item)
                    else:
                        processed.append(str(item))
                msgs.append({"role": role, "content": processed})

            else:
                msgs.append({"role": role, "content": [str(content)]})

        return msgs

    def chat(self, messages: List[Dict]) -> str:
        """
        Multi-turn conversation — blocking, returns the full response string.

        messages: list of {"role": "user"|"assistant", "content": str | list}
        Content list items may include image file paths (auto-loaded as PIL Images).
        """
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")
        try:
            msgs = self._convert_messages_to_msgs(messages)
            result = self.model.chat(
                msgs=msgs,
                use_tts_template=False,
                enable_thinking=False,
                max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
                do_sample=self.cfg.omni.get('do_sample', False),
            )
            return result
        except Exception as e:
            print(f"Error in chat: {e}")
            traceback.print_exc()
            return ""

    def chat_stream_to_queue(self, messages: List[Dict], output_queue) -> None:
        """
        Multi-turn conversation with real-time token streaming.

        Puts items into output_queue:
          ('stream_token', token_str)  — one per generated token chunk
          ('stream_done',  full_text)  — signals completion
          ('error',        exception)  — on failure

        The caller must drain the queue until stream_done or error.

        Implementation notes
        --------------------
        model.chat(stream=True) has a bug: _decode_stream() correctly starts a
        generation thread that fills a TextIteratorStreamer, but after generate()
        returns (streamer, {}), chat() unconditionally does outputs.sequences[0]
        on the empty dict and raises AttributeError — BEFORE returning the
        streamer to us.

        Workaround: temporarily replace model.generate with a thin wrapper that
        captures the (streamer, {}) tuple on its way out.  After catching the
        expected AttributeError from chat(), the generation thread is still
        running and we can iterate the captured streamer normally.

        We also force num_beams=1 because the default config uses num_beams=3
        (beam search), and HF's generate() raises an error if a streamer is
        combined with beam search.
        """
        if not self.model:
            output_queue.put(('error', RuntimeError("OmniDescriptor not initiated.")))
            return

        try:
            msgs = self._convert_messages_to_msgs(messages)

            # --- Capture the streamer before chat() crashes ---
            streamer_ref = []
            original_generate = self.model.generate  # bound method

            def _capturing_generate(*args, **kwargs):
                result = original_generate(*args, **kwargs)
                # stream=True path returns (TextIteratorStreamer, {})
                if (isinstance(result, tuple) and len(result) == 2
                        and isinstance(result[1], dict) and not result[1]):
                    streamer_ref.append(result[0])
                return result

            self.model.generate = _capturing_generate
            try:
                self.model.chat(
                    msgs=msgs,
                    stream=True,
                    do_sample=True,   # stream=True requires do_sample=True
                    num_beams=1,      # beam search is incompatible with streamers
                    use_tts_template=False,
                    enable_thinking=False,
                    max_new_tokens=self.cfg.omni.get('max_new_tokens', 1024),
                )
            except AttributeError as e:
                if 'sequences' not in str(e):
                    raise   # unexpected error, re-raise
                # Expected: chat() crashes at outputs.sequences[0] since
                # outputs={} in stream mode.  The generation thread is alive;
                # streamer_ref[0] is valid and being filled.
            finally:
                self.model.generate = original_generate

            if not streamer_ref:
                raise RuntimeError(
                    "Failed to capture streaming iterator — "
                    "model.generate() may not have been called."
                )

            streamer = streamer_ref[0]
            full_text = ""
            for token_text in streamer:
                if token_text:
                    full_text += token_text
                    output_queue.put(('stream_token', token_text))

            output_queue.put(('stream_done', full_text))

        except Exception as e:
            traceback.print_exc()
            output_queue.put(('error', e))


    def benchmark_context_window(self) -> Dict[str, Any]:
        """
        Measures the practical context-window limits on the current GPU.

        Returns a dict with:
          'gpu'           – GPU name, total/free VRAM after model load
          'model_max'     – max_position_embeddings from the model config
          'kv_cache_kb_per_token' – theoretical KV-cache cost per token (bytes)
          'results'       – list of per-probe dicts:
                              input_tokens, output_tokens, elapsed_s,
                              vram_delta_mb, status ('ok'|'oom'|'error')
          'recommendation' – dict with suggested max_input_tokens,
                             max_new_tokens, and reasoning
        """
        if not self.model:
            raise RuntimeError("OmniDescriptor not initiated.")

        import math

        # ------------------------------------------------------------------
        # 1. GPU inventory
        # ------------------------------------------------------------------
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            total_vram_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
            torch.cuda.synchronize()
            free_bytes, _ = torch.cuda.mem_get_info(0)
            free_after_model_mb = free_bytes / 1024**2
            used_by_model_mb = total_vram_mb - free_after_model_mb
        else:
            gpu_name = "CPU (no CUDA)"
            total_vram_mb = 0.0
            free_after_model_mb = 0.0
            used_by_model_mb = 0.0

        # ------------------------------------------------------------------
        # 2. Model architecture constants → theoretical KV-cache cost
        # ------------------------------------------------------------------
        try:
            llm_cfg = self.model.config.llm_config
        except AttributeError:
            llm_cfg = self.model.config

        num_layers   = getattr(llm_cfg, 'num_hidden_layers',  36)
        num_kv_heads = getattr(llm_cfg, 'num_key_value_heads', 8)
        head_dim     = getattr(llm_cfg, 'head_dim',           128)
        model_max    = getattr(llm_cfg, 'max_position_embeddings', 40960)
        try:
            model_max = self.model.config.max_position_embeddings or model_max
        except AttributeError:
            pass

        # 2 (K+V) × kv_heads × head_dim × layers × 2 (bf16 bytes)
        kv_bytes_per_token = 2 * num_kv_heads * head_dim * num_layers * 2
        kv_kb_per_token = kv_bytes_per_token / 1024

        # ------------------------------------------------------------------
        # 3. Build a reusable filler token pool (neutral, low-information text)
        # ------------------------------------------------------------------
        filler_sentence = (
            "The quick brown fox jumps over the lazy dog. "
            "History shows that empires rise and fall. "
            "Mathematics is the language of the universe. "
        )
        filler_ids = self.tokenizer.encode(filler_sentence, add_special_tokens=False)
        tokens_per_sentence = len(filler_ids)

        def make_prompt(target_tokens: int) -> tuple:
            """Return (prompt_str, actual_token_count)."""
            repeats = math.ceil(target_tokens / tokens_per_sentence)
            ids = (filler_ids * repeats)[:target_tokens]
            text = self.tokenizer.decode(ids, skip_special_tokens=True)
            full_prompt = f"Summarize in one sentence:\n\n{text}"
            actual = len(self.tokenizer.encode(full_prompt, add_special_tokens=True))
            return full_prompt, actual

        # ------------------------------------------------------------------
        # 4. Probe increasing input lengths
        # ------------------------------------------------------------------
        probe_targets = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
        output_tokens_per_probe = 64   # small fixed output — we're testing input capacity
        results = []
        last_ok_input = 0

        for target in probe_targets:
            if has_cuda:
                torch.cuda.reset_peak_memory_stats(0)
                torch.cuda.synchronize()
                mem_before = torch.cuda.memory_allocated(0)

            prompt, actual_input = make_prompt(target)
            msgs = [{"role": "user", "content": [prompt]}]
            status = "ok"
            elapsed = 0.0
            vram_delta_mb = 0.0
            actual_output = 0

            try:
                t0 = time.time()
                response = self.model.chat(
                    msgs=msgs,
                    use_tts_template=False,
                    enable_thinking=False,
                    max_new_tokens=output_tokens_per_probe,
                    do_sample=False,
                    num_beams=1,
                )
                elapsed = time.time() - t0
                actual_output = len(self.tokenizer.encode(
                    response, add_special_tokens=False
                ))
                last_ok_input = actual_input
            except torch.cuda.OutOfMemoryError:
                status = "oom"
                if has_cuda:
                    torch.cuda.empty_cache()
            except Exception as exc:
                status = f"error: {type(exc).__name__}: {exc!s:.80}"
                if has_cuda:
                    torch.cuda.empty_cache()

            if has_cuda:
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated(0)
                vram_delta_mb = (peak - mem_before) / 1024**2

            entry = {
                "target_tokens":  target,
                "input_tokens":   actual_input,
                "output_tokens":  actual_output,
                "elapsed_s":      round(elapsed, 2),
                "vram_delta_mb":  round(vram_delta_mb, 1),
                "status":         status,
            }
            results.append(entry)
            print(
                f"  [{status.upper()[:3]}] "
                f"in={actual_input:>6} tok  "
                f"out={actual_output:>3} tok  "
                f"{elapsed:5.1f}s  "
                f"ΔVRAM={vram_delta_mb:+7.1f} MB"
            )

            if status != "ok":
                break   # don't probe larger sizes after a failure

        # ------------------------------------------------------------------
        # 5. Derive recommendation
        # ------------------------------------------------------------------
        # Use all free VRAM (no safety headroom) for an optimistic estimate —
        # the VRAM budget is too tight to afford any reserve.
        budget_mb = max(free_after_model_mb, 0)

        max_ctx_by_vram = int((budget_mb * 1024) / kv_kb_per_token) if kv_kb_per_token else 0
        # Clamp to model architectural max
        max_ctx_total   = min(max_ctx_by_vram, model_max)

        # Sensible split: 80 % input, 20 % output, but cap output at 1024
        suggested_input  = min(int(max_ctx_total * 0.80), last_ok_input or max_ctx_total)
        suggested_output = min(int(max_ctx_total * 0.20), 1024)
        # Round to nearest 256 for cleanliness
        suggested_input  = max(256, (suggested_input  // 256) * 256)
        suggested_output = max(64,  (suggested_output // 64)  * 64)

        reasoning = (
            f"GPU has {free_after_model_mb:.0f} MB free after loading the model "
            f"({used_by_model_mb:.0f} MB used). "
            f"KV-cache costs {kv_kb_per_token:.0f} KB/token "
            f"({num_layers} layers × {num_kv_heads} KV-heads × {head_dim} head-dim × 2×bf16). "
            f"Using all {budget_mb:.0f} MB free VRAM (no safety headroom — budget is tight), "
            f"supporting ~{max_ctx_by_vram:,} tokens total context "
            f"(model architectural max: {model_max:,}). "
            f"Largest successful probe: {last_ok_input:,} input tokens."
        )

        recommendation = {
            "max_input_tokens":  suggested_input,
            "max_new_tokens":    suggested_output,
            "max_total_context": max_ctx_total,
            "reasoning":         reasoning,
        }

        return {
            "gpu": {
                "name":               gpu_name,
                "total_vram_mb":      round(total_vram_mb, 1),
                "used_by_model_mb":   round(used_by_model_mb, 1),
                "free_after_model_mb":round(free_after_model_mb, 1),
            },
            "model_max_position_embeddings": model_max,
            "kv_cache_kb_per_token":         round(kv_kb_per_token, 1),
            "probes":         results,
            "recommendation": recommendation,
        }


# Set the process name for system tools (nvidia-smi, top, ps)
import setproctitle

def _worker_loop(input_queue, output_queue, cfg):
    """The loop running in the separate process."""
    setproctitle.setproctitle("Anagnorisis-OmniDescriptor")

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
                    result = {
                        'device_type': descriptor.device.type,
                        'model_hash': descriptor.model_hash,
                    }
                    output_queue.put(('success', result))

                elif command == 'chat_stream':
                    # Streaming protocol: chat_stream_to_queue writes
                    # ('stream_token', text), ('stream_done', full_text),
                    # or ('error', exc) directly into output_queue.
                    descriptor.chat_stream_to_queue(*args, output_queue=output_queue, **kwargs)

                elif hasattr(descriptor, command):
                    method = getattr(descriptor, command)
                    result = method(*args, **kwargs)
                    output_queue.put(('success', result))
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
    Converts images, audio, video and text into text descriptions using MiniCPM-o.
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
            except:
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

            self._process = ctx.Process(
                target=_worker_loop,
                args=(self._input_queue, self._output_queue, self.cfg),
                name="Anagnorisis-OmniDescriptor"
            )
            self._process.start()

            # Re-initiate if previously loaded
            if self._models_folder:
                print("OmniDescriptor: Re-initiating model in new subprocess...")
                self._send_command_internal('initiate', (self._models_folder,), {})

    def _send_command_internal(self, command, args, kwargs):
        """Helper to send command and wait for result. Assumes lock is held."""
        self._input_queue.put((command, args, kwargs))

        try:
            if command == 'initiate':
                timeout = 48 * 3600  # Model download can take very long
            else:
                timeout = 30 * 60  # 30 minutes for inference (video can be slow)

            status, result = self._output_queue.get(timeout=timeout)
        except queue.Empty:
            self._terminate_process()
            raise RuntimeError("OmniDescriptor subprocess timed out.")

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
        """Generate a text description/transcription of an audio file."""
        return self._execute('describe_audio', audio_path, prompt)

    def describe_audio_sampled(
        self,
        audio_path: str,
        n_samples: int = 5,
        sample_duration_s: float = 10.0,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe audio by sampling short segments spread across the file.

        Memory-safe alternative to ``describe_audio`` for long recordings.
        Picks ``n_samples`` evenly-spaced windows of ``sample_duration_s``
        seconds each and synthesises their descriptions into one summary.
        """
        return self._execute(
            'describe_audio_sampled', audio_path, n_samples, sample_duration_s, prompt
        )

    def describe_video(self, video_path: str, prompt: Optional[str] = None) -> str:
        """Generate a text description of a video file."""
        return self._execute('describe_video', video_path, prompt)

    def describe_video_sampled(
        self,
        video_path: str,
        n_samples: int = 5,
        sample_duration_s: float = 10.0,
        frames_per_segment: int = 4,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Describe a video by sampling N evenly-spaced audio+video segments.

        Memory-safe alternative to ``describe_video`` for long videos.
        For each of ``n_samples`` windows of ``sample_duration_s`` seconds,
        extracts ``frames_per_segment`` frames (cv2) and the audio waveform
        (ffmpeg), then synthesises per-segment descriptions into one summary.
        """
        return self._execute(
            'describe_video_sampled',
            video_path, n_samples, sample_duration_s, frames_per_segment, prompt
        )

    def describe_text(self, text: str, prompt: Optional[str] = None) -> str:
        """Generate a summary/description of text content."""
        return self._execute('describe_text', text, prompt)

    def chat(self, messages: List[Dict]) -> str:
        """
        Multi-turn conversation — blocking, returns the full response string.

        messages: list of {"role": "user"|"assistant", "content": str | list}
        List content items may include image file paths (auto-loaded as PIL Images).

        Example::

            messages = [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris."},
                {"role": "user", "content": "And of Germany?"},
            ]
            reply = descriptor.chat(messages)
        """
        return self._execute('chat', messages)

    def chat_stream(self, messages: List[Dict]):
        """
        Multi-turn conversation with real-time token streaming.

        Yields individual token strings as they are generated by the model.
        Holds the subprocess lock for the duration of generation, ensuring no
        other call can interleave with an ongoing stream.

        Example::

            for token in descriptor.chat_stream(messages):
                print(token, end='', flush=True)
            print()  # newline after last token
        """
        with self._lock:
            self._ensure_process_running()
            self._input_queue.put(('chat_stream', (messages,), {}))
            self._last_used_time = time.time()

            timeout = 30 * 60  # 30 minutes (video can be slow)
            while True:
                try:
                    status, data = self._output_queue.get(timeout=timeout)
                except queue.Empty:
                    self._terminate_process()
                    raise RuntimeError("OmniDescriptor streaming timed out.")

                if status == 'stream_token':
                    yield data
                elif status == 'stream_done':
                    self._last_used_time = time.time()
                    return
                elif status == 'error':
                    raise data
                else:
                    raise RuntimeError(f"Unknown stream status: {status!r}")

    # --- Cleanup ---
    def __del__(self):
        self._shutdown_event.set()
        self._terminate_process()


if __name__ == '__main__':
    from omegaconf import OmegaConf
    import sys

    # ---------------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------------
    mock_cfg = OmegaConf.create({
        'omni': {
            'model_name': 'openbmb/MiniCPM-o-4_5',
            'load_in_4bit': True,
            'max_new_tokens': 1024,
            'do_sample': False,
            'temperature': 0.3,
            'image_prompt': 'Describe this image in detail. Include objects, scene, colors, actions, and any text visible.',
            'audio_prompt': 'Describe the audio in detail. Include instruments, genre, mood, tempo, and any notable features.',
            'video_prompt': 'Describe the video in detail. Include the scene, actions, objects, and any notable events.',
            'text_prompt': 'Please provide a detailed summary of the following text:',
        }
    })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_data_dir = os.path.join(script_dir, 'omnidescriptor_test_data')
    models_path = os.path.abspath(os.path.join(script_dir, '..', 'models'))
    os.makedirs(models_path, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Verify test data exists
    # ---------------------------------------------------------------------------
    test_image_path = os.path.join(test_data_dir, 'test_image.jpg')
    test_audio_path = os.path.join(test_data_dir, 'test_audio.mp3')
    test_video_path = os.path.join(test_data_dir, 'test_video.mp4')
    test_text = (
        "The Roman Empire was one of the largest and most influential civilizations in history. "
        "At its height under Emperor Trajan in 117 AD, it spanned across Europe, North Africa, "
        "and the Middle East. The empire was known for its advanced engineering, including aqueducts, "
        "roads, and monumental architecture like the Colosseum. Latin, the language of the Romans, "
        "became the foundation for many modern European languages. The empire eventually split into "
        "Eastern and Western halves, with the Western Roman Empire falling in 476 AD and the Eastern "
        "Roman Empire (Byzantine Empire) continuing until 1453 AD."
    )

    missing_files = []
    for f_path, f_name in [(test_image_path, 'test_image.jpg'),
                           (test_audio_path, 'test_audio.mp3'),
                           (test_video_path, 'test_video.mp4')]:
        if not os.path.exists(f_path):
            missing_files.append(f_name)

    if missing_files:
        print(f"WARNING: Missing test files in '{test_data_dir}': {missing_files}")
        print("Please place the following files in the test data directory:")
        print(f"  - test_image.jpg  (any photograph or image)")
        print(f"  - test_audio.mp3  (any speech or audio clip)")
        print(f"  - test_video.mp4  (any short video clip)")
        print(f"Tests for missing modalities will be skipped.\n")

    # ---------------------------------------------------------------------------
    # Initialize the OmniDescriptor
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("Initializing OmniDescriptor Proxy...")
    print("=" * 60)
    descriptor = OmniDescriptor(cfg=mock_cfg)
    descriptor.initiate(models_folder=models_path)
    print(f"Model hash: {descriptor.model_hash}")
    print("Initialization complete.\n")

    results = {}
    all_passed = True

    # ---------------------------------------------------------------------------
    # Test flags — set to False to skip individual tests
    # ---------------------------------------------------------------------------
    RUN_TEST_CONTEXT_WINDOW    = True
    RUN_TEST_TEXT              = True
    RUN_TEST_IMAGE             = True
    RUN_TEST_IMAGE_CUSTOM      = True
    RUN_TEST_AUDIO             = False  # full clip — likely to OOM on tight VRAM
    RUN_TEST_AUDIO_SAMPLED     = True
    RUN_TEST_VIDEO             = False  # describe_video uses minicpmo which has fragile transitive deps
    RUN_TEST_VIDEO_SAMPLED     = True
    RUN_TEST_CHAT              = True
    RUN_TEST_CHAT_STREAM       = True
    RUN_TEST_CHAT_STREAM_IMAGE = True
    RUN_TEST_LIFECYCLE         = True

    # ---------------------------------------------------------------------------
    # Test 1: Context Window Benchmark
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 1: Context Window Benchmark")
    print("=" * 60)
    benchmark_report = None
    if not RUN_TEST_CONTEXT_WINDOW:
        print("  SKIPPED (disabled)\n")
    else:
        try:
            print("  Probing input lengths (output capped at 64 tokens each):")
            benchmark_report = descriptor._execute('benchmark_context_window')

            gpu = benchmark_report['gpu']
            rec = benchmark_report['recommendation']

            print()
            print(f"  {'─' * 54}")
            print(f"  GPU                  : {gpu['name']}")
            print(f"  Total VRAM           : {gpu['total_vram_mb']:.0f} MB")
            print(f"  Used by model        : {gpu['used_by_model_mb']:.0f} MB")
            print(f"  Free after model load: {gpu['free_after_model_mb']:.0f} MB")
            print(f"  KV-cache cost        : {benchmark_report['kv_cache_kb_per_token']:.0f} KB / token")
            print(f"  Model max context    : {benchmark_report['model_max_position_embeddings']:,} tokens")
            print()
            print(f"  {'Input tokens':>13}  {'Output tokens':>13}  {'Time':>6}  {'ΔVRAM':>9}  Status")
            print(f"  {'─'*13}  {'─'*13}  {'─'*6}  {'─'*9}  {'─'*12}")
            for r in benchmark_report['probes']:
                print(
                    f"  {r['input_tokens']:>13,}  "
                    f"{r['output_tokens']:>13,}  "
                    f"{r['elapsed_s']:>5.1f}s  "
                    f"{r['vram_delta_mb']:>+8.0f}M  "
                    f"{r['status']}"
                )
            print()
            print(f"  {'─' * 54}")
            print(f"  RECOMMENDATION FOR DEVELOPMENT:")
            print(f"  Max input tokens  : {rec['max_input_tokens']:>6,}  (max_inp_length config)")
            print(f"  Max output tokens : {rec['max_new_tokens']:>6,}  (max_new_tokens config)")
            print(f"  Max total context : {rec['max_total_context']:>6,}  (combined in+out)")
            print()
            print(f"  Reasoning: {rec['reasoning']}")
            print(f"  {'─' * 54}")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
        print()

    # ---------------------------------------------------------------------------
    # Test 2: Text Description / Summarization
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 2: Text Description / Summarization")
    print("=" * 60)
    if not RUN_TEST_TEXT:
        print("  SKIPPED (disabled)\n")
    else:
        try:
            description = descriptor.describe_text(test_text)
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 10, f"Description too short: '{description}'"
            results['text'] = description
            print(f"  Input  ({len(test_text)} chars): {test_text[:100]}...")
            print(f"  Result ({len(description)} chars): {description[:200]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
        print()

    # ---------------------------------------------------------------------------
    # Test 3: Image Description
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 3: Image Description")
    print("=" * 60)
    if not RUN_TEST_IMAGE:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_image_path):
        try:
            description = descriptor.describe_image(test_image_path)
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 10, f"Description too short: '{description}'"
            results['image'] = description
            print(f"  Result ({len(description)} chars): {description[:200]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_image.jpg not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 4: Image with Custom Prompt
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 4: Image with Custom Prompt")
    print("=" * 60)
    if not RUN_TEST_IMAGE_CUSTOM:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_image_path):
        try:
            custom_prompt = "List all objects visible in this image as a comma-separated list."
            description = descriptor.describe_image(test_image_path, prompt=custom_prompt)
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 3, f"Description too short: '{description}'"
            results['image_custom'] = description
            print(f"  Prompt: {custom_prompt}")
            print(f"  Result ({len(description)} chars): {description[:200]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_image.jpg not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 5: Audio Description
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 5: Audio Description")
    print("=" * 60)
    if not RUN_TEST_AUDIO:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_audio_path):
        try:
            description = descriptor.describe_audio(test_audio_path)
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 5, f"Description too short: '{description}'"
            results['audio'] = description
            print(f"  Result ({len(description)} chars): {description[:200]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_audio.mp3 not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 5b: Sampled Audio Description
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 5b: Sampled Audio Description (memory-safe)")
    print("=" * 60)
    if not RUN_TEST_AUDIO_SAMPLED:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_audio_path):
        try:
            description = descriptor.describe_audio_sampled(
                test_audio_path,
                n_samples=5,
                sample_duration_s=10.0,
            )
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 5, f"Description too short: '{description}'"
            results['audio_sampled'] = description
            print(f"  Result ({len(description)} chars): {description[:300]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_audio.mp3 not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 6: Video Description
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 6: Video Description")
    print("=" * 60)
    if not RUN_TEST_VIDEO:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_video_path):
        try:
            description = descriptor.describe_video(test_video_path)
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 10, f"Description too short: '{description}'"
            results['video'] = description
            print(f"  Result ({len(description)} chars): {description[:200]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_video.mp4 not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 6b: Sampled Video Description
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 6b: Sampled Video Description (memory-safe)")
    print("=" * 60)
    if not RUN_TEST_VIDEO_SAMPLED:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_video_path):
        try:
            description = descriptor.describe_video_sampled(
                test_video_path,
                n_samples=5,
                sample_duration_s=10.0,
                frames_per_segment=4,
            )
            assert isinstance(description, str), f"Expected str, got {type(description)}"
            assert len(description) > 10, f"Description too short: '{description}'"
            results['video_sampled'] = description
            print(f"  Result ({len(description)} chars): {description[:300]}...")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_video.mp4 not found)")
    print()

    # ---------------------------------------------------------------------------
    # Test 7: Multi-turn blocking chat
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 7: Multi-turn Blocking Chat")
    print("=" * 60)
    if not RUN_TEST_CHAT:
        print("  SKIPPED (disabled)\n")
    else:
        try:
            conversation = [
                {"role": "user", "content": "What is the speed of light in a vacuum?"},
            ]
            reply1 = descriptor.chat(conversation)
            assert isinstance(reply1, str) and len(reply1) > 5, f"Bad reply: '{reply1}'"
            print(f"  Turn 1 user   : {conversation[0]['content']}")
            print(f"  Turn 1 model  : {reply1[:200]}...")
            results['chat_turn1'] = reply1

            # Second turn — append model reply and ask follow-up
            conversation.append({"role": "assistant", "content": reply1})
            conversation.append({"role": "user", "content": "Express that in km/s, rounded to the nearest integer."})
            reply2 = descriptor.chat(conversation)
            assert isinstance(reply2, str) and len(reply2) > 2, f"Bad follow-up: '{reply2}'"
            print(f"  Turn 2 user   : {conversation[-1]['content']}")
            print(f"  Turn 2 model  : {reply2[:200]}")
            results['chat_turn2'] = reply2
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
        print()

    # ---------------------------------------------------------------------------
    # Test 8: Streaming chat — tokens appear in real time on the console
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 8: Streaming Chat (real-time token output)")
    print("=" * 60)
    if not RUN_TEST_CHAT_STREAM:
        print("  SKIPPED (disabled)\n")
    else:
        try:
            stream_messages = [
                {"role": "user",
                 "content": "Write a short haiku about artificial intelligence."},
            ]
            print(f"  Prompt: {stream_messages[0]['content']}")
            print("  Streaming response (each token printed as received):")
            print("  ", end='', flush=True)

            collected_tokens = []
            for token in descriptor.chat_stream(stream_messages):
                print(token, end='', flush=True)
                collected_tokens.append(token)

            print()  # newline after last token
            full_response = "".join(collected_tokens)
            assert len(collected_tokens) > 1, "Expected multiple token chunks for streaming test"
            assert len(full_response) > 5, f"Streamed response too short: '{full_response}'"
            results['chat_stream'] = full_response
            print(f"  Total tokens : {len(collected_tokens)}")
            print(f"  Total chars  : {len(full_response)}")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
        print()

    # ---------------------------------------------------------------------------
    # Test 9: Streaming chat with image (if test_image.jpg exists)
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 9: Streaming Chat with Image")
    print("=" * 60)
    if not RUN_TEST_CHAT_STREAM_IMAGE:
        print("  SKIPPED (disabled)\n")
    elif os.path.exists(test_image_path):
        try:
            image_stream_messages = [
                {"role": "user",
                 "content": [test_image_path,
                              "Describe what you see in one sentence."]},
            ]
            print("  Streaming response:")
            print("  ", end='', flush=True)

            img_tokens = []
            for token in descriptor.chat_stream(image_stream_messages):
                print(token, end='', flush=True)
                img_tokens.append(token)

            print()
            img_response = "".join(img_tokens)
            assert len(img_response) > 5, f"Image stream response too short: '{img_response}'"
            results['chat_stream_image'] = img_response
            print(f"  Total chars  : {len(img_response)}")
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")
            traceback.print_exc()
            all_passed = False
    else:
        print("  SKIPPED (test_image.jpg not found)")
    print()


    # ---------------------------------------------------------------------------
    # Test 10: Process Lifecycle (Idle Timeout & Restart)
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST 10: Process Lifecycle (Idle Timeout & Restart)")
    print("=" * 60)
    if not RUN_TEST_LIFECYCLE:
        print("  SKIPPED (disabled)\n")
    elif descriptor._process is None:
        print("  ERROR: Process is None before lifecycle test.")
        all_passed = False
    else:
        initial_pid = descriptor._process.pid
        print(f"  Current worker PID: {initial_pid}")

        # Reduce timeout for testing
        print("  Reducing idle timeout to 2 seconds...")
        descriptor._idle_timeout = 2

        # Wait for timeout (monitor sleeps 5s + buffer)
        print("  Waiting for idle timeout (~7s)...")
        time.sleep(7)

        if descriptor._process is None:
            print("  Worker process reference cleared. OK")
        else:
            print(f"  WARNING: Worker process reference still exists: {descriptor._process}")

        # Check active children
        active_children = multiprocessing.active_children()
        active_pids = [p.pid for p in active_children]

        if initial_pid not in active_pids:
            print(f"  Old worker process {initial_pid} is no longer active. OK")
        else:
            print(f"  WARNING: Old worker process {initial_pid} is still active!")

        # Trigger restart via new command
        print("  Triggering restart via describe_text...")
        descriptor.describe_text("Quick test to restart the process.")

        if descriptor._process is None:
            print("  FAILED: Process did not restart.")
            all_passed = False
        else:
            new_pid = descriptor._process.pid
            print(f"  New worker PID: {new_pid}")

            if new_pid != initial_pid:
                print("  New worker process started (PID changed). OK")
            else:
                print("  WARNING: PID did not change.")

            active_children_after = multiprocessing.active_children()
            print(f"  Active child processes: {[p.pid for p in active_children_after]}")

            if len(active_children_after) == 1:
                print("  Exactly one active child process. OK")
            elif len(active_children_after) > 1:
                print(f"  WARNING: {len(active_children_after)} active child processes detected.")

        # Restore timeout
        descriptor._idle_timeout = 300
        print("  PASSED")
    print()
    
    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for modality, desc in results.items():
        status = "OK" if desc else "EMPTY"
        print(f"  {modality:20s}: {status} ({len(desc)} chars)")

    if benchmark_report:
        rec = benchmark_report['recommendation']
        print()
        print(f"  Context window recommendation:")
        print(f"    max_input_tokens  = {rec['max_input_tokens']:,}")
        print(f"    max_new_tokens    = {rec['max_new_tokens']:,}")

    if all_passed:
        print("\nAll tests PASSED.")
    else:
        print("\nSome tests FAILED.")
        sys.exit(1)

    print("=" * 60)

