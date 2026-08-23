"""Audio internal metadata — ID3 / Vorbis / MP4 tags via TinyTag."""

import base64
import io

import fs
from PIL import Image
from tinytag import TinyTag

import anagnorisis_core.storage.virtual_file_system as vfs


def _empty_metadata() -> dict:
    """Consistent, safe default values on failure."""
    return {
        'title': "N/A",
        'artist': "N/A",
        'album': "N/A",
        'track_num': "N/A",
        'genre': "N/A",
        'date': "N/A",
        'duration': 0,
        'bitrate': None,
        'lyrics': "",
        'image': None
    }


def read_audio_metadata(file_path: str, full_image: bool = False) -> dict:
    """
    Extracts metadata from an audio file (local or remote) using TinyTag and PyFilesystem2.
    Only streams the metadata sections of the file over the network, keeping execution fast.
    """
    metadata = _empty_metadata()

    # 1. Parse and extract the base connection URL and target path inside the FS
    try:
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
    except Exception as e:
        print(f"[Extractors:audio] Error resolving path {file_path}: {e}")
        return _empty_metadata()

    try:
        # 2. Open the filesystem connection
        with fs.open_fs(base_url) as my_fs:
            # 3. Open the file in binary read mode (supported across all VFS providers)
            with my_fs.open(path_in_fs, 'rb') as f:

                # 4. Pass 'filename' so TinyTag detects the format via extension,
                # and pass 'file_obj' so it reads and seeks directly over the stream.
                tag = TinyTag.get(filename=path_in_fs, file_obj=f, image=True)

                metadata['title'] = tag.title or "N/A"
                metadata['artist'] = tag.artist or "N/A"
                metadata['album'] = tag.album or "N/A"
                metadata['track_num'] = tag.track if tag.track else "N/A"
                metadata['genre'] = tag.genre if tag.genre else "N/A"
                metadata['date'] = str(tag.year) if tag.year else "N/A"

                metadata['duration'] = tag.duration  # seconds
                metadata['bitrate'] = tag.bitrate  # kbps

                metadata['lyrics'] = tag.extra.get('lyrics', "")

                img = tag.get_image()
                if img is not None:
                    try:
                        if not full_image:
                            image = Image.open(io.BytesIO(img))
                            max_size = 512
                            ratio = min(max_size / image.width, max_size / image.height, 1.0)
                            new_size = (int(image.width * ratio), int(image.height * ratio))

                            try:
                                resample = Image.Resampling.LANCZOS  # Pillow >= 10
                            except AttributeError:
                                resample = getattr(Image, 'LANCZOS', Image.BICUBIC)  # Pillow < 10 fallback

                            image = image.resize(new_size, resample=resample)
                            buffered = io.BytesIO()
                            image.save(buffered, format="PNG")
                            img_bytes = buffered.getvalue()
                        else:
                            img_bytes = img

                        base64_image = base64.b64encode(img_bytes).decode('utf-8')
                        metadata['image'] = f"data:image/png;base64,{base64_image}"
                    except Exception as e:
                        print(f"[Extractors:audio] Cover art processing failed for {file_path}: {e}")
                        metadata['image'] = None
                else:
                    metadata['image'] = None

    except Exception as e:
        print(f"[Extractors:audio] Error extracting metadata from {file_path}: {e}")

    return metadata
