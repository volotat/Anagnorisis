"""Small media plumbing shared by the descriptor and the embedder.

Both of them pull frames out of a video by asking ffmpeg for several images on
one pipe, because starting a process per frame costs more than the decoding
does. Nothing here loads a model or imports torch, so it is safe to import from
anywhere.
"""

from __future__ import annotations

from typing import List

# image2pipe writes PNGs end to end with nothing in between, so the file
# signature is the only boundary marker in the stream.
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


def split_png_stream(blob: bytes) -> List[bytes]:
    """Split concatenated PNGs into the individual files."""
    parts: List[bytes] = []
    start = blob.find(PNG_SIGNATURE)
    while start != -1:
        nxt = blob.find(PNG_SIGNATURE, start + len(PNG_SIGNATURE))
        parts.append(blob[start:nxt] if nxt != -1 else blob[start:])
        start = nxt
    return parts
