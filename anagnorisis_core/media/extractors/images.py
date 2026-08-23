"""Image internal metadata — format, size and EXIF via PIL."""

from PIL import Image

import anagnorisis_core.storage.virtual_file_system as vfs


def read_image_metadata(file_path: str) -> dict:
    """
    Extracts relevant metadata from an image file and returns a flat dictionary
    with string values suitable for metadata search.

    PIL cannot open a VFS URL, so local paths are unwrapped first. Remote files
    are skipped rather than downloaded — callers that want remote metadata must
    fetch the file themselves.
    """
    metadata = {}
    max_value_length = 1000  # Skip very long values (e.g., base64 data)

    if not vfs.is_local_url(file_path):
        return metadata

    local_path, _ = vfs.resolve_to_local_path(file_path)

    try:
        with Image.open(local_path) as img:
            # Basic image properties
            metadata['format'] = str(img.format) if img.format else 'Unknown'
            metadata['mode'] = str(img.mode) if img.mode else 'Unknown'

            if img.size:
                width, height = img.size
                metadata['width'] = str(width)
                metadata['height'] = str(height)
                metadata['resolution'] = f"{width}x{height}"

            # Extract EXIF data if available
            exif_data = img.getexif()
            if exif_data:
                from PIL.ExifTags import TAGS
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, f'EXIF_{tag_id}')

                    # Convert bytes to string
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8', errors='ignore').strip()
                        except Exception:
                            continue

                    # Convert to string and filter by length
                    value_str = str(value)
                    if len(value_str) <= max_value_length and value_str.strip():
                        metadata[str(tag)] = value_str

            # Extract general info dict if available (PNG, GIF, etc.)
            if hasattr(img, 'info') and img.info:
                for key, value in img.info.items():
                    # Skip binary/bytes data
                    if isinstance(value, bytes):
                        continue

                    # Convert to string and filter
                    if isinstance(value, (str, int, float, bool)):
                        value_str = str(value)
                        if len(value_str) <= max_value_length and value_str.strip():
                            # Prefix with 'INFO_' to distinguish from EXIF
                            metadata[f'INFO_{key}'] = value_str

    except Exception as e:
        # On error, return basic info
        metadata['error'] = f"Failed to extract metadata: {str(e)[:200]}"

    return metadata
