"""Paths: what is inside a folder, and what is outside the ones we serve.

Filesystem work that has nothing to do with a web request — counting what is in a
folder, opening one in the desktop file manager, and above all deciding whether a
path the user supplied stays inside the directory we agreed to serve.

``resolve_subpath`` is security-critical. It is the guard between a URL and the
filesystem, and it lives here rather than beside the request handlers so that the
CLI and the annotator are protected by the same code the web routes use, rather
than by a second implementation that drifts. Its tests are in
tests/test_file_manager.py and tests/test_security_path_traversal.py.
"""

import os
import subprocess
import sys
from pathlib import Path


def get_folder_structure(folder_path, media_extensions=None):
    # Check if directory exists and return None if not
    if not os.path.isdir(folder_path):
        return None
  
    def count_files(folder):
        return sum(1 for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in media_extensions)

    def build_structure(path):
        folder_dict = {
        'name': os.path.basename(path),
        'num_files': count_files(path),
        'total_files': 0,
        'subfolders': {}
        }
        folder_dict['total_files'] = folder_dict['num_files']
        
        for subfolder in os.listdir(path):
            subfolder_path = os.path.join(path, subfolder)
            if os.path.isdir(subfolder_path):
                subfolder_structure = build_structure(subfolder_path)
                folder_dict['subfolders'][subfolder] = subfolder_structure
                folder_dict['total_files'] += subfolder_structure['total_files']
        
        return folder_dict
    return build_structure(folder_path)


def open_file_in_folder(file_path):
    file_path = os.path.normpath(file_path)
    logger.info(f'Opening file with path: "{file_path}"')
    
    # Assuming file_path is the full path to the file
    folder_path = os.path.dirname(file_path)
    if os.path.isfile(file_path):
      if sys.platform == "win32":  # Windows
        subprocess.run(["explorer", "/select,", file_path], check=True)
      elif sys.platform == "darwin":  # macOS
        subprocess.run(["open", "-R", file_path], check=True)
      else:  # Linux and other Unix-like OS
        # Convert the file path to an absolute path
        abs_path = os.path.abspath(file_path)

         # Check for the file manager and use the appropriate command on Linux
        if os.environ.get('XDG_CURRENT_DESKTOP') in ['GNOME', 'Unity']:
          subprocess.run(['nautilus', '--no-desktop', abs_path])
        elif os.environ.get('XDG_CURRENT_DESKTOP') == 'KDE':
          subprocess.run(['dolphin', '--select', abs_path])
        else:
          logger.warning("Unsupported desktop environment. Please add support for your file manager.")
    else:
      logger.error("File does not exist.")


class PathTraversalError(Exception):
    pass

def resolve_subpath(base_dir: str, user_path: str | None) -> Path:
    """
    Safely resolve user_path inside base_dir. Raises PathTraversalError if escape attempt.
    Empty / None user_path returns base_dir.
    """
    base = Path(base_dir).resolve()
    candidate = base if not user_path else (base / user_path)
    try:
        resolved = candidate.resolve()
        resolved.relative_to(base)  # raises ValueError if outside
    except Exception:
        raise PathTraversalError(f"[FileManager] Invalid path: {user_path}")
    return resolved

# filters = {
#    "by_file": by_file_sort_function,
#    "by_text": by_text_sort_function,
#    "custom": custom_sort_function,
#    ...
# }
