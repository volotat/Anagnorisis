"""
DRAFT — NOT FINALIZED, API SUBJECT TO CHANGE

Share API — Read-only REST endpoints that expose an Anagnorisis instance's
file structure, metadata, and raw file content over HTTP.

Any Anagnorisis instance can be visited in a browser to browse files.
Another Anagnorisis instance can connect to it as a "peer" to pull metadata
and run local recommendation/search on top of that metadata.

These endpoints are lightweight — no ML, no GPU, no embeddings. They only
read the filesystem and return JSON or stream files.

Endpoints:
    GET  /api/v1/share/info                           → instance metadata
    GET  /api/v1/share/<module>/folders?path=          → folder tree
    GET  /api/v1/share/<module>/files?path=&recursive= → file list
    GET  /api/v1/share/<module>/metadata?file_path=    → single file metadata + .meta
    POST /api/v1/share/<module>/metadata_batch         → batch metadata
    GET  /api/v1/share/<module>/file?file_path=        → stream raw file bytes
    GET  /api/v1/share/<module>/thumbnail?file_path=   → thumbnail/preview image
"""

import os
import hashlib
from pathlib import Path
from flask import Blueprint, jsonify, request, send_file, abort

from pages.file_manager import get_folder_structure


# ---------------------------------------------------------------------------
#  Blueprint
# ---------------------------------------------------------------------------
share_api = Blueprint('share_api', __name__)

# Module registry — populated by init_share_api()
_modules: dict = {}
_instance_name: str = ""

# Limits
_MAX_META_LINES = 300
_MAX_META_CHARS = 30_000
_MAX_BATCH_SIZE = 500


# ---------------------------------------------------------------------------
#  Initialisation (called once from app.py)
# ---------------------------------------------------------------------------
def init_share_api(app, cfg):
    """Register the share API blueprint and bind module configuration."""
    global _modules, _instance_name

    _instance_name = cfg.main.get(
        'instance_name',
        f"{cfg.main.host}:{cfg.main.port}",
    )

    for module_name in ('music', 'images', 'text', 'videos'):
        module_cfg = getattr(cfg, module_name, None)
        if module_cfg is None:
            continue
        media_dir = module_cfg.get('media_directory')
        if not media_dir or not os.path.isdir(media_dir):
            continue
        _modules[module_name] = {
            'media_directory': media_dir,
            'media_formats': set(module_cfg.get('media_formats', [])),
        }

    app.register_blueprint(share_api, url_prefix='/api/v1/share')


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------
def _get_module_or_404(module_name: str) -> dict:
    """Return module config dict or abort with 404."""
    if module_name not in _modules:
        abort(404, description=f"Module '{module_name}' not available")
    return _modules[module_name]


def _safe_resolve(base_dir: str, user_path: str | None) -> Path:
    """
    Safely resolve *user_path* inside *base_dir*.
    Aborts with 400 on traversal attempts, 404 if result doesn't exist.
    """
    base = Path(base_dir).resolve()
    if not user_path:
        return base
    candidate = (base / user_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError:
        abort(400, description="Invalid path")
    return candidate


def _read_meta_file(meta_path: str) -> tuple[str | None, bool]:
    """
    Read the first N lines / M chars of a .meta sidecar file.
    Returns (content_string | None, was_truncated).
    """
    if not os.path.isfile(meta_path):
        return None, False
    lines: list[str] = []
    total_chars = 0
    truncated = False
    try:
        with open(meta_path, 'r', encoding='utf-8', errors='replace') as fh:
            for i, line in enumerate(fh):
                if i >= _MAX_META_LINES or total_chars + len(line) > _MAX_META_CHARS:
                    truncated = True
                    break
                lines.append(line)
                total_chars += len(line)
    except Exception:
        return None, False
    return ''.join(lines), truncated


def _file_stat_dict(resolved: Path, media_dir: str) -> dict:
    """Return a compact stat dict for a resolved file path."""
    st = resolved.stat()
    return {
        'file_path': os.path.relpath(str(resolved), media_dir),
        'file_name': resolved.name,
        'size':      st.st_size,
        'mtime':     st.st_mtime,
        'extension': resolved.suffix.lower(),
    }


def _enrich_with_meta(entry: dict, resolved: Path) -> dict:
    """
    Add .meta sidecar content (if any) to an existing entry dict.
    Convention: the sidecar is ``<full_filename>.meta``
    e.g. ``song.mp3.meta``, ``photo.jpg.meta``.
    """
    meta_path = str(resolved) + '.meta'
    content, truncated = _read_meta_file(meta_path)
    entry['meta_content']   = content
    entry['meta_truncated'] = truncated if content is not None else None
    return entry


# ---------------------------------------------------------------------------
#  Endpoints
# ---------------------------------------------------------------------------

@share_api.route('/info', methods=['GET'])
def share_info():
    """Instance-level metadata for peer discovery / handshake."""
    return jsonify({
        'name':    _instance_name,
        'modules': list(_modules.keys()),
        'api_version': '1',
    })


@share_api.route('/<module_name>/folders', methods=['GET'])
def share_folders(module_name):
    """
    Return the folder-tree JSON for *module_name*, rooted at the optional
    ``path`` query parameter (relative to the module's media directory).

    The response format matches the existing ``get_folder_structure()``
    output that ``FolderViewComponent`` already consumes.
    """
    mod = _get_module_or_404(module_name)
    rel_path = request.args.get('path', '')

    resolved = _safe_resolve(mod['media_directory'], rel_path)
    if not resolved.is_dir():
        abort(404, description="Directory not found")

    tree = get_folder_structure(str(resolved), mod['media_formats'])
    if tree is None:
        abort(404, description="Directory not found")

    return jsonify(tree)


@share_api.route('/<module_name>/files', methods=['GET'])
def share_files(module_name):
    """
    List media files under ``path`` (relative to media dir).
    ``recursive`` (default true) controls depth.

    Returns lightweight entries: path, name, size, mtime, extension.
    No metadata or .meta content — use ``/metadata`` or ``/metadata_batch``
    for that (on demand).
    """
    mod = _get_module_or_404(module_name)
    rel_path  = request.args.get('path', '')
    recursive = request.args.get('recursive', 'true').lower() == 'true'

    resolved = _safe_resolve(mod['media_directory'], rel_path)
    if not resolved.is_dir():
        abort(404, description="Directory not found")

    media_exts = mod['media_formats']
    files: list[dict] = []

    if recursive:
        for root, _dirs, filenames in os.walk(str(resolved)):
            for fname in filenames:
                if os.path.splitext(fname)[1].lower() in media_exts:
                    full = Path(root) / fname
                    try:
                        files.append(_file_stat_dict(full, mod['media_directory']))
                    except (OSError, ValueError):
                        continue
    else:
        try:
            for entry in os.scandir(str(resolved)):
                if entry.is_file(follow_symlinks=False):
                    if os.path.splitext(entry.name)[1].lower() in media_exts:
                        try:
                            files.append(
                                _file_stat_dict(Path(entry.path), mod['media_directory'])
                            )
                        except (OSError, ValueError):
                            continue
        except OSError:
            abort(404, description="Directory not found")

    return jsonify({
        'module': module_name,
        'path':   rel_path,
        'total':  len(files),
        'files':  files,
    })


@share_api.route('/<module_name>/metadata', methods=['GET'])
def share_metadata(module_name):
    """
    Return metadata for a **single** file, including:
    - basic stat info (size, mtime, extension)
    - ``.meta`` sidecar content (first 300 lines / 30 000 chars)

    ``file_path`` is relative to the module's media directory.
    """
    mod = _get_module_or_404(module_name)
    file_path = request.args.get('file_path', '')
    if not file_path:
        abort(400, description="'file_path' query parameter is required")

    resolved = _safe_resolve(mod['media_directory'], file_path)
    if not resolved.is_file():
        abort(404, description="File not found")

    entry = _file_stat_dict(resolved, mod['media_directory'])
    _enrich_with_meta(entry, resolved)

    return jsonify(entry)


@share_api.route('/<module_name>/metadata_batch', methods=['POST'])
def share_metadata_batch(module_name):
    """
    Batch variant of ``/metadata``.  Accepts a JSON body::

        {"file_paths": ["path/to/a.mp3", "path/to/b.mp3", ...]}

    Returns metadata for each requested file (max 500 per call).
    """
    mod = _get_module_or_404(module_name)

    data = request.get_json(silent=True)
    if not data or 'file_paths' not in data:
        abort(400, description="JSON body with 'file_paths' array required")

    file_paths = data['file_paths']
    if not isinstance(file_paths, list):
        abort(400, description="'file_paths' must be an array")
    if len(file_paths) > _MAX_BATCH_SIZE:
        abort(400, description=f"Maximum {_MAX_BATCH_SIZE} files per batch")

    results: list[dict] = []
    for fp in file_paths:
        try:
            resolved = _safe_resolve(mod['media_directory'], fp)
            if not resolved.is_file():
                results.append({'file_path': fp, 'error': 'not_found'})
                continue
            entry = _file_stat_dict(resolved, mod['media_directory'])
            _enrich_with_meta(entry, resolved)
            results.append(entry)
        except Exception:
            results.append({'file_path': fp, 'error': 'invalid_path'})

    return jsonify({
        'module':  module_name,
        'total':   len(results),
        'results': results,
    })


@share_api.route('/<module_name>/file', methods=['GET'])
def share_file(module_name):
    """
    Stream the raw bytes of a media file.  Only called when a user
    actually opens / plays a file — no pre-fetching.

    ``file_path`` is relative to the module's media directory.
    """
    mod = _get_module_or_404(module_name)
    file_path = request.args.get('file_path', '')
    if not file_path:
        abort(400, description="'file_path' query parameter is required")

    resolved = _safe_resolve(mod['media_directory'], file_path)
    if not resolved.is_file():
        abort(404, description="File not found")

    return send_file(str(resolved))


@share_api.route('/<module_name>/thumbnail', methods=['GET'])
def share_thumbnail(module_name):
    """
    Serve a thumbnail or preview image for a media file.

    - **images**: serves the original file (the browser / frontend handles
      resizing).
    - **videos**: looks for a ``.preview.png`` sidecar next to the video file.
    - **music**: looks for embedded album art (falls back to 404).
    - **text**: not applicable — returns 404.

    ``file_path`` is relative to the module's media directory.
    """
    mod = _get_module_or_404(module_name)
    file_path = request.args.get('file_path', '')
    if not file_path:
        abort(400, description="'file_path' query parameter is required")

    resolved = _safe_resolve(mod['media_directory'], file_path)
    if not resolved.is_file():
        abort(404, description="File not found")

    if module_name == 'images':
        return send_file(str(resolved))

    if module_name == 'videos':
        # Convention used by the videos module
        preview = resolved.with_suffix('.preview.png')
        if preview.is_file():
            return send_file(str(preview))
        abort(404, description="No thumbnail available for this video")

    # music / text — no thumbnail support in the share API for now
    abort(404, description="Thumbnails not supported for this module")


# ---------------------------------------------------------------------------
#  Self-tests  (run with:  python3 -m src.share_api)
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import json
    import shutil
    import tempfile
    from flask import Flask
    from omegaconf import OmegaConf

    passed = 0
    failed = 0

    def ok(label):
        global passed
        passed += 1
        print(f"  ✅  {label}")

    def fail(label, detail=""):
        global failed
        failed += 1
        msg = f"  ❌  {label}"
        if detail:
            msg += f"  — {detail}"
        print(msg)

    def check(condition, label, detail=""):
        if condition:
            ok(label)
        else:
            fail(label, detail)

    # ── Create a temporary media tree ─────────────────────────────────────
    tmp = tempfile.mkdtemp(prefix="share_api_test_")
    music_dir  = os.path.join(tmp, "music")
    images_dir = os.path.join(tmp, "images")

    os.makedirs(os.path.join(music_dir, "Jazz"))
    os.makedirs(os.path.join(music_dir, "Rock"))
    os.makedirs(images_dir)

    # Dummy media files
    song1 = os.path.join(music_dir, "Jazz", "coltrane.mp3")
    song2 = os.path.join(music_dir, "Rock", "acdc.mp3")
    img1  = os.path.join(images_dir, "photo.jpg")

    for fp in (song1, song2, img1):
        with open(fp, 'wb') as f:
            f.write(os.urandom(256))  # small dummy content

    # .meta sidecar for one song
    with open(song1 + ".meta", 'w') as f:
        f.write("artist: John Coltrane\nalbum: A Love Supreme\ngenre: Jazz\n")

    # A very long .meta to test truncation
    long_meta_path = song2 + ".meta"
    with open(long_meta_path, 'w') as f:
        for i in range(_MAX_META_LINES + 50):
            f.write(f"line {i}: " + "x" * 80 + "\n")

    # ── Build a minimal Flask app with the Share API ──────────────────────
    cfg = OmegaConf.create({
        'main': {
            'host': '0.0.0.0',
            'port': 5001,
            'instance_name': 'test-instance',
        },
        'music': {
            'media_directory': music_dir,
            'media_formats': ['.mp3', '.flac', '.wav'],
        },
        'images': {
            'media_directory': images_dir,
            'media_formats': ['.jpg', '.jpeg', '.png'],
        },
        'text': {
            'media_directory': '/nonexistent_dir_for_test',
            'media_formats': ['.txt'],
        },
        'videos': {
            'media_directory': '/nonexistent_dir_for_test',
            'media_formats': ['.mp4'],
        },
    })

    app = Flask(__name__)
    init_share_api(app, cfg)
    client = app.test_client()

    try:
        # ==================================================================
        print("\n" + "=" * 60)
        print("Share API — Self-Tests")
        print("=" * 60)

        # ── /info ─────────────────────────────────────────────────────
        print("\n── GET /api/v1/share/info ──")
        r = client.get('/api/v1/share/info')
        check(r.status_code == 200, "/info returns 200")
        data = r.get_json()
        check(data['name'] == 'test-instance', f"/info name = '{data['name']}'")
        check('music' in data['modules'], "music module listed")
        check('images' in data['modules'], "images module listed")
        # text and videos dirs don't exist, so they should be excluded
        check('text' not in data['modules'], "text module excluded (dir missing)")
        check('videos' not in data['modules'], "videos module excluded (dir missing)")

        # ── /folders ──────────────────────────────────────────────────
        print("\n── GET /api/v1/share/music/folders ──")
        r = client.get('/api/v1/share/music/folders')
        check(r.status_code == 200, "/music/folders returns 200")
        tree = r.get_json()
        check('subfolders' in tree, "folder tree has 'subfolders'")
        check('Jazz' in tree['subfolders'], "Jazz subfolder present")
        check('Rock' in tree['subfolders'], "Rock subfolder present")
        check(tree['total_files'] == 2, f"total_files = {tree['total_files']} (expected 2)")

        print("\n── GET /api/v1/share/music/folders?path=Jazz ──")
        r = client.get('/api/v1/share/music/folders?path=Jazz')
        check(r.status_code == 200, "/music/folders?path=Jazz returns 200")
        sub = r.get_json()
        check(sub['num_files'] == 1, f"Jazz folder has {sub['num_files']} file(s)")

        print("\n── GET /api/v1/share/nonexistent/folders ──")
        r = client.get('/api/v1/share/nonexistent/folders')
        check(r.status_code == 404, "unknown module returns 404")

        # ── /files ────────────────────────────────────────────────────
        print("\n── GET /api/v1/share/music/files (recursive) ──")
        r = client.get('/api/v1/share/music/files')
        check(r.status_code == 200, "/music/files returns 200")
        data = r.get_json()
        check(data['total'] == 2, f"total = {data['total']} (expected 2)")
        names = {f['file_name'] for f in data['files']}
        check('coltrane.mp3' in names, "coltrane.mp3 listed")
        check('acdc.mp3' in names, "acdc.mp3 listed")

        print("\n── GET /api/v1/share/music/files?path=Jazz&recursive=false ──")
        r = client.get('/api/v1/share/music/files?path=Jazz&recursive=false')
        data = r.get_json()
        check(data['total'] == 1, f"Jazz non-recursive total = {data['total']}")

        print("\n── GET /api/v1/share/images/files ──")
        r = client.get('/api/v1/share/images/files')
        data = r.get_json()
        check(data['total'] == 1, f"images total = {data['total']}")

        # ── /metadata ────────────────────────────────────────────────
        print("\n── GET /api/v1/share/music/metadata ──")
        r = client.get('/api/v1/share/music/metadata?file_path=Jazz/coltrane.mp3')
        check(r.status_code == 200, "/metadata returns 200")
        meta = r.get_json()
        check(meta['file_name'] == 'coltrane.mp3', f"file_name = '{meta['file_name']}'")
        check(meta['meta_content'] is not None, ".meta content loaded")
        check('John Coltrane' in (meta['meta_content'] or ''), ".meta contains artist")
        check(meta['meta_truncated'] == False, ".meta not truncated (short file)")

        print("\n── GET /api/v1/share/music/metadata (truncated .meta) ──")
        r = client.get('/api/v1/share/music/metadata?file_path=Rock/acdc.mp3')
        meta = r.get_json()
        check(meta['meta_truncated'] == True, f".meta truncated flag = {meta['meta_truncated']}")

        print("\n── GET /api/v1/share/music/metadata (missing file) ──")
        r = client.get('/api/v1/share/music/metadata?file_path=nope.mp3')
        check(r.status_code == 404, "missing file returns 404")

        print("\n── GET /api/v1/share/music/metadata (no param) ──")
        r = client.get('/api/v1/share/music/metadata')
        check(r.status_code == 400, "missing param returns 400")

        # ── /metadata_batch ──────────────────────────────────────────
        print("\n── POST /api/v1/share/music/metadata_batch ──")
        r = client.post(
            '/api/v1/share/music/metadata_batch',
            data=json.dumps({'file_paths': ['Jazz/coltrane.mp3', 'Rock/acdc.mp3', 'nope.mp3']}),
            content_type='application/json',
        )
        check(r.status_code == 200, "/metadata_batch returns 200")
        batch = r.get_json()
        check(batch['total'] == 3, f"batch total = {batch['total']}")
        # First two should succeed, third should have error
        check('error' not in batch['results'][0], "first file OK")
        check('error' not in batch['results'][1], "second file OK")
        check(batch['results'][2].get('error') == 'not_found', "third file not_found")

        print("\n── POST /api/v1/share/music/metadata_batch (bad body) ──")
        r = client.post(
            '/api/v1/share/music/metadata_batch',
            data='not json',
            content_type='application/json',
        )
        check(r.status_code == 400, "bad JSON body returns 400")

        # ── /file ────────────────────────────────────────────────────
        print("\n── GET /api/v1/share/music/file ──")
        r = client.get('/api/v1/share/music/file?file_path=Jazz/coltrane.mp3')
        check(r.status_code == 200, "/file returns 200")
        check(len(r.data) == 256, f"file size = {len(r.data)} bytes (expected 256)")

        print("\n── GET /api/v1/share/music/file (missing) ──")
        r = client.get('/api/v1/share/music/file?file_path=nope.mp3')
        check(r.status_code == 404, "missing file returns 404")

        # ── /thumbnail ───────────────────────────────────────────────
        print("\n── GET /api/v1/share/images/thumbnail ──")
        r = client.get('/api/v1/share/images/thumbnail?file_path=photo.jpg')
        check(r.status_code == 200, "image thumbnail returns 200 (serves original)")

        print("\n── GET /api/v1/share/music/thumbnail ──")
        r = client.get('/api/v1/share/music/thumbnail?file_path=Jazz/coltrane.mp3')
        check(r.status_code == 404, "music thumbnail returns 404 (not supported)")

        # ── Path traversal ───────────────────────────────────────────
        print("\n── Path traversal protection ──")
        r = client.get('/api/v1/share/music/file?file_path=../../etc/passwd')
        check(r.status_code == 400, "traversal via ../../ blocked")

        r = client.get('/api/v1/share/music/folders?path=../../../')
        check(r.status_code == 400, "traversal in folders blocked")

        r = client.get('/api/v1/share/music/metadata?file_path=../../../etc/passwd')
        check(r.status_code == 400, "traversal in metadata blocked")

        # ── Summary ──────────────────────────────────────────────────
        print("\n" + "=" * 60)
        total = passed + failed
        print(f"Results: {passed}/{total} passed, {failed} failed")
        if failed:
            print("SOME TESTS FAILED")
        else:
            print("ALL TESTS PASSED")
        print("=" * 60 + "\n")

    finally:
        # Cleanup temp directory
        shutil.rmtree(tmp, ignore_errors=True)
