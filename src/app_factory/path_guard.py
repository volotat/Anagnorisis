"""Authorization for a path the client named, outside the HTTP request guard.

`SecurityManager` installs `before_request` middleware that inspects
`request.path`, its arguments, its form and its JSON body. That covers HTTP
requests and nothing else: Socket.IO events arrive as Engine.IO `POST`s to
`/socket.io/` with a `text/plain` body, so the payload never reaches those
checks, and the websocket transport does not pass through Flask at all. A module
server that opens the `file_path` an event carries is therefore opening whatever
the client asked for — any path the process can read, and for the metadata
handlers any path it can write.

The rule here is the same one `/files/<path:filename>` applies, so there is one
answer to "may this path be served" rather than one per entry point: default
deny, with the local media folder plus every configured remote server allowed,
dotfiles refused, and `project_config/modules` limited to sidecar suffixes.
"""
import os

import fs

from anagnorisis_core.storage.virtual_file_system import (
    join_fs_url, resolve_base_and_path_from_url,
)

ALLOWED_MODULE_SUFFIXES = ('.link', '.preview.jpg', '.meta')

# Where a media directory may live when it is a local folder.
_MEDIA_SECTIONS = ('files', 'images', 'music', 'text', 'videos')


class PathNotAuthorized(Exception):
    """The path is not one this application serves to a client."""


def allowed_roots(app) -> list[str]:
    """Every root the web client is permitted to read from.

    Always includes the local media folder; any user-configured remote server
    (`app.user_cfg.servers`) is appended as well. *app* may be None for callers
    that have only a module's config to hand, in which case the local root is
    the whole of the list.
    """
    roots = ['osfs:///mnt/media/']
    user_cfg = getattr(app, 'user_cfg', None)
    if user_cfg is not None and hasattr(user_cfg, 'servers'):
        for srv in (user_cfg.servers or []):
            srv_url = srv.get('url') if hasattr(srv, 'get') else getattr(srv, 'url', None)
            if srv_url:
                roots.append(srv_url.rstrip('/') + '/')
    return roots


def is_authorized_path(app, normalized_url, file_name) -> bool:
    """Default-deny authorization for a resolved VFS URL. Two rules:

      A. Inside the media folder or any configured server root (no dotfiles).
      B. Inside the project_config/modules folder, but only for sidecar files
         (.link / .preview.jpg / .meta).
    """
    # Rule A: media folder or any configured remote server
    if any(normalized_url.startswith(root) for root in allowed_roots(app)):
        # Block hidden dotfiles (like .git, .env) inside media
        return not file_name.startswith('.')

    # Rule B: project_config/modules — only the sidecar suffixes are exposed
    if normalized_url.startswith('osfs:///mnt/project_config/modules/'):
        # .endswith handles double extensions like '.preview.jpg' correctly
        return file_name.endswith(ALLOWED_MODULE_SUFFIXES)

    return False


def authorize_client_url(app, fs_url):
    """Resolve a client-supplied VFS URL and return ``(base_url, path_in_fs)``.

    Raises :class:`PathNotAuthorized` when the URL is not one we serve, and does
    so *before* anything is opened, so an unconfigured authority is never
    contacted. Callers must use the returned pair rather than opening the URL
    themselves.
    """
    if not fs_url or not isinstance(fs_url, str) or '\0' in fs_url:
        raise PathNotAuthorized(f'empty or null-byte path: {fs_url!r}')

    try:
        base_url, path_in_fs = resolve_base_and_path_from_url(fs_url)
    except Exception as exc:
        raise PathNotAuthorized(f'could not resolve {fs_url!r}: {exc}') from exc

    clean_path_in_fs = fs.path.abspath(fs.path.normpath(path_in_fs))
    normalized_url = join_fs_url(base_url, clean_path_in_fs)
    file_name = fs.path.basename(clean_path_in_fs)

    if not is_authorized_path(app, normalized_url, file_name):
        raise PathNotAuthorized(f'not inside an allowed root: {fs_url!r}')

    return base_url, clean_path_in_fs


def local_path_of(fs_url):
    """The filesystem path behind an ``osfs://`` URL, or None for anything else.

    ``osfs:///mnt/media/images/`` → ``/mnt/media/images``. A remote URL has no
    local path and returns None: hand a webdav URL to ``open()`` and you get a
    file called ``webdav:/...`` in the working directory, which is how the .meta
    handlers end up writing somewhere nobody asked for.
    """
    if not fs_url or not isinstance(fs_url, str) or '://' not in fs_url:
        return None
    protocol, remainder = fs_url.split('://', 1)
    if protocol != 'osfs':
        return None
    return os.path.normpath('/' + remainder.lstrip('/'))


def local_media_roots(cfg) -> list[str]:
    """Every local folder that holds media, taken from the configured modules."""
    roots = ['/mnt/media']
    for section in _MEDIA_SECTIONS:
        try:
            section_cfg = cfg.get(section)
        except Exception:
            section_cfg = None
        if not section_cfg:
            continue
        local = local_path_of(getattr(section_cfg, 'media_directory', None))
        if local:
            roots.append(local)
    return roots


def is_authorized_local_path(path, roots) -> bool:
    """True when a raw filesystem path from an event payload is inside *roots*.

    Used by the handlers that write beside a media file with the builtin
    ``open()``, where the payload is a path rather than a VFS URL.
    """
    if not path or not isinstance(path, str) or '\0' in path:
        return False

    candidate = os.path.abspath(os.path.normpath(path))
    if os.path.basename(candidate).startswith('.'):
        return False

    for root in roots:
        root = os.path.abspath(os.path.normpath(str(root)))
        if candidate == root or candidate.startswith(root + os.sep):
            return True
    return False
