"""file_walker.py — enumerate media files across every configured server.

One process-wide walker owns the traversal, the per-directory cache and the
server availability state. It has no notion of modules or of media types, so
anything that needs a file list can ask it: a module's FileManager for its own
media directory, or the metadata indexer for ``walk("/", every_extension)``.

Splitting this out of FileManager also collapses what used to be one server
monitor thread per module into a single one.
"""

import os
import threading
import time
import traceback
from typing import Callable, Dict, Optional

import fs

import anagnorisis_core.storage.virtual_file_system as vfs
from anagnorisis_core.storage.caching import get_two_level_cache
import logging

logger = logging.getLogger("FileWalker")

# How often the background monitor re-checks every server.
_MONITOR_INTERVAL_SECONDS = 600.0


class FileWalker:
    """Cached, VFS-aware directory traversal over the configured servers."""

    def __init__(self, servers, cache_path: str):
        self.servers = servers

        # Per-directory listing cache, shared with every consumer of this walker.
        self._fast_cache = get_two_level_cache(
            cache_dir=os.path.join(cache_path, "file_manager"), name="file_manager"
        )

        self._availability: Dict[str, Optional[bool]] = {}
        self._availability_lock = threading.Lock()

        threading.Thread(
            target=self._monitor_servers_loop, daemon=True, name="ServerMonitor"
        ).start()

    # ---- public API ---------------------------------------------------

    def walk(self, path: str, media_exts: set[str],
             status_callback: Optional[Callable[[str], None]] = None) -> list[str]:
        """Every file under *path* whose extension is in *media_exts*.

        ``path="/"`` walks all configured servers; an empty *media_exts* accepts
        every extension. *status_callback* receives throttled progress strings.
        """
        return self._walk(path, media_exts, status_callback=status_callback)

    def list_dir(self, path: str, media_exts: set[str], active_fs=None) -> tuple[list[str], list[str]]:
        """Return ``(files_in_dir, subdirs)`` for *path*, cached by directory mtime."""
        return self._list_dir_cached(path, media_exts, active_fs=active_fs)

    def availability(self, url: str) -> Optional[bool]:
        """Last known reachability of a server: True, False, or None if unknown."""
        with self._availability_lock:
            return self._availability.get(url)

    # ---- server availability -----------------------------------------

    def _update_availability(self, url: str, available: bool) -> None:
        with self._availability_lock:
            self._availability[url] = available

    def _resolve_server_from_path(self, path: str):
        for server in self.servers:
            if path.startswith(server.url):
                return server
        return None

    def _check_server_availability(self, url: str) -> bool:
        """Attempts to open the filesystem at url and execute a quick read to verify connection."""
        if url.startswith("osfs://"):
            return True
        try:
            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(url)
            with fs.open_fs(base_url) as f:
                # Force a network round-trip handshake
                list(f.listdir(path_in_fs))
            return True
        except Exception:
            return False

    def _monitor_servers_loop(self):
        """Standard Polling Heartbeat: Periodically checks all servers in the background."""
        while True:
            for server in self.servers:
                self._update_availability(
                    server.url, self._check_server_availability(server.url)
                )
            time.sleep(_MONITOR_INTERVAL_SECONDS)

    # ---- traversal ----------------------------------------------------

    def _list_dir_cached(self, path: str, media_exts: set[str], active_fs=None) -> tuple[list[str], list[str]]:
        """
        Return (files_in_dir, subdirs) for path using TwoLevelCache keyed by dir mtime.
        """

        # 1. Check that the path is within one of the allowed roots
        if not any(path.startswith(server.url) for server in self.servers):
            logger.warning(f"Security check failed: {path} is not within allowed roots.")
            return [], []

        # 2. Resolve base_url and path_in_fs to avoid PyFilesystem's greedy URL parser
        try:
            base_url, path_in_fs = vfs.resolve_base_and_path_from_url(path)
        except Exception as e:
            logger.error(f"Error resolving path \"{path}\": {e}")
            return [], []

        # Internal helper to perform the actual directory scanning on an open FS
        def _scan_fs(media_fs):
            # Let any network exceptions propagate so they can be caught by the outer block
            info = media_fs.getinfo(path_in_fs, namespaces=['details'])
            modified_dt_timestamp = info.modified.timestamp() if info.modified else None

            if modified_dt_timestamp is None:
                # Fallback: Cache directory structure for 10 seconds to make rapid UI sorting instant
                modified_dt_timestamp = int(time.time() / 10) * 10
                logger.debug(f"Could not get modified timestamp for {path}. Using 10s ephemeral cache key.")
            else:
                logger.debug(f"Directory mtime for {path}: {modified_dt_timestamp}")

            ext_sig = ",".join(sorted(media_exts)) if media_exts else "-"
            key = f"MEDIAFILES_OF:{path}|{modified_dt_timestamp}|{ext_sig}"

            # 1. CACHE HIT: Return immediately. Do NOT update server availability
            # because we did not perform any network operations.
            cached = self._fast_cache.get(key)
            if cached is not None:
                return cached

            # 2. CACHE MISS: Perform the actual network directory scan
            files: list[str] = []
            subdirs: list[str] = []
            for e in media_fs.scandir(path_in_fs, namespaces=['details']):
                try:
                    # Hidden entries are not user content: version-control
                    # directories, thumbnail caches, trash folders and
                    # editor droppings. Descending into them would index junk and,
                    # for the annotator, write .meta files into it.
                    if e.name.startswith('.'):
                        continue
                    if e.is_file:
                        ext = os.path.splitext(e.name)[1].lower()
                        if not media_exts or ext in media_exts:
                            files.append(os.path.join(path, e.name))
                    elif e.is_dir:
                        subdirs.append(os.path.join(path, e.name))
                except Exception as inner_e:
                    traceback.print_exc()
                    logger.error(f"Error processing entry \"{e.name}\" in \"{path}\": {inner_e}")
                    continue

            # 3. SUCCESS: The network scan succeeded! We can now safely promote the server to online.
            srv = self._resolve_server_from_path(path)
            if srv:
                self._update_availability(srv.url, True)

            # Cache the result for this directory
            self._fast_cache.set(key, (files, subdirs))
            return files, subdirs

        # 3. Open or Reuse FS Connection
        if active_fs is not None:
            try:
                return _scan_fs(active_fs)
            except Exception as e:
                srv = self._resolve_server_from_path(path)
                if srv:
                    self._update_availability(srv.url, False)
                logger.error(f"Error scanning active filesystem: \"{path}\", Exception: {e}")
                return [], []

        try:
            with fs.open_fs(base_url) as media_fs:
                return _scan_fs(media_fs)
        except Exception as e:
            # FAILURE: Connection or scan failed. Instantly demote the server (Red Dot)
            srv = self._resolve_server_from_path(path)
            if srv:
                self._update_availability(srv.url, False)
            logger.error(f"Error opening filesystem: \"{path}\", Exception: {e}")
            return [], []

    def _walk(self, path: str, media_exts: set[str], progress: dict = None,
              active_fs=None, status_callback=None) -> list[str]:
        if progress is None:
            progress = {'count': 0, 'last_update': time.time()}

        all_files = []

        if path == "/":
            # Scanning multiple servers: let each server establish its own pooled connection
            for server in self.servers:
                try:
                    all_files.extend(self._walk(
                        server.url, media_exts, progress, status_callback=status_callback
                    ))
                except Exception as e:
                    self._update_availability(server.url, False)
                    logger.error(f"Error walking root {server.url}: {e}")
            return all_files

        # Handle a specific single root / subpath
        try:
            base_url, _ = vfs.resolve_base_and_path_from_url(path)
        except Exception as e:
            logger.error(f"Error resolving path \"{path}\": {e}")
            return []

        # If we don't have an active connection, open one at the top of the recursion
        if active_fs is None:
            try:
                with fs.open_fs(base_url) as media_fs:
                    return self._walk(path, media_exts, progress,
                                      active_fs=media_fs, status_callback=status_callback)
            except Exception as e:
                srv = self._resolve_server_from_path(path)
                if srv:
                    self._update_availability(srv.url, False)
                logger.error(f"Error opening filesystem connection for {path}: {e}")
                return []

        # We have an active connection! Perform the cached directory scan
        files, subdirs = self._list_dir_cached(path, media_exts, active_fs=active_fs)
        all_files.extend(files)
        progress['count'] += len(files)

        # Report a clean, shortened display path (throttled to once a second)
        now = time.time()
        if status_callback and now - progress['last_update'] >= 1.0:
            status_callback(
                f"Scanning files ({_display_path(path)}): "
                f"Found {progress['count']} so far..."
            )
            progress['last_update'] = now

        # Recurse into subdirectories reusing the active connection pool
        for subpath in subdirs:
            all_files.extend(self._walk(subpath, media_exts, progress,
                                        active_fs=active_fs, status_callback=status_callback))

        return all_files


def _display_path(path: str) -> str:
    """Shorten a VFS URL for status messages."""
    path_split = path.split('://', 1)
    if len(path_split) < 2:
        return path
    root, tail = path_split[0] + "://", path_split[1]
    return root + " ..." + tail[-17:] if len(tail) > 20 else path


# ---- module-level accessor --------------------------------------------

_WALKERS: dict[tuple, FileWalker] = {}
_WALKER_LOCK = threading.Lock()


def get_file_walker(servers, cache_path: str) -> FileWalker:
    """Return the walker for these roots, building it on first use.

    Memoised per (roots, cache path) rather than being a single global. One
    walker owns a directory-listing cache and a server-monitor thread, so
    building a fresh one per call would leak threads; but keying on a single
    global would hand back a walker whose allowed roots belong to someone else's
    request, and every path outside them would be silently refused.

    Takes the server list itself rather than the Flask app that happens to hold
    it: walking files needs a list of filesystems and somewhere to cache the
    result, and nothing about a web request. The application passes
    ``app.user_cfg.servers``; the command line passes its own list.
    """
    if servers is None:
        raise ValueError("get_file_walker needs a servers list.")
    key = (tuple(sorted(getattr(s, 'url', s['url']) for s in servers)), cache_path)
    walker = _WALKERS.get(key)
    if walker is None:
        with _WALKER_LOCK:
            walker = _WALKERS.get(key)
            if walker is None:
                walker = FileWalker(servers, cache_path)
                _WALKERS[key] = walker
    return walker
