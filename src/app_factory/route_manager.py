import os
import markdown
from pathlib import Path
from flask import render_template, send_from_directory, abort, send_file, request
from markdown.extensions.codehilite import CodeHiliteExtension
from pymdownx.arithmatex import ArithmatexExtension
from .extension_manager import ExtensionManager

import mimetypes
import fs
import src.virtual_file_system as vfs
from fs.path import abspath, normpath, basename

class FSFileWrapper:
    """
    Wraps a PyFilesystem2 file object to ensure both the file stream
    and the parent filesystem connection are closed cleanly when Flask
    finishes streaming the response.

    Exposes the stream's size and a fully seekable interface so Flask/Werkzeug
    can honor HTTP Range requests (audio scrubbing/seeking) and emit correct
    206 Partial Content responses.
    """
    def __init__(self, my_fs, file_obj, size=None):
        self.my_fs = my_fs
        self.file_obj = file_obj
        self.size = size  # total file size in bytes (from FS Info), or None if unknown

    def read(self, *args, **kwargs):
        return self.file_obj.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
        # PyFilesystem2 file objects support the standard 2-arg seek(offset, whence),
        # which Werkzeug relies on (e.g. seek(0, 2) to measure length).
        return self.file_obj.seek(*args, **kwargs)

    def tell(self):
        return self.file_obj.tell()

    def seekable(self):
        # Crucial for Flask/Werkzeug to support range requests (seeking/scrubbing in players)
        return True

    def close(self):
        try:
            self.file_obj.close()
        except Exception:
            pass
        try:
            self.my_fs.close()
        except Exception:
            pass

class RouteManager:
    """Manages Static Routing, the Wiki structure, and Secure Mounted File Access."""
    
    @classmethod
    def init_app(cls, app):
        extension_names = ExtensionManager.get_extension_names(app.root_folder)
        
        # Core pages shown first in a fixed order; auto-discovered extras go after a divider
        CORE_PAGE_ORDER = ['files', 'images', 'music', 'text', 'videos', 'train']
        core_pages = [p for p in CORE_PAGE_ORDER if p in extension_names]
        extra_pages = [p for p in extension_names if p not in CORE_PAGE_ORDER]

        markdown_extensions = [
            'tables',
            ArithmatexExtension(generic=True, preview=False, smart_dollar=False),
            'fenced_code', 
            'codehilite',
            'nl2br',
        ]

        @app.template_filter('module_title')
        def module_title_filter(name: str) -> str:
            """Convert a module name like 'web_search' to 'WebSearch'."""
            if '_' not in name:
                return name[0].upper() + name[1:]
            return ''.join(word.capitalize() for word in name.split('_'))

        @app.context_processor
        def inject_nav_pages():
            """Make core_pages and extra_pages available in every template."""
            return dict(core_pages=core_pages, extra_pages=extra_pages)

        # ---- WIKI PAGE FUNCTIONALITY ----
        @app.route('/')
        def index():
            readme_path = os.path.join(app.root_folder, 'README.md')
            with open(readme_path, 'r') as file:
                markdown_text = file.read()
                html = markdown.markdown(markdown_text, extensions=markdown_extensions)
            return render_template('wiki.html', content=html, cfg=app.cfg, pages=extension_names, current_page='wiki')

        @app.route('/wiki/<page_name>')
        def page_wiki(page_name):
            # Assume your Markdown files are in a folder named 'wiki'
            file_path = os.path.join(app.root_folder, 'wiki', f'{page_name}')
            if os.path.exists(file_path):
                with open(file_path, 'r') as file:
                    markdown_text = file.read()
                    html = markdown.markdown(markdown_text, extensions=markdown_extensions)
            else:
                html = '<p>Page not found</p>'
            return render_template('wiki.html', content=html, cfg=app.cfg, pages=extension_names, current_page='wiki')

        # ---- SERVING FILES FROM PAGES FOLDER ----
        @app.route('/modules/<path:filename>')
        @app.auth_decorator
        def custom_static(filename):
            return send_from_directory(os.path.join(app.root_folder, 'modules'), filename)

        # ---------------- SERVING FILES FROM MNT WITH SECURITY CHECKS ----------------

        # We want to allow the web application access to all necessary files mounted by user.
        # IMPORTANT! High security vulnerability, there should be a filter that allows to look only to a specific files, i.e:
        #   * For files from 'mnt/media' folder - probably all the files, but this need to be thought through carefully
        #   * For files from 'mnt/project_config/modules' - probably only .link, .preview.jpg and .meta files.
        #   No other file types or folders should ever be served in the web, to avoid db leakage and stuff like that. 
        #   Also need a path traversal check as well

        # This eventually should be the only point of access to the mounted files from the web application, and all extensions should use this route to access their files, to ensure we have a single point of control and monitoring for security.
        
        # 1. Pre-calculate absolute, resolved base paths.
        # Using .resolve() ensures we have the canonical absolute paths of the directories.
        # This prevents directory-prefix attacks (e.g., matching /mnt/media_backup instead of /mnt/media)
        # BASE_MNT_DIR = Path('/mnt').resolve()
        # MEDIA_DIR = (BASE_MNT_DIR / 'media').resolve()
        # MODULES_DIR = (BASE_MNT_DIR / 'project_config' / 'modules').resolve()

        ALLOWED_MODULE_SUFFIXES = ('.link', '.preview.jpg', '.meta')

        def _collect_allowed_roots():
            """All roots the web client is permitted to read from.

            Always includes the local media folder; any user-configured remote
            server (app.user_cfg.servers) is appended as well.
            """
            allowed_roots = ['osfs:///mnt/media/']

            if hasattr(app, 'user_cfg') and app.user_cfg and hasattr(app.user_cfg, 'servers'):
                user_servers = app.user_cfg.servers or []
                print(f"[ROUTE MANAGER] List of allowed servers: {user_servers}")
                for srv in user_servers:
                    srv_url = srv.get('url') if hasattr(srv, 'get') else getattr(srv, 'url', None)
                    if srv_url:
                        allowed_roots.append(srv_url.rstrip('/') + '/')

            return allowed_roots

        def _is_authorized(normalized_url, file_name):
            """Default-deny authorization. Two rules:
              A. Inside the media folder or any configured server root (no dotfiles).
              B. Inside the project_config/modules folder, but only for sidecar files
                 (.link / .preview.jpg / .meta).
            """
            # Rule A: media folder or any configured remote server
            if any(normalized_url.startswith(root) for root in _collect_allowed_roots()):
                # Block hidden dotfiles (like .git, .env) inside media
                return not file_name.startswith('.')

            # Rule B: project_config/modules — only the sidecar suffixes are exposed
            if normalized_url.startswith('osfs:///mnt/project_config/modules/'):
                # .endswith handles double extensions like '.preview.jpg' correctly
                return file_name.endswith(ALLOWED_MODULE_SUFFIXES)

            return False

        @app.route('/files/<path:filename>')
        def serve_any_file(filename):
            # 1. Security: block null-byte injections immediately
            if '\0' in filename:
                print(f"[SECURITY WARNING] Null byte detected in filename: {filename}")
                abort(400)

            try:
                # 2. Secure path resolution via the VFS helper.
                #    Parses protocol/authority/path cleanly and normalizes the inner path
                #    (normpath + abspath) to defeat traversal/escape attempts.
                base_url, path_in_fs = vfs.resolve_base_and_path_from_url(filename)
                clean_path_in_fs = abspath(normpath(path_in_fs))
                # Reconstruct the canonical URL with the correct number of slashes
                normalized_url = vfs.join_fs_url(base_url, clean_path_in_fs)
            except Exception:
                print(f"[SECURITY WARNING] Path resolution failed for URL: {filename}")
                abort(400)

            # 3. Open the filesystem and verify the resource exists and is a file
            try:
                my_fs = fs.open_fs(base_url)
                info = my_fs.getinfo(clean_path_in_fs, namespaces=['details'])
            except fs.errors.ResourceNotFound:
                print(f"[SECURITY WARNING] Attempt to access non-existent file: {filename}")
                abort(404)
            except Exception as e:
                print(f"[WARNING] Unexpected error while opening filesystem for: {filename}. Error: {e}")
                abort(400)

            if not info.is_file:
                my_fs.close()
                print(f"[SECURITY WARNING] Attempt to access a non-file path: {filename}")
                abort(404)

            # 4. Authorization (default deny)
            file_name = basename(clean_path_in_fs)
            if not _is_authorized(normalized_url, file_name):
                my_fs.close()
                print(f"[SECURITY WARNING] Unauthorized access attempt out of bounds: {filename}")
                abort(403)

            # 5. Serve the file with HTTP Range support (audio seeking/scrubbing).
            #    Werkzeug's send_file only auto-detects size for real paths / BytesIO — for a
            #    stream object it passes complete_length=None to make_conditional, so Range
            #    requests are silently ignored and the browser can't seek. We fix that by
            #    re-running make_conditional with the known size (from the FS Info), which lets
            #    Werkzeug parse the Range header, emit 206 + Content-Range + Accept-Ranges,
            #    and wrap the stream so it seeks to the requested offset.
            try:
                f = my_fs.open(clean_path_in_fs, 'rb')
                wrapper = FSFileWrapper(my_fs, f, size=info.size)
                mimetype, _ = mimetypes.guess_type(file_name)
                last_modified = info.modified if hasattr(info, 'modified') else None

                rv = send_file(
                    wrapper,
                    download_name=file_name,
                    mimetype=mimetype,
                    last_modified=last_modified,
                )
                # size is mandatory for Range processing; re-run conditional handling with it.
                if info.size:
                    rv.make_conditional(request, accept_ranges=True, complete_length=info.size)
                return rv
            except Exception as e:
                my_fs.close()
                print(f"[ERROR] Failed to serve file stream: {e}")
                abort(500)