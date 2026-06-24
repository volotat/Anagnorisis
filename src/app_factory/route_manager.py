import os
import markdown
from pathlib import Path
from flask import render_template, send_from_directory, abort, send_file
from markdown.extensions.codehilite import CodeHiliteExtension
from pymdownx.arithmatex import ArithmatexExtension
from .extension_manager import ExtensionManager

from urllib.parse import unquote
import mimetypes
import fs
import src.virtual_file_system as vfs
from fs.path import abspath, normpath, basename

class FSFileWrapper:
    """
    Wraps a PyFilesystem2 file object to ensure both the file stream
    and the parent filesystem connection are closed cleanly when Flask 
    finishes streaming the response.
    """
    def __init__(self, my_fs, file_obj):
        self.my_fs = my_fs
        self.file_obj = file_obj

    def read(self, *args, **kwargs):
        return self.file_obj.read(*args, **kwargs)

    def seek(self, *args, **kwargs):
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

        @app.route('/files/<path:filename>')
        def serve_any_file(filename):
            # 2. Security: Block Null-Byte injections immediately
            if '\0' in filename:
                print(f"[SECURITY WARNING] Null byte detected in filename: {filename}")
                abort(400)

            # Decode the URL-encoded filename (handles Cyrillic and special characters cleanly)
            decoded_filename = unquote(filename)

            # Standardize protocol slashes in case they got collapsed by Flask's path router
            url = decoded_filename
            if ':/' in url and '://' not in url:
                protocol, rest = url.split(':/', 1)
                if protocol == 'osfs':
                    url = f"osfs:///{rest.lstrip('/')}"
                else:
                    url = f"{protocol}://{rest.lstrip('/')}"
            elif '://' in url:
                protocol, rest = url.split('://', 1)
                if protocol == 'osfs' and not rest.startswith('/'):
                    url = f"osfs:///{rest}"

            try:
                # 3. Secure Path Resolution
                # resolve_base_and_path_from_url parses the URL.
                # We then clean the path inside the FS using normpath/abspath to prevent traversal/escape hacks.
                base_url, path_in_fs = vfs.resolve_base_and_path_from_url(filename)
                
                clean_path_in_fs = abspath(normpath(path_in_fs))
                
                # Do not use rstrip('/') directly on base_url. For 'osfs:///', rstrip('/') removes 
                # ALL trailing slashes, turning it into 'osfs:', which mangles protocol matching.
                if '://' in base_url:
                    protocol, rest = base_url.split('://', 1)
                    clean_rest = rest.rstrip('/')
                    clean_path = clean_path_in_fs.lstrip('/')
                    if clean_rest:
                        normalized_url = f"{protocol}://{clean_rest}/{clean_path}"
                    else:
                        normalized_url = f"{protocol}:///{clean_path}"
                else:
                    normalized_url = f"{base_url.rstrip('/')}/{clean_path_in_fs.lstrip('/')}"
            except Exception:
                print(f"[SECURITY WARNING] Path resolution failed for URL: {filename}")
                abort(400)

            # Open filesystem to verify file details
            try:
                my_fs = fs.open_fs(base_url)
                info = my_fs.getinfo(clean_path_in_fs, namespaces=['details'])
            except fs.errors.ResourceNotFound:
                print(f"[SECURITY WARNING] Attempt to access non-existent file: {filename}")
                abort(404)
            except Exception as e:
                print(f"[WARNING] Unexpected error while opening filesystem for: {filename}. Error: {e}")
                abort(400)

            # 4. Ensure the requested path is a file, not a directory
            if not info.is_file:
                my_fs.close()
                print(f"[SECURITY WARNING] Attempt to access a non-file path: {filename}")
                abort(404)

            # 5. Boundary & Constraint Validations (Default: Deny All)
            is_authorized = False
            file_name = basename(clean_path_in_fs)

            # Build list of allowed roots dynamically
            allowed_roots = ['osfs:///mnt/media/']
            
            # Extract dynamic allowed roots from app.user_cfg.servers
            user_servers = []
            if hasattr(app, 'user_cfg') and app.user_cfg and hasattr(app.user_cfg, 'servers'):
                user_servers = app.user_cfg.servers
                print(f"[ROUTE MANAGER] List of allowed servers: {user_servers}")
                
            for srv in user_servers:
                srv_url = srv.get('url') if hasattr(srv, 'get') else getattr(srv, 'url', None)
                if srv_url:
                    allowed_roots.append(srv_url.rstrip('/') + '/')

            # Rule A: Files from '/mnt/media' folder or any dynamic configured server
            # Standardizing matches by checking startswith against allowed_roots
            if any(normalized_url.startswith(root) for root in allowed_roots):
                is_authorized = True
                
                # Block hidden dotfiles (like .git, .env) in media folder
                if file_name.startswith('.'):
                    is_authorized = False

            # Rule B: Files from '/mnt/project_config/modules' folder
            elif normalized_url.startswith('osfs:///mnt/project_config/modules/'):
                # .endswith is better than .suffix because it gracefully handles double extensions like .preview.jpg
                if file_name.endswith(ALLOWED_MODULE_SUFFIXES):
                    is_authorized = True

            # 6. Reject if neither rule matched
            if not is_authorized:
                my_fs.close()
                print(f"[SECURITY WARNING] Unauthorized access attempt out of bounds: {filename}")
                abort(403)

            # 7. Serve the file 
            try:
                # Open the stream in binary mode (supported by 100% of FS drivers)
                f = my_fs.open(clean_path_in_fs, 'rb')
                wrapper = FSFileWrapper(my_fs, f)
                
                # Guess mimetype based on filename for browser compatibility
                mimetype, _ = mimetypes.guess_type(file_name)
                return send_file(wrapper, download_name=file_name, mimetype=mimetype)
            except Exception as e:
                my_fs.close()
                print(f"[ERROR] Failed to serve file stream: {e}")
                abort(500)