import os
import markdown
from pathlib import Path
from flask import render_template, send_from_directory, abort, send_file
from markdown.extensions.codehilite import CodeHiliteExtension
from pymdownx.arithmatex import ArithmatexExtension
from .extension_manager import ExtensionManager

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
        BASE_MNT_DIR = Path('/mnt').resolve()
        MEDIA_DIR = (BASE_MNT_DIR / 'media').resolve()
        MODULES_DIR = (BASE_MNT_DIR / 'project_config' / 'modules').resolve()

        ALLOWED_MODULE_SUFFIXES = ('.link', '.preview.jpg', '.meta')

        @app.route('/files/<path:filename>')
        def serve_any_file(filename):
            # 2. Security: Block Null-Byte injections immediately
            if '\0' in filename:
                print(f"[SECURITY WARNING] Null byte detected in filename: {filename}")
                abort(400)

            # Flask's <path:filename> strips leading slashes (e.g., /files/mnt/media/x.jpg -> mnt/media/x.jpg).
            # We assume `filename` is always passed as a full path, we can restore the root.
            if not filename.startswith('/'):
                filename = '/' + filename

            try:
                # 3. Secure Path Resolution
                # .resolve(strict=True) does three crucial things simultaneously:
                #   A. Normalizes path traversal ('..', '.')
                #   B. Follows and resolves any symlinks to their *real* target destination
                #   C. Raises FileNotFoundError if the file does not exist (checking existence securely)
                requested_path = Path(filename).resolve(strict=True)
            except FileNotFoundError:
                print(f"[SECURITY WARNING] Attempt to access non-existent file: {filename}")
                abort(404)
            except RuntimeError: # Catches infinite symlink loops
                print(f"[SECURITY WARNING] Infinite symlink loop detected for file: {filename}")
                abort(400)
            except Exception:
                print(f"[WARNING] Unexpected error while accessing file: {filename}")
                abort(400)

            # 4. Ensure the requested path is a file, not a directory
            if not requested_path.is_file():
                print(f"[SECURITY WARNING] Attempt to access a non-file path: {filename}")
                abort(404)

            # 5. Boundary & Constraint Validations (Default: Deny All)
            is_authorized = False

            # Rule A: Files from '/mnt/media' folder
            if requested_path.is_relative_to(MEDIA_DIR):
                is_authorized = True
                
                # Block hidden dotfiles (like .git, .env) in media folder
                if requested_path.name.startswith('.'):
                    is_authorized = False

            # Rule B: Files from '/mnt/project_config/modules' folder
            elif requested_path.is_relative_to(MODULES_DIR):
                # .endswith is better than .suffix because it gracefully handles double extensions like .preview.jpg
                if requested_path.name.endswith(ALLOWED_MODULE_SUFFIXES):
                    is_authorized = True

            # 6. Reject if neither rule matched
            if not is_authorized:
                print(f"[SECURITY WARNING] Unauthorized access attempt out of bounds: {filename}")
                abort(403)

            # 7. Serve the file 
            return send_file(requested_path)