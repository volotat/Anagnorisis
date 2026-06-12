import os
from urllib.parse import unquote
from flask import request, abort
from flask_httpauth import HTTPBasicAuth

class SecurityManager:
    """Manages Authentication and Application-wide Security Middleware."""
    
    @classmethod
    def init_app(cls, app):
        # Initialize Flask-HTTPAuth
        auth = HTTPBasicAuth()

        # If these environment variables are not set, os.environ.get() will return None.
        # If they are set to an empty string, they will be empty.
        AUTH_USERNAME = os.environ.get('ANAGNORISIS_USERNAME')
        AUTH_PASSWORD = os.environ.get('ANAGNORISIS_PASSWORD')

        # Authentication is active only if BOTH username and password are set and non-empty.
        AUTH_ACTIVE = bool(AUTH_USERNAME and AUTH_PASSWORD)

        if AUTH_ACTIVE:
            print("HTTP Basic Authentication is ACTIVE.")
        else:
            print("HTTP Basic Authentication is INACTIVE (ANAGNORISIS_USERNAME or ANAGNORISIS_PASSWORD not set or empty).")

        @auth.verify_password
        def verify_password(username, password):
            # Only perform verification if authentication is active
            if AUTH_ACTIVE and username == AUTH_USERNAME and password == AUTH_PASSWORD:
                print(f"Authentication successful for user: {username}")
                return username
            print(f"Authentication failed for user: {username}")
            return None # Important: return None if auth is active but credentials are wrong

        @auth.error_handler
        def unauthorized():
            return "Unauthorized access. Please provide valid credentials.", 401

        if AUTH_ACTIVE:
            # If auth is active, use auth.login_required
            auth_decorator = auth.login_required
        else:
            # If auth is inactive, use a no-op decorator (does nothing)
            def no_auth_decorator(f):
                return f
            auth_decorator = no_auth_decorator

        # Expose the decorator safely to the app for use in routes
        app.auth_decorator = auth_decorator

        @app.before_request
        @auth_decorator # This decorator will either require login or do nothing
        def before_request_auth():
            pass

        # ---- PREVENT PATH TRAVERSAL ----
        def _looks_like_path(s: str) -> bool:
            return ('/' in s) or ('\\' in s)

        def _has_parent_segment(value: str) -> bool:
            # Decode once or twice to defeat %2e%2e / double-encoding tricks
            v = unquote(unquote(value))
            v = v.replace('\\', '/')
            parts = [p for p in v.split('/') if p]  # drop empty parts
            # Only block when a segment is exactly '..'
            return any(p == '..' for p in parts)

        @app.before_request 
        def block_path_traversal():
            # Keep absolute sensitive paths
            dangerous_substrings = ['/etc/', '/proc/']

            # Collect values to check
            check_values = [request.path]
            check_values.extend(request.args.values())
            check_values.extend(request.form.values())

            if request.is_json and request.json:
                def extract_strings(obj):
                    if isinstance(obj, dict):
                        for v in obj.values():
                            yield from extract_strings(v)
                    elif isinstance(obj, list):
                        for item in obj:
                            yield from extract_strings(item)
                    elif isinstance(obj, str):
                        yield obj
                check_values.extend(extract_strings(request.json))

            for value in check_values:
                if not isinstance(value, str):
                    continue
                # Fast reject for truly dangerous absolute paths
                v_dec = unquote(unquote(value)).lower()
                if any(s in v_dec for s in dangerous_substrings):
                    print(f"Path traversal attempt blocked: {value}")
                    abort(403)
                # Only apply '..' check to path-like strings; allow filenames like '....flac'
                if _looks_like_path(value) and _has_parent_segment(value):
                    print(f"Parent directory traversal blocked: {value}")
                    abort(403)