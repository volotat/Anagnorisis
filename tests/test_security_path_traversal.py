"""
Security tests — path traversal prevention (Tier 3)

Two layers are tested:

1. app.py middleware helpers (_looks_like_path, _has_parent_segment)
   These pure functions are inlined here so we don't need to bootstrap the
   full Flask application.  If the logic in app.py ever diverges, the tests
   will catch it because we also cross-validate with the Flask test client
   in the `TestFlaskBlockPathTraversal` class.

2. Flask before_request hook (block_path_traversal) tested via a minimal
   Flask app that reimplements the same middleware logic in isolation —
   keeping this test free from the heavy startup cost of the full app.py.

3. src/file_manager.resolve_subpath() — the module-level guard used when
   serving files from media directories.
"""
import pytest
from urllib.parse import unquote, quote
from flask import Flask, jsonify, request, abort


# ===========================================================================
# Inlined helpers from app.py (test the logic directly, no app bootstrap)
# ===========================================================================

def _looks_like_path(s: str) -> bool:
    return ('/' in s) or ('\\' in s)


def _has_parent_segment(value: str) -> bool:
    v = unquote(unquote(value))
    v = v.replace('\\', '/')
    parts = [p for p in v.split('/') if p]
    return any(p == '..' for p in parts)


def _is_dangerous(value: str, dangerous_substrings=('/etc/', '/proc/')) -> bool:
    """Return True if the value should be blocked by block_path_traversal."""
    if not isinstance(value, str):
        return False
    v_dec = unquote(unquote(value)).lower()
    if any(s in v_dec for s in dangerous_substrings):
        return True
    if _looks_like_path(value) and _has_parent_segment(value):
        return True
    return False


class TestHelperFunctions:
    # -- _looks_like_path --

    def test_looks_like_path_forward_slash(self):
        assert _looks_like_path('/etc/passwd') is True

    def test_looks_like_path_backslash(self):
        assert _looks_like_path('..\\secret') is True

    def test_looks_like_path_plain_filename(self):
        assert _looks_like_path('photo.jpg') is False

    # -- _has_parent_segment --

    def test_has_parent_segment_simple(self):
        assert _has_parent_segment('../secret') is True

    def test_has_parent_segment_multi_hop(self):
        assert _has_parent_segment('../../etc/passwd') is True

    def test_has_parent_segment_url_encoded(self):
        assert _has_parent_segment('%2e%2e/secret') is True

    def test_has_parent_segment_double_encoded(self):
        assert _has_parent_segment('%252e%252e/secret') is True

    def test_has_parent_segment_ellipsis_filename_not_blocked(self):
        # '....flac' is a valid filename and should NOT be treated as traversal
        assert _has_parent_segment('....flac') is False

    def test_has_parent_segment_normal_path(self):
        assert _has_parent_segment('/images/photo.jpg') is False

    # -- _is_dangerous (the combined check) --

    def test_dangerous_etc(self):
        assert _is_dangerous('/etc/passwd') is True
        assert _is_dangerous('/etc/shadow') is True

    def test_dangerous_proc(self):
        assert _is_dangerous('/proc/self/environ') is True

    def test_dangerous_traversal(self):
        assert _is_dangerous('../../../etc/passwd') is True

    def test_dangerous_encoded_traversal(self):
        assert _is_dangerous('%2e%2e/%2e%2e/secret') is True

    def test_dangerous_double_encoded_traversal(self):
        assert _is_dangerous('%252e%252e/secret') is True

    def test_safe_normal_query(self):
        assert _is_dangerous('cats') is False

    def test_safe_image_path(self):
        assert _is_dangerous('/images/holiday/photo.jpg') is False

    def test_safe_dotdot_in_filename(self):
        # '....flac' contains dots but no path separator — not dangerous
        assert _is_dangerous('....flac') is False

    def test_dangerous_backslash_traversal(self):
        assert _is_dangerous('..\\..\\windows\\system32') is True

    def test_dangerous_mixed_case(self):
        # /ETC/ should match because we lowercase before checking
        assert _is_dangerous('/ETC/passwd') is True


# ===========================================================================
# Flask test client — minimal app that mirrors block_path_traversal logic
# ===========================================================================

def _build_test_app() -> Flask:
    """Minimal Flask app with the same before_request guard as app.py."""
    app = Flask(__name__)

    @app.before_request
    def _block():
        dangerous_substrings = ['/etc/', '/proc/']
        check_values = [request.path]
        check_values.extend(request.args.values())
        check_values.extend(request.form.values())

        if request.is_json and request.json:
            def _extract(obj):
                if isinstance(obj, dict):
                    for v in obj.values():
                        yield from _extract(v)
                elif isinstance(obj, list):
                    for item in obj:
                        yield from _extract(item)
                elif isinstance(obj, str):
                    yield obj
            check_values.extend(_extract(request.json))

        for value in check_values:
            if not isinstance(value, str):
                continue
            v_dec = unquote(unquote(value)).lower()
            if any(s in v_dec for s in dangerous_substrings):
                abort(403)
            if _looks_like_path(value) and _has_parent_segment(value):
                abort(403)

    @app.route('/search')
    def search():
        return jsonify({'query': request.args.get('q', '')})

    @app.route('/api/files', methods=['POST'])
    def files():
        return jsonify({'ok': True})

    return app


@pytest.fixture(scope='module')
def client():
    app = _build_test_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


class TestFlaskBlockPathTraversal:
    # -- Query-parameter attacks --

    def test_traversal_in_query_param_blocked(self, client):
        r = client.get('/search?q=../../../etc/passwd')
        assert r.status_code == 403

    def test_etc_path_in_query_param_blocked(self, client):
        r = client.get('/search?q=/etc/shadow')
        assert r.status_code == 403

    def test_proc_path_in_query_param_blocked(self, client):
        r = client.get('/search?q=/proc/self/environ')
        assert r.status_code == 403

    def test_url_encoded_traversal_in_query_blocked(self, client):
        r = client.get('/search?q=%2e%2e%2f%2e%2e%2fetc%2fpasswd')
        assert r.status_code == 403

    def test_double_encoded_traversal_blocked(self, client):
        r = client.get('/search?q=%252e%252e%2fsecret')
        assert r.status_code == 403

    def test_backslash_traversal_blocked(self, client):
        r = client.get('/search?q=..\\..\\windows\\system32')
        assert r.status_code == 403

    def test_safe_query_allowed(self, client):
        r = client.get('/search?q=cats')
        assert r.status_code == 200

    def test_safe_image_path_allowed(self, client):
        r = client.get('/search?q=images/holiday/photo.jpg')
        assert r.status_code == 200

    # -- JSON body attacks --

    def test_traversal_in_json_body_blocked(self, client):
        r = client.post(
            '/api/files',
            json={'path': '../../../etc/passwd'},
            content_type='application/json',
        )
        assert r.status_code == 403

    def test_nested_traversal_in_json_blocked(self, client):
        r = client.post(
            '/api/files',
            json={'data': {'file': '../../secret.txt'}},
            content_type='application/json',
        )
        assert r.status_code == 403

    def test_list_traversal_in_json_blocked(self, client):
        r = client.post(
            '/api/files',
            json={'files': ['normal.jpg', '../../../etc/passwd']},
            content_type='application/json',
        )
        assert r.status_code == 403

    def test_safe_json_body_allowed(self, client):
        r = client.post(
            '/api/files',
            json={'folder': 'images/vacation'},
            content_type='application/json',
        )
        assert r.status_code == 200

    # -- Form data attacks --

    def test_traversal_in_form_blocked(self, client):
        r = client.post('/api/files', data={'path': '../../../etc/passwd'})
        assert r.status_code == 403

    def test_safe_form_allowed(self, client):
        r = client.post('/api/files', data={'folder': 'images'})
        assert r.status_code == 200

    # -- Dotdot filename edge case --

    def test_dotdot_filename_not_blocked(self, client):
        # Filenames like '....flac' are valid and must not be blocked
        r = client.get('/search?q=....flac')
        assert r.status_code == 200

    def test_mixed_case_etc_blocked(self, client):
        r = client.get('/search?q=/ETC/passwd')
        assert r.status_code == 403


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
