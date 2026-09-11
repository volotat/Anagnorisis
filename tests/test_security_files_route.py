"""
Security tests — the /files route must authorize before it connects (Tier 3).

The route takes a whole VFS URL from the client, including its protocol and
authority. Anything that opens that filesystem before checking it against the
configured roots is an SSRF: `/files/ftp://<host>:<port>/...` makes the server
dial an address the caller chose, and the answer it gets back (400 for "could
not connect", 403 for "connected, outside the roots", 404 for "connected, no
such file") is a port-scanner and file-existence oracle for that address.

These tests register the REAL route through RouteManager.init_app() — the logic
is not copied here — and replace fs.open_fs with a recorder, so "did the server
try to connect" is an assertion rather than an inference.
"""
import os
import sys

import pytest
from flask import Flask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


class _RecordingOpenFs:
    """Stands in for fs.open_fs and remembers every URL it was asked to open."""

    def __init__(self, real_open_fs, boom_host=None):
        self.real = real_open_fs
        self.boom_host = boom_host
        self.opened = []

    def __call__(self, url, *args, **kwargs):
        self.opened.append(url)
        if self.boom_host and self.boom_host in url:
            raise OSError(f'connection refused: {url}')
        return self.real(url, *args, **kwargs)


@pytest.fixture()
def files_client(monkeypatch, tmp_path):
    """A Flask app carrying the app's real /files route and a fake fs.open_fs."""
    from omegaconf import OmegaConf
    import src.app_factory.route_manager as route_manager

    recorder = _RecordingOpenFs(route_manager.fs.open_fs, boom_host='127.0.0.1:9')
    monkeypatch.setattr(route_manager.fs, 'open_fs', recorder)

    app = Flask(__name__)
    app.root_folder = REPO_ROOT
    app.auth_decorator = lambda f: f          # no authentication configured
    app.cfg = OmegaConf.create({})
    app.user_cfg = OmegaConf.create({'servers': []})
    app.paths = {'project_config': str(tmp_path)}

    route_manager.RouteManager.init_app(app)
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client, recorder


class TestFilesRouteAuthorizesBeforeConnecting:

    def test_unconfigured_remote_host_is_never_contacted(self, files_client):
        """An authority that is not a configured server must not be dialled."""
        client, recorder = files_client
        resp = client.get('/files/ftp://127.0.0.1:9/secret')
        assert resp.status_code == 403
        assert recorder.opened == [], (
            f'the route opened {recorder.opened!r} before authorizing it')

    def test_cloud_metadata_address_is_never_contacted(self, files_client):
        client, recorder = files_client
        resp = client.get('/files/ftp://169.254.169.254/latest/meta-data/')
        assert resp.status_code == 403
        assert recorder.opened == []

    def test_missing_file_outside_the_roots_is_forbidden_not_missing(self, files_client):
        """403, not 404: the response must not report whether the file exists."""
        client, recorder = files_client
        resp = client.get('/files/osfs:///root/.ssh/id_rsa')
        assert resp.status_code == 403
        assert recorder.opened == []

    def test_absolute_host_path_is_not_served(self, files_client):
        client, recorder = files_client
        resp = client.get('/files/osfs:///etc/passwd')
        assert resp.status_code == 403
        assert recorder.opened == []

    def test_a_path_inside_the_media_root_still_reaches_the_filesystem(self, files_client):
        """The guard is a filter, not a wall: allowed roots are still served."""
        client, recorder = files_client
        resp = client.get('/files/osfs:///mnt/media/text/notes.txt')
        # /mnt/media does not exist on the test machine, so the lookup 404s —
        # what matters is that the request got past authorization to the
        # filesystem, which is what the recorder proves.
        assert resp.status_code == 404
        assert recorder.opened == ['osfs:///']

    def test_dotfiles_inside_the_media_root_are_refused(self, files_client):
        client, recorder = files_client
        resp = client.get('/files/osfs:///mnt/media/text/.env')
        assert resp.status_code == 403
        assert recorder.opened == []

    def test_module_sidecars_are_allowed_but_other_files_are_not(self, files_client):
        client, recorder = files_client
        allowed = client.get('/files/osfs:///mnt/project_config/modules/notes.txt.meta')
        assert allowed.status_code == 404           # authorized, then not found
        refused = client.get('/files/osfs:///mnt/project_config/modules/notes.txt')
        assert refused.status_code == 403
        assert recorder.opened == ['osfs:///']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
