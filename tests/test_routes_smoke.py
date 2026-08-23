"""
Route-level smoke tests.

The suite had no test that so much as loaded a page, which made any refactor of
the request path a matter of hope: handler bodies could be rewired and every test
would still pass while the site returned 500s. This is the floor — build the app
for real, then ask every GET route to answer.

It is deliberately shallow. It does not check what a page contains, only that
asking for it does not raise. That is exactly the property that breaks when
handlers are moved.
"""
import os
import pytest

# Opt-in, and not out of caution: create_app() starts the watchdog and the
# schedulers on threads this fixture cannot stop, so pytest finishes the tests
# and then never exits. Run it in its own process:
#
#     ANAGNORISIS_ROUTE_TESTS=1 pytest tests/test_routes_smoke.py
#
# Making the app's background threads daemons (or giving the app a shutdown hook)
# would let this join the default suite — worth doing before Stage 4, since these
# are the tests that make that refactor safe.
pytestmark = [
    pytest.mark.skipif(not os.path.isdir('/app'),
                       reason='needs the container layout (repo at /app)'),
    pytest.mark.skipif(os.environ.get('ANAGNORISIS_ROUTE_TESTS') != '1',
                       reason='builds the real app, which leaves threads running; '
                              'set ANAGNORISIS_ROUTE_TESTS=1 to run in its own process'),
]


@pytest.fixture(scope='module')
def app_and_client(tmp_path_factory):
    """Build the real app once, with nothing running in the background.

    The schedulers are disabled before the app is created: left on, they would
    start walking filesystems and loading models during the test run.
    """
    tmp = tmp_path_factory.mktemp('routes')
    for sub in ('project_config', 'media/images', 'media/music',
                'media/text', 'media/videos'):
        os.makedirs(tmp / sub, exist_ok=True)

    import src.app_factory.config_manager as cm
    original = cm.ConfigManager.setup.__func__

    def quiet_setup(cls, root_folder):
        cfg, user_cfg, paths = original(cls, root_folder)
        # setup() has already resolved and created its paths, so leave them be;
        # what matters is that nothing starts ticking.
        for section in ('images', 'music', 'videos', 'text'):
            if section in cfg:
                cfg[section].rating_update_interval_minutes = None
                cfg[section].embedding_update_interval_minutes = None
                cfg[section].media_directory = f"osfs://{tmp}/media/{section}/"
        if 'metadata_search' in cfg:
            cfg.metadata_search.auto_description_interval_minutes = None
            cfg.metadata_search.embedding_interval_minutes = None
        return cfg, user_cfg, paths

    cm.ConfigManager.setup = classmethod(quiet_setup)
    try:
        from src.app_factory import create_app
        app, socketio, cfg = create_app('/app')
    finally:
        cm.ConfigManager.setup = classmethod(original)

    app.config['TESTING'] = True
    return app, app.test_client()


def _get_routes(app):
    routes = []
    for rule in app.url_map.iter_rules():
        if 'GET' not in rule.methods or rule.arguments:
            continue          # skip parameterised routes; they need real ids
        if rule.rule.startswith('/static'):
            continue
        routes.append(rule.rule)
    return sorted(set(routes))


def test_app_builds_and_has_routes(app_and_client):
    app, _ = app_and_client
    routes = _get_routes(app)
    print(f'\nexercising {len(routes)} routes: {routes}')
    # Guard the guard: if route discovery silently returns almost nothing, the
    # test below passes while checking nothing.
    assert len(routes) >= 8, f'suspiciously few routes discovered: {routes}'


def test_every_plain_get_route_answers(app_and_client):
    """A 500 here is the failure this file exists to catch."""
    app, client = app_and_client
    broken = []
    for route in _get_routes(app):
        try:
            resp = client.get(route, follow_redirects=False)
        except Exception as exc:
            broken.append((route, f'raised {type(exc).__name__}: {exc}'))
            continue
        if resp.status_code >= 500:
            broken.append((route, f'HTTP {resp.status_code}'))
    assert not broken, 'routes failing:\n' + '\n'.join(f'  {r}: {w}' for r, w in broken)
