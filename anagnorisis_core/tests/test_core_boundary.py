"""
The boundary of anagnorisis_core.

The core is imported by three things: the Flask app, the `anagnorisis` command
line, and the data server. Only the first has Flask, a database or a browser. So
a single `from src...` or `import flask` inside the core would work perfectly in
development and then fail at import time for anyone who installed the package
without the application.

`pyproject.toml` already omits those dependencies, which makes it a build-time
boundary. This makes it a test-time one, which fails a great deal sooner.
"""
import ast
import os
import pathlib

import anagnorisis_core
import pytest

# This test sits in the project directory beside the package, which is under src/
# in the usual layout. Resolved from the package itself rather than by counting
# parent directories, so moving the tests cannot silently narrow what is scanned.
CORE = pathlib.Path(anagnorisis_core.__file__).resolve().parent

# Things the core must never reach for. `src` is the Flask application; the rest
# are the web/database layers that only the application is allowed to know about.
FORBIDDEN_ROOTS = {'src', 'flask', 'flask_sqlalchemy', 'flask_socketio',
                   'sqlalchemy', 'flask_limiter', 'werkzeug', 'jinja2'}


def _core_modules():
    """Every module of the engine, excluding its own tests.

    The tests are not shipped and are allowed to know about the application —
    the seam tests in the repository's `tests/` directory do exactly that — so
    scanning them here would be measuring the wrong thing. Same for benchmarks:
    they sit in this directory under the flat layout but are not distributed.
    """
    skip = {'tests', 'benchmarks'}
    return sorted(p for p in CORE.rglob('*.py')
                  if not skip & set(p.relative_to(CORE).parts))


def _imported_roots(path):
    """Top-level package of every import in a file, via AST rather than text.

    Parsing means a mention inside a docstring or a comment does not count —
    only a real import does.
    """
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:      # a relative import stays inside the package
                continue
            if node.module:
                roots.add(node.module.split('.')[0])
    return roots


def test_core_package_exists():
    """Guard the guard: if the path is wrong, every other test here passes."""
    assert CORE.is_dir(), f'{CORE} not found'
    mods = _core_modules()
    assert len(mods) >= 8, f'only found {len(mods)} modules — is the path right?'


@pytest.mark.parametrize('path', _core_modules(), ids=lambda p: p.name)
def test_module_imports_nothing_app_shaped(path):
    offenders = _imported_roots(path) & FORBIDDEN_ROOTS
    assert not offenders, (
        f'{path.relative_to(CORE.parent)} imports {sorted(offenders)}. The core '
        f'must run without the Flask app — an install without it has no Flask '
        f'and no database. Pass what you need in as an argument instead.'
    )


def test_the_check_would_actually_catch_something(tmp_path):
    """A test that cannot fail is not a test."""
    bad = tmp_path / 'bad.py'
    bad.write_text('from src.db_models import db\nimport flask\n')
    assert _imported_roots(bad) & FORBIDDEN_ROOTS == {'src', 'flask'}


def test_taxonomy_ships_with_the_package():
    """A bare install must be able to resolve media types with no repo present."""
    from anagnorisis_core.media.media_types import bundled_media_types_dir
    d = bundled_media_types_dir()
    assert os.path.isfile(os.path.join(d, 'media_types.yaml'))
    tags = os.listdir(os.path.join(d, 'tags'))
    assert len([t for t in tags if t.endswith('.yaml')]) >= 5, tags
