"""Loading configuration without a Flask app.

The engine reads a small, fixed set of settings: three paths under ``main``, and
the whole ``embedder`` and ``omni`` sections. Everything else in the
application's config.yaml — modules, ports, secrets — is none of its business.

The package stands alone: ``data/defaults.yaml`` ships those settings, so an
install from a wheel works with no Anagnorisis checkout on the machine and no
config file to point at. Anything passed as *config_paths* is merged on top of
it, key by key, which is how the application's config.yaml keeps winning when
the app loads the engine.

The media-type taxonomy is merged in the same way the application merges it, so
the command line and the app agree about what an image is. The package's own copy
is the default; a directory passed as *media_types_dir* overrides it, which is
how someone's edited tag vocabularies keep applying.
"""
import glob
import os
from typing import Iterable, Optional

from omegaconf import OmegaConf

from anagnorisis_core.media.media_types import bundled_media_types_dir


def bundled_defaults_path() -> str:
    """The default settings shipped with this package.

    At the top of the package rather than tucked into a data directory: it is
    the one file here somebody may want to read or copy before changing
    anything, so it should be the first thing they see.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'defaults.yaml')


def load_config(
    config_paths: Iterable[str] = (),
    *,
    project_config_path: Optional[str] = None,
    models_path: Optional[str] = None,
    overrides: Optional[dict] = None,
    use_user_config: bool = True,
):
    """Assemble a config the engine can use.

    Args:
        config_paths: YAML files, applied in order so later ones win. May be
            empty: the bundled defaults are enough to run.
        project_config_path: the one folder holding cache, memory and the
            trained evaluator. Laid out exactly like the application's
            ``project_config/``, so pointing at one makes the command line and
            the app share everything.
        models_path: where the downloaded model weights live. Deliberately not
            inside the project folder — they are large and identical for
            everyone, so several users can share one copy.
        overrides: last word, for library callers.
        use_user_config: read ``~/.config/anagnorisis/config.yaml`` underneath
            everything else, which is where ``anag config set`` writes. On by
            default so a script sees the same library the command line does —
            somebody who configures a machine and then writes five lines of
            Python should not silently get a different one. Pass False for an
            application that carries its own configuration and must not inherit
            the settings of whoever launched it.

    Those two are the whole surface. Cache, memory and the trained evaluator are
    *derived* from the project folder rather than settable beside it — one path
    to think about instead of four, and no way to end up with a memory folder
    belonging to one library and a cache belonging to another.
    """
    # Always the copy inside the package, found relative to it. The taxonomy is
    # not configuration: the tag vocabularies and the extension lists are what
    # the cache keys and the trained evaluator were built against, so a machine
    # pointing at a different one would be reading its own cache wrong.
    taxonomy_dir = bundled_media_types_dir()

    cfg = OmegaConf.load(bundled_defaults_path())
    for path in sorted(glob.glob(os.path.join(taxonomy_dir, '*.yaml'))):
        cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    for path in resolved_config_paths(config_paths, use_user_config):
        if path and os.path.isfile(path):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(path))

    if 'main' not in cfg:
        cfg.main = OmegaConf.create({})
    if project_config_path:
        cfg.main.project_config_path = project_config_path
    if models_path:
        cfg.main.embedding_models_path = models_path
    cfg.main.media_types_path = taxonomy_dir

    # Derived, not merged: whatever a config file may say about cache_path and
    # friends, they follow the project folder. Letting them be set individually
    # is how you end up with a memory folder belonging to one library and a
    # cache belonging to another, which is silent and hard to notice.
    _require(cfg, 'main.project_config_path')
    project_dir = os.path.expanduser(str(cfg.main.project_config_path))
    cfg.main.project_config_path = project_dir
    for key, subdir in (('cache_path', 'cache'),
                        ('memory_path', 'memory'),
                        ('personal_models_path', 'models')):
        cfg.main[key] = os.path.join(project_dir, subdir)

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(overrides))

    _require(cfg, 'main.cache_path')
    _require(cfg, 'main.embedding_models_path')
    _require(cfg, 'omni.model_name')
    _require(cfg, 'embedder.model_name')

    # `~` in the bundled defaults is literal until something expands it, and a
    # directory named "~" in the working directory is a memorable way to ruin an
    # afternoon.
    for key in ('project_config_path', 'cache_path', 'embedding_models_path',
                'personal_models_path', 'memory_path'):
        value = OmegaConf.select(cfg, f'main.{key}', default=None)
        if value:
            cfg.main[key] = os.path.expanduser(str(value))

    os.makedirs(cfg.main.cache_path, exist_ok=True)
    return cfg



def resolved_config_paths(config_paths=(), use_user_config: bool = True):
    """The files to merge, lowest precedence first.

    The user's own file goes underneath anything passed explicitly, so a config
    given per invocation still wins. Kept as a function because both the loader
    and `config list` have to agree about the order — one of them getting it
    wrong is exactly the sort of thing nobody notices.
    """
    explicit = [p for p in (config_paths or ())]
    return ([user_config_path()] if use_user_config else []) + explicit


def _require(cfg, dotted: str) -> None:
    """Fail with the name of the missing setting rather than an AttributeError
    thrown three layers down when a model is already loading."""
    if OmegaConf.select(cfg, dotted, default=None) in (None, ''):
        raise ValueError(
            f"missing required configuration '{dotted}'. Pass a config file that "
            f"defines it, or set it explicitly."
        )


def ensure_writable_cache_dirs(cfg, who: str = 'worker') -> None:
    """Point the libraries' compilation caches at our own cache directory.

    Two of our dependencies want to write compiled artefacts somewhere, and both
    pick a location that does not exist for a non-root container user — which is
    how you avoid root-owned ``.meta`` files, so it is the normal case, not an
    exotic one:

    * **numba**, which librosa JIT-compiles through with ``cache=True``. It tries
      librosa's own source directory, then a per-user cache. With neither
      writable it reports "no locator available" and *raises*, so every audio file
      fails to be described with a traceback that points at librosa and says
      nothing about permissions.
    * **torch**, for CUDA kernels it JIT-compiles. It warns that "the specified
      kernel cache directory could not be created" and carries on with caching
      disabled, so this one costs time rather than failing outright.

    Call this at the top of a worker process, before anything imports librosa or
    touches CUDA. An explicit value in the environment always wins, so a
    deployment can still put these wherever it likes.
    """
    import os

    from omegaconf import OmegaConf

    cache_path = OmegaConf.select(cfg, 'main.cache_path', default=None)
    if not cache_path:
        return

    for var, subdir in (('NUMBA_CACHE_DIR', 'numba'),
                        ('PYTORCH_KERNEL_CACHE_PATH', 'torch_kernels')):
        if os.environ.get(var):
            continue
        target = os.path.join(cache_path, subdir)
        try:
            os.makedirs(target, exist_ok=True)
            os.environ[var] = target
        except OSError as exc:
            print(f"{who}: could not create {target} ({exc}); "
                  f"{var} is unset and the library will fall back to its default.")


# ---------------------------------------------------------------------------
# The user's own settings
# ---------------------------------------------------------------------------

# Only machine-specific paths. Deliberately not the model names or generation
# settings: those describe *what the engine does*, belong with a project, and are
# passed with --config. What belongs to a machine is where things are kept.
#
# Restricting the set is not tidiness. `set` validates against it, so a typo is
# an error instead of a setting that exists in the file and is read by nothing.
SETTABLE_KEYS = {
    'project_config_path':
        'folder holding cache, memory and the trained evaluator',
    'embedding_models_path':
        'downloaded model weights, shared between projects',
}

# Everything else follows from those two and is not listed, because a setting you
# cannot set is noise in the output of a command whose job is to tell you what
# you can change. Where the derived folders land is documented in README.md.



def user_config_path() -> str:
    """The one file a person edits to configure a machine.

    A fixed location, not somewhere under the project folder: its entire job is
    to say where the project folder *is*, so putting it inside would be circular.
    Honours XDG_CONFIG_HOME like anything else that keeps settings there.
    """
    base = os.environ.get('XDG_CONFIG_HOME') or os.path.join(
        os.path.expanduser('~'), '.config')
    return os.path.join(base, 'anagnorisis', 'config.yaml')


def read_user_config():
    """What the user has set, or an empty config if they have set nothing."""
    path = user_config_path()
    if not os.path.isfile(path):
        return OmegaConf.create({})
    try:
        return OmegaConf.load(path)
    except Exception as exc:
        print(f"anagnorisis: could not read {path} ({exc}); ignoring it.")
        return OmegaConf.create({})


def set_user_setting(key: str, value: str) -> str:
    """Record one setting, and return the file it was written to.

    Only the keys named above are accepted, and only the keys actually set are
    written — never a copy of the defaults. That way a later release changing a
    default still reaches the user, and the file stays short enough to read.
    """
    if key not in SETTABLE_KEYS:
        raise ValueError(
            f"unknown setting {key!r}. Settable: {', '.join(sorted(SETTABLE_KEYS))}")

    # A relative path would be read relative to wherever the command happened to
    # run from, which is never what someone means by a stored setting. `~` is
    # left alone: it stays portable and load_config expands it.
    if not value.startswith('~') and not os.path.isabs(value):
        value = os.path.abspath(value)

    cfg = read_user_config()
    if 'main' not in cfg:
        cfg.main = OmegaConf.create({})
    cfg.main[key] = value
    _save_user_config(cfg)
    return user_config_path()


def unset_user_setting(key: str) -> bool:
    """Drop one setting so its default applies again. True if it was there."""
    if key not in SETTABLE_KEYS:
        raise ValueError(
            f"unknown setting {key!r}. Settable: {', '.join(sorted(SETTABLE_KEYS))}")
    cfg = read_user_config()
    if OmegaConf.select(cfg, f'main.{key}', default=None) is None:
        return False
    del cfg.main[key]
    _save_user_config(cfg)
    return True


def _save_user_config(cfg) -> None:
    """Write via a temporary file in the same directory, then rename, so an
    interrupted write cannot leave a half-parsed settings file behind."""
    path = user_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.partial'
    OmegaConf.save(cfg, tmp)
    os.replace(tmp, path)


def describe_settings(config_paths=(), *, project_config_path=None,
                      models_path=None, use_user_config: bool = True):
    """The settable paths, their values, and where each value came from.

    The provenance is the point. "It is not taking effect" is the most common
    thing to go wrong with layered configuration, and the answer is always which
    layer won — which is invisible unless something prints it.

    Takes the same arguments as :func:`load_config` and derives the merge order
    from the same function, so the two cannot disagree about which file wins.

    Returns:
        list of (key, value, source) sorted by key.
    """
    cfg = load_config(config_paths,
                      project_config_path=project_config_path,
                      models_path=models_path,
                      use_user_config=use_user_config)

    resolved = resolved_config_paths(config_paths, use_user_config)
    labelled_paths = [
        ('user config' if p == user_config_path() else f'--config {p}', p)
        for p in resolved
    ]

    sources = {k: 'package default' for k in SETTABLE_KEYS}

    for label, path in labelled_paths:
        if not (path and os.path.isfile(path)):
            continue
        try:
            raw = OmegaConf.load(path)
        except Exception:
            continue
        for key in SETTABLE_KEYS:
            if OmegaConf.select(raw, f'main.{key}', default=None) is not None:
                sources[key] = label

    for key, value in (('project_config_path', project_config_path),
                       ('embedding_models_path', models_path)):
        if value:
            sources[key] = 'command line'

    return [(k, OmegaConf.select(cfg, f'main.{k}', default=None), sources[k])
            for k in sorted(SETTABLE_KEYS)]
