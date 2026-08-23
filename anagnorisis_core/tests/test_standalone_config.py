"""The package has to work on its own.

Installed from a wheel, with no Anagnorisis checkout anywhere and no config file
to point at, `anagnorisis describe /some/folder` must start. These tests pin the
two halves of that: settings come from the package, and the command line does not
require a config file to exist.
"""
import os

import pytest
from omegaconf import OmegaConf

from anagnorisis_core import cli
from anagnorisis_core.config import bundled_defaults_path, load_config


class TestBundledDefaults:
    def test_the_defaults_file_ships_with_the_package(self):
        assert os.path.isfile(bundled_defaults_path()), (
            'defaults.yaml is missing — an install with no checkout cannot start')

    def test_config_loads_with_no_config_file_at_all(self, tmp_path):
        """The whole point: no --config, no repository, still a usable config."""
        cfg = load_config([], project_config_path=str(tmp_path / 'proj'),
                          models_path=str(tmp_path / 'models'))
        assert cfg.omni.model_name
        assert cfg.embedder.model_name
        assert cfg.main.cache_path

    def test_it_carries_the_settings_the_engine_cannot_default(self, tmp_path):
        """Prompts fall back to '' in code, which would silently produce useless
        descriptions rather than an error. They belong in the defaults file."""
        cfg = load_config([], project_config_path=str(tmp_path / 'p'),
                          models_path=str(tmp_path / 'm'))
        for key in ('image_prompt', 'audio_prompt', 'video_prompt', 'text_prompt'):
            assert cfg.omni.get(key, ''), f'omni.{key} is not in the defaults'
        assert cfg.omni.get('max_new_tokens')
        assert cfg.embedder.get('embedding_dimension')

    def test_a_tilde_is_expanded(self, tmp_path, monkeypatch):
        """`~` is literal to YAML. Left alone it creates a directory called '~'
        in whatever folder the command happened to be run from.

        Driven by an explicit value rather than by whatever defaults.yaml
        currently says: that file is meant to be edited, so asserting the paths
        it ships with would make somebody's own configuration a test failure.
        """
        monkeypatch.setenv('HOME', str(tmp_path))
        cfg = load_config([], project_config_path='~/proj',
                          models_path='~/weights')
        for key in ('project_config_path', 'cache_path', 'memory_path',
                    'personal_models_path', 'embedding_models_path'):
            value = cfg.main[key]
            assert not value.startswith('~'), (key, value)
            assert value.startswith(str(tmp_path)), (key, value)

    def test_whatever_the_shipped_defaults_say_is_absolute(self):
        """The values are the user's business; that they are usable is not."""
        cfg = load_config([])
        for key in ('project_config_path', 'cache_path', 'memory_path',
                    'personal_models_path', 'embedding_models_path'):
            value = cfg.main[key]
            assert value and os.path.isabs(value), (key, value)

    def test_a_config_file_overrides_key_by_key(self, tmp_path):
        """Overriding one value must not mean restating the whole section — this
        is what lets the application's config.yaml win where it differs."""
        user = tmp_path / 'mine.yaml'
        OmegaConf.save(OmegaConf.create({'omni': {'max_new_tokens': 64}}), user)

        cfg = load_config([str(user)], project_config_path=str(tmp_path / 'p'),
                          models_path=str(tmp_path / 'm'))

        assert cfg.omni.max_new_tokens == 64, 'the override did not apply'
        assert cfg.omni.model_name, 'the rest of the section was lost'
        assert cfg.omni.get('image_prompt'), 'the rest of the section was lost'

    def test_explicit_paths_beat_the_defaults(self, tmp_path):
        cfg = load_config([], project_config_path=str(tmp_path / 'chosen'),
                          models_path=str(tmp_path / 'models'))
        assert cfg.main.project_config_path == str(tmp_path / 'chosen')
        assert cfg.main.embedding_models_path == str(tmp_path / 'models')


class TestHelpForOneCommand:
    """`anag -h describe` printed the top-level help — the one page the user was
    trying to get past. argparse gives `-h` to whichever parser sees it first, so
    the argument list is rewritten before parsing.
    """

    @pytest.fixture
    def parser(self):
        return cli._build_parser()

    @pytest.mark.parametrize('argv', [
        ['-h', 'describe'],
        ['--help', 'describe'],
        ['describe', '-h'],
        ['help', 'describe'],
    ])
    def test_every_spelling_reaches_the_subcommand(self, argv, parser):
        assert cli._reorder_help(argv, parser) == ['describe', '--help'], argv

    @pytest.mark.parametrize('argv', [['-h'], ['--help'], ['help']])
    def test_bare_help_still_shows_the_overview(self, argv, parser):
        """With no command named there is nothing to redirect to, so the result
        just has to still be a request for the overview — `-h` passed through
        unchanged is as correct as `--help`."""
        result = cli._reorder_help(argv, parser)
        assert result in (['-h'], ['--help']), result

    def test_the_overview_still_prints(self, capsys, parser):
        with pytest.raises(SystemExit):
            cli.main(['-h'])
        out = capsys.readouterr().out
        assert 'describe' in out and 'search' in out

    def test_an_unknown_topic_falls_back_to_the_overview(self, parser):
        assert cli._reorder_help(['help', 'nonsense'], parser) == ['--help']

    def test_a_normal_invocation_is_untouched(self, parser):
        argv = ['describe', '/media', '--sink', 'meta']
        assert cli._reorder_help(argv, parser) == argv

    def test_help_actually_prints_the_subcommand_usage(self, capsys, parser):
        with pytest.raises(SystemExit):
            cli.main(['-h', 'describe'])
        out = capsys.readouterr().out
        assert '--sink' in out, 'that is not the describe help'
        assert 'never touched' in out or 'sidecar' in out


class TestPathsThatMustHaveDefaults:
    """Ratings and the evaluator need somewhere to live too.

    `--memory` used to be required=True on rate, sort and train, so three of the
    six commands could not run without being told a path — which is the opposite
    of standing alone.
    """

    def test_memory_and_personal_models_have_defaults(self):
        """Both must resolve to something without being asked for, and both must
        sit inside the project folder wherever that has been pointed."""
        cfg = load_config([])
        project = cfg.main.project_config_path
        assert cfg.main.memory_path.startswith(project), cfg.main.memory_path
        assert cfg.main.personal_models_path.startswith(project), (
            cfg.main.personal_models_path)

    def test_memory_is_data_not_cache(self, tmp_path, monkeypatch):
        """Memory files cannot be regenerated — the file they describe may be
        gone — so they must not sit under a cache directory somebody might wipe.
        """
        monkeypatch.setenv('HOME', str(tmp_path))
        cfg = load_config([])
        assert cfg.main.cache_path not in cfg.main.memory_path, (
            f'memory ({cfg.main.memory_path}) is inside the cache '
            f'({cfg.main.cache_path})')

    @pytest.mark.parametrize('command', ['rate', 'sort', 'train'])
    def test_memory_is_no_longer_a_required_flag(self, command):
        parser = cli._build_parser()
        sub = parser._subparsers._group_actions[0].choices[command]
        memory = next(a for a in sub._actions if '--memory' in a.option_strings)
        assert not memory.required, f'{command} still demands --memory'


class TestDefaultsFileLocation:
    def test_it_sits_at_the_top_of_the_package(self):
        """Not in a data/ subdirectory: it is the file somebody edits, so it
        should be the first one they see."""
        path = bundled_defaults_path()
        parent = os.path.basename(os.path.dirname(path))
        assert parent == 'anagnorisis_core', path
        assert os.path.basename(path) == 'defaults.yaml'


class TestDescribeDefaults:
    def test_the_default_sink_is_the_cache(self):
        """Writing into somebody's media folders is opt-in. `--sink meta` is for
        sharing a folder; it should never be what a bare `describe` does.
        """
        parser = cli._build_parser()
        describe = parser._subparsers._group_actions[0].choices['describe']
        sink = next(a for a in describe._actions if '--sink' in a.option_strings)
        assert sink.default == 'cache', sink.default

    def test_a_bare_describe_parses_with_no_flags_at_all(self):
        parser = cli._build_parser()
        args = parser.parse_args(['describe', '/some/folder'])
        assert args.sink == 'cache'
        assert args.config == []
        assert args.project_config is None and args.models is None


class TestOneProjectFolder:
    """Cache, memory and the trained evaluator are one folder, laid out the way
    the application lays out project_config/ — so pointing the command line at an
    app checkout makes the two share everything, and there is one path to
    configure rather than four.
    """

    def test_the_three_subfolders_are_derived(self, tmp_path):
        cfg = load_config([], project_config_path=str(tmp_path / 'proj'))
        assert cfg.main.cache_path == str(tmp_path / 'proj' / 'cache')
        assert cfg.main.memory_path == str(tmp_path / 'proj' / 'memory')
        assert cfg.main.personal_models_path == str(tmp_path / 'proj' / 'models')

    def test_it_matches_the_application_layout(self, tmp_path):
        """The app builds project_config/{cache,memory,models}. If these ever
        disagree, pointing the CLI at a checkout would quietly use a second set
        of folders next to the real ones."""
        proj = tmp_path / 'project_config'
        cfg = load_config([], project_config_path=str(proj))
        for sub in ('cache', 'memory', 'models'):
            assert str(proj / sub) in (cfg.main.cache_path, cfg.main.memory_path,
                                       cfg.main.personal_models_path), sub

    def test_models_stay_outside_the_project_folder(self, tmp_path):
        """Weights are gigabytes and identical for everyone, so they are shared
        rather than duplicated per project."""
        cfg = load_config([], project_config_path=str(tmp_path / 'proj'))
        assert not cfg.main.embedding_models_path.startswith(
            str(tmp_path / 'proj')), cfg.main.embedding_models_path

    def test_a_config_file_cannot_split_them_apart(self, tmp_path):
        """The sub-paths follow the project folder and are not settable beside
        it. Allowing that is how you end up with a memory folder belonging to one
        library and a cache belonging to another — silent, and hard to notice.
        """
        user = tmp_path / 'c.yaml'
        OmegaConf.save(OmegaConf.create(
            {'main': {'memory_path': str(tmp_path / 'my-memory'),
                      'cache_path': str(tmp_path / 'my-cache')}}), user)

        cfg = load_config([str(user)], project_config_path=str(tmp_path / 'proj'))

        assert cfg.main.memory_path == str(tmp_path / 'proj' / 'memory')
        assert cfg.main.cache_path == str(tmp_path / 'proj' / 'cache')

    def test_the_taxonomy_always_comes_from_the_package(self, tmp_path):
        """Not configurable at all: the tag vocabularies are what the caches and
        the trained evaluator were built against."""
        from anagnorisis_core.media.media_types import bundled_media_types_dir
        cfg = load_config([], project_config_path=str(tmp_path / 'proj'))
        assert cfg.main.media_types_path == bundled_media_types_dir()

    def test_the_cli_exposes_it(self):
        parser = cli._build_parser()
        args = parser.parse_args(['describe', '/m', '--project-config', '/p'])
        assert args.project_config == '/p'


class TestNoSymlinks:
    """The packaged data files must be real files.

    They were symlinks for a while — the project root held the originals and the
    package linked up to them, so the files could be seen without digging. That
    cost portability: a filesystem or a git checkout without symlink support gets
    a broken package, and the failure is a confusing missing-file rather than
    anything that names the cause. Neither file is meant to be edited any more,
    so the visibility was not worth the fragility.
    """

    def test_the_packaged_files_are_not_links(self):
        from anagnorisis_core.config import bundled_defaults_path
        from anagnorisis_core.media.media_types import bundled_media_types_dir

        for path in (bundled_defaults_path(), bundled_media_types_dir()):
            assert os.path.exists(path), path
            assert not os.path.islink(path), f'{path} is a symlink'

    def test_nothing_in_the_package_is_a_link(self):
        """Not just those two — a link anywhere under the package would ship
        broken for the same reason."""
        import pathlib

        import anagnorisis_core
        package = pathlib.Path(anagnorisis_core.__file__).parent
        links = [str(p) for p in package.rglob('*') if p.is_symlink()]
        assert not links, links

    def test_the_data_lives_beside_the_code(self):
        """Which is what lets dirname(__file__) find it identically whether the
        package is installed or run from the source tree."""
        import pathlib

        import anagnorisis_core
        from anagnorisis_core.config import bundled_defaults_path

        package = pathlib.Path(anagnorisis_core.__file__).parent.resolve()
        assert pathlib.Path(bundled_defaults_path()).resolve().parent == package
