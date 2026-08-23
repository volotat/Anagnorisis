"""Configuring a machine after installing the package.

`defaults.yaml` is distribution content: it lives inside the installed package,
so editing it to configure a machine loses the change on the next upgrade and
ships the values to whoever installs the wheel. The settings a person owns go in
a file of their own, and these tests pin where it is, what may go in it, and
which layer wins.
"""
import os

import pytest
from omegaconf import OmegaConf

from anagnorisis_core import cli
from anagnorisis_core.config import (SETTABLE_KEYS, describe_settings,
                                     load_config, read_user_config,
                                     resolved_config_paths, set_user_setting,
                                     unset_user_setting, user_config_path)


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Never touch the real ~/.config while testing."""
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path / 'cfg'))
    return tmp_path


class TestWhereItLives:
    def test_it_is_outside_the_installed_package(self, tmp_path):
        import anagnorisis_core
        package = os.path.dirname(anagnorisis_core.__file__)
        assert not user_config_path().startswith(package), (
            'settings inside the package would be lost on upgrade')

    def test_it_honours_xdg(self, tmp_path):
        assert user_config_path() == str(
            tmp_path / 'cfg' / 'anagnorisis' / 'config.yaml')

    def test_it_falls_back_to_dot_config(self, tmp_path, monkeypatch):
        monkeypatch.delenv('XDG_CONFIG_HOME', raising=False)
        monkeypatch.setenv('HOME', str(tmp_path / 'home'))
        assert user_config_path() == str(
            tmp_path / 'home' / '.config' / 'anagnorisis' / 'config.yaml')

    def test_reading_a_missing_file_is_not_an_error(self):
        assert not os.path.exists(user_config_path())
        assert read_user_config() is not None


class TestSetting:
    def test_set_then_load_uses_it(self, tmp_path):
        set_user_setting('project_config_path', str(tmp_path / 'lib'))
        cfg = load_config()
        assert cfg.main.project_config_path == str(tmp_path / 'lib')
        assert cfg.main.cache_path == str(tmp_path / 'lib' / 'cache')

    def test_only_what_was_set_is_written(self, tmp_path):
        """Not a copy of the defaults: a later release changing one must still
        reach the user."""
        set_user_setting('project_config_path', str(tmp_path / 'lib'))
        written = OmegaConf.load(user_config_path())
        assert list(written.main.keys()) == ['project_config_path']
        assert 'omni' not in written and 'embedder' not in written

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match='unknown setting'):
            set_user_setting('projekt_config_path', '/x')
        assert not os.path.exists(user_config_path()), 'it wrote the typo anyway'

    def test_a_relative_path_is_made_absolute(self, tmp_path, monkeypatch):
        """A stored relative path would resolve against whatever directory the
        next command happened to run from."""
        monkeypatch.chdir(tmp_path)
        set_user_setting('project_config_path', 'library')
        assert OmegaConf.load(user_config_path()).main.project_config_path == str(
            tmp_path / 'library')

    def test_a_tilde_is_kept_as_written(self):
        """It stays portable between machines, and load_config expands it."""
        set_user_setting('project_config_path', '~/elsewhere')
        assert OmegaConf.load(
            user_config_path()).main.project_config_path == '~/elsewhere'

    def test_unset_restores_the_default(self, tmp_path):
        default = load_config(use_user_config=False).main.project_config_path
        set_user_setting('project_config_path', str(tmp_path / 'lib'))
        assert unset_user_setting('project_config_path') is True
        assert load_config().main.project_config_path == default

    def test_unset_of_something_never_set_says_so(self):
        assert unset_user_setting('embedding_models_path') is False

    def test_the_file_is_written_atomically(self, tmp_path):
        set_user_setting('embedding_models_path', str(tmp_path / 'm'))
        leftovers = [f for f in os.listdir(os.path.dirname(user_config_path()))
                     if f.endswith('.partial')]
        assert not leftovers, leftovers


class TestPrecedence:
    def test_an_explicit_config_file_beats_the_user_file(self, tmp_path):
        set_user_setting('project_config_path', str(tmp_path / 'mine'))
        project = tmp_path / 'project.yaml'
        OmegaConf.save(OmegaConf.create(
            {'main': {'project_config_path': str(tmp_path / 'theirs')}}), project)

        cfg = load_config([str(project)])
        assert cfg.main.project_config_path == str(tmp_path / 'theirs')

    def test_a_flag_beats_everything(self, tmp_path):
        set_user_setting('project_config_path', str(tmp_path / 'mine'))
        cfg = load_config(project_config_path=str(tmp_path / 'flag'))
        assert cfg.main.project_config_path == str(tmp_path / 'flag')

    def test_the_user_file_sits_under_explicit_ones_in_the_chain(self):
        chain = resolved_config_paths(['/tmp/a.yaml'])
        assert chain[0] == user_config_path()
        assert chain[-1] == '/tmp/a.yaml'

    def test_a_missing_user_file_is_simply_one_fewer_layer(self):
        assert not os.path.exists(user_config_path())
        assert load_config().main.project_config_path

    def test_the_api_sees_what_the_command_line_sees(self, tmp_path):
        """The whole point of the default: configure a machine with
        `anag config set`, then write a script, and get the same library."""
        set_user_setting('project_config_path', str(tmp_path / 'chosen'))
        assert load_config().main.project_config_path == str(tmp_path / 'chosen')

    def test_it_can_be_turned_off(self, tmp_path):
        """For an application that carries its own configuration and must not
        inherit the settings of whoever launched it."""
        set_user_setting('project_config_path', str(tmp_path / 'chosen'))
        cfg = load_config(use_user_config=False)
        assert cfg.main.project_config_path != str(tmp_path / 'chosen')

    def test_turning_it_off_also_drops_it_from_provenance(self, tmp_path):
        set_user_setting('project_config_path', str(tmp_path / 'chosen'))
        sources = {k: s for k, _, s in describe_settings(use_user_config=False)}
        assert sources['project_config_path'] == 'package default'


class TestProvenance:
    """The reason `list` exists: "it is not taking effect" is always a question
    about which layer won, and that is invisible unless something prints it.
    """

    def _sources(self, explicit=(), **flags):
        return {k: s for k, _, s in describe_settings(list(explicit), **flags)}

    def test_untouched_settings_say_package_default(self):
        assert self._sources()['embedding_models_path'] == 'package default'

    def test_a_user_setting_is_attributed_to_the_user_file(self, tmp_path):
        set_user_setting('embedding_models_path', str(tmp_path / 'm'))
        assert self._sources()['embedding_models_path'] == 'user config'

    def test_a_flag_is_attributed_to_the_command_line(self, tmp_path):
        set_user_setting('project_config_path', str(tmp_path / 'mine'))
        sources = self._sources(project_config_path=str(tmp_path / 'flag'))
        assert sources['project_config_path'] == 'command line'

    def test_only_the_settable_paths_are_listed(self):
        """A setting you cannot set is noise in the output of a command whose
        job is to say what you can change. Where the derived folders land is in
        README.md instead."""
        assert set(self._sources()) == set(SETTABLE_KEYS)

    def test_only_two_things_can_be_set(self):
        assert set(SETTABLE_KEYS) == {'project_config_path',
                                      'embedding_models_path'}


class TestTheCommand:
    def _run(self, capsys, *argv):
        code = cli.main(list(argv))
        return code, capsys.readouterr()

    def test_path_prints_the_file(self, capsys):
        code, out = self._run(capsys, 'config', 'path')
        assert code == 0 and out.out.strip() == user_config_path()

    def test_set_get_roundtrip(self, capsys, tmp_path):
        self._run(capsys, 'config', 'set', 'project_config_path', str(tmp_path / 'l'))
        code, out = self._run(capsys, 'config', 'get', 'project_config_path')
        assert code == 0 and out.out.strip() == str(tmp_path / 'l')

    def test_set_reports_where_it_went(self, capsys, tmp_path):
        _, out = self._run(capsys, 'config', 'set', 'embedding_models_path',
                           str(tmp_path / 'm'))
        assert user_config_path() in out.out

    def test_a_derived_path_cannot_be_set(self, capsys, tmp_path):
        code, out = self._run(capsys, 'config', 'set', 'cache_path',
                              str(tmp_path / 'c'))
        assert code == 2
        assert 'project_config_path' in out.err

    def test_list_shows_values_and_sources(self, capsys, tmp_path):
        self._run(capsys, 'config', 'set', 'project_config_path', str(tmp_path / 'l'))
        code, out = self._run(capsys, 'config', 'list')
        assert code == 0
        assert 'project_config_path' in out.out
        assert '(user config)' in out.out
        assert '(package default)' in out.out

    def test_an_unknown_key_exits_nonzero_and_names_the_valid_ones(self, capsys):
        code, out = self._run(capsys, 'config', 'set', 'nonsense', '/x')
        assert code == 2
        assert 'project_config_path' in out.err

    def test_set_without_a_value_is_refused(self, capsys):
        code, out = self._run(capsys, 'config', 'set', 'embedding_models_path')
        assert code == 2 and 'needs a value' in out.err

    def test_get_without_a_key_is_refused(self, capsys):
        code, out = self._run(capsys, 'config', 'get')
        assert code == 2

    def test_config_runs_even_when_settings_are_unusable(self, capsys, tmp_path):
        """`config set` is how you fix a broken configuration, so it must not
        need a working one."""
        os.makedirs(os.path.dirname(user_config_path()), exist_ok=True)
        with open(user_config_path(), 'w') as fh:
            fh.write('main: [this is not a mapping\n')
        code, _ = self._run(capsys, 'config', 'path')
        assert code == 0
