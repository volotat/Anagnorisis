"""
Tests for the URL block in anagnorisis_core/embedder.py

The embedding model takes text and media as the same type — plain strings — so
it cannot tell them apart by signature. It guesses instead: anything starting
with http(s) is handed to `urllib.request.urlretrieve` and whatever comes back
is sniffed for a media type. That guess runs over every string in every batch,
before any embedding happens. A note beginning with a link therefore became an
outbound request, and the fetched bytes could be embedded in place of the text.

Reading a file must read the file and nothing else. These tests pin that down
without needing the model, so the day the vendored code changes shape, the
failure is a red test rather than silent network access.
"""
import sys
import types
import pytest

from anagnorisis_core.models.embedder import _block_url_fetching


def _fake_custom_st(name):
    """A stand-in for the model's vendored module, with its two markers."""
    mod = types.ModuleType(name)
    mod.touched = []          # anything the resolver did to the outside world

    def _download_if_url(x):
        if isinstance(x, str) and x.startswith(('http://', 'https://')):
            mod.touched.append(('network', x))
            return f'/tmp/downloaded/{len(mod.touched)}'
        return x

    def _resolve_input(x):
        if isinstance(x, str):
            local = _download_if_url(x)
            mod.touched.append(('stat', local))     # stands in for isfile+sniff
            if local.endswith(('.jpg', '.png')):
                return ('image', local)
            return ('text', x)
        return ('text', str(x))

    mod._download_if_url = _download_if_url
    mod._resolve_input = _resolve_input
    mod._is_media_string = lambda x: mod._resolve_input(x)[0] != 'text'
    return mod


@pytest.fixture
def vendored(monkeypatch):
    mod = _fake_custom_st('transformers_modules.jinaai_fake.custom_st')
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return mod


class TestBlocking:
    def test_url_reaches_the_network_before_blocking(self, vendored):
        """Guard the guard: if this stops being true, the rest proves nothing."""
        vendored._is_media_string('https://example.com/cat.jpg')
        assert ('network', 'https://example.com/cat.jpg') in vendored.touched

    def test_url_is_never_fetched_after_blocking(self, vendored):
        _block_url_fetching()
        vendored.touched.clear()

        assert vendored._resolve_input('https://example.com/cat.jpg') \
            == ('text', 'https://example.com/cat.jpg')
        assert vendored._is_media_string('https://example.com/cat.jpg') is False
        assert vendored.touched == [], 'the model touched the outside world'

    def test_url_is_not_even_stat_checked(self, vendored):
        """The whole resolution is skipped, not just the download."""
        _block_url_fetching()
        vendored.touched.clear()
        vendored._resolve_input('http://example.com/whatever')
        assert vendored.touched == []

    def test_malformed_url_in_prose_is_just_text(self, vendored):
        """The real report: a note whose first line was a link with a colon."""
        _block_url_fetching()
        line = 'https://example.com: vinteo - f9jbmyrXyuKTnCV8 - notes'
        assert vendored._resolve_input(line) == ('text', line)

    def test_local_media_still_resolves(self, vendored):
        """The block targets URLs only — local files must be unaffected."""
        _block_url_fetching()
        assert vendored._resolve_input('/mnt/media/images/cat.jpg') \
            == ('image', '/mnt/media/images/cat.jpg')
        assert vendored._is_media_string('/mnt/media/images/cat.jpg') is True

    def test_ordinary_text_still_resolves_as_text(self, vendored):
        _block_url_fetching()
        assert vendored._resolve_input('a note about cats') == ('text', 'a note about cats')

    def test_blocks_every_vendored_copy(self, monkeypatch):
        """The module is loaded dynamically and may be present more than once."""
        mods = [_fake_custom_st(f'transformers_modules.copy{i}.custom_st')
                for i in range(3)]
        for m in mods:
            monkeypatch.setitem(sys.modules, m.__name__, m)

        _block_url_fetching()

        for m in mods:
            m.touched.clear()
            assert m._resolve_input('https://x/y.png') == ('text', 'https://x/y.png')
            assert m.touched == []

    def test_applying_twice_does_not_nest(self, vendored):
        """initiate() can run more than once in a process; wrappers must not stack."""
        _block_url_fetching()
        first = vendored._resolve_input
        _block_url_fetching()
        assert vendored._resolve_input is first


class TestWarning:
    def test_warns_when_nothing_could_be_patched(self, monkeypatch, capsys):
        """Silence here would mean the network access quietly came back."""
        monkeypatch.setattr(
            sys, 'modules',
            {k: v for k, v in sys.modules.items()
             if not (hasattr(v, '_resolve_input') and hasattr(v, '_is_media_string'))},
        )
        _block_url_fetching()
        assert 'WARNING' in capsys.readouterr().out

    def test_silent_when_it_worked(self, vendored, capsys):
        _block_url_fetching()
        assert 'WARNING' not in capsys.readouterr().out

    def test_ignores_unrelated_and_none_modules(self, monkeypatch, vendored):
        """sys.modules holds None entries and unrelated modules; neither may break it."""
        monkeypatch.setitem(sys.modules, 'a_none_entry', None)
        unrelated = types.ModuleType('unrelated')
        unrelated._resolve_input = 'not even callable'
        monkeypatch.setitem(sys.modules, 'unrelated', unrelated)

        _block_url_fetching()  # must not raise

        assert vendored._resolve_input('https://x/y.png') == ('text', 'https://x/y.png')
        assert unrelated._resolve_input == 'not even callable', 'patched the wrong module'
