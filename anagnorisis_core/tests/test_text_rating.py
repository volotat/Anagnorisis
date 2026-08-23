"""Rating and scoring text that has no file behind it.

Two halves of the same idea: ask the evaluator what it makes of a paragraph, and
teach it a paragraph you have an opinion about. Both matter because the model's
subject is descriptions, not files — so anything you can describe, you can rate.
"""
import os

import numpy as np
import pytest
from omegaconf import OmegaConf

from anagnorisis_core import api, cli
from anagnorisis_core.media.media_types import bundled_media_types_dir
from anagnorisis_core.media import description
from anagnorisis_core.ratings.memory import (save_text_rating, text_memory_key)
from anagnorisis_core.ratings.training import _parse_memory_file


@pytest.fixture
def cfg(tmp_path):
    conf = OmegaConf.load(os.path.join(bundled_media_types_dir(),
                                       'media_types.yaml'))
    conf.main = {'cache_path': str(tmp_path / 'cache'),
                 'memory_path': str(tmp_path / 'memory'),
                 'personal_models_path': str(tmp_path / 'personal'),
                 'embedding_models_path': str(tmp_path / 'models'),
                 'media_types_path': bundled_media_types_dir()}
    conf.omni = {'model_name': 'x'}
    conf.embedder = {'model_name': 'y'}
    return conf


class TestRememberingText:
    def test_it_writes_the_shape_training_reads(self, tmp_path):
        """Rating on line 1, text below — identical to a rated file's memory, so
        training cannot tell the two apart, which is the point."""
        path = save_text_rating('slow acoustic recordings', 9,
                                memory_dir=str(tmp_path / 'm'))
        rating, body = _parse_memory_file(open(path).read())
        assert rating == 9.0
        assert body == 'slow acoustic recordings'

    def test_multiline_text_survives(self, tmp_path):
        text = 'a long noisy live set\nrecorded badly\nwith clipping'
        path = save_text_rating(text, 2, memory_dir=str(tmp_path / 'm'))
        rating, body = _parse_memory_file(open(path).read())
        assert rating == 2.0 and body == text

    def test_the_same_text_replaces_its_earlier_record(self, tmp_path):
        """Named by its own content, because there is no path that could move.
        Two files would teach the model the same example twice."""
        memory = str(tmp_path / 'm')
        first = save_text_rating('quiet fingerpicking', 8, memory_dir=memory)
        second = save_text_rating('quiet fingerpicking', 3, memory_dir=memory)
        assert first == second
        rating, _ = _parse_memory_file(open(second).read())
        assert rating == 3.0, 'the newer rating did not win'

    def test_surrounding_whitespace_does_not_make_a_new_record(self, tmp_path):
        memory = str(tmp_path / 'm')
        a = save_text_rating('quiet fingerpicking', 8, memory_dir=memory)
        b = save_text_rating('  quiet fingerpicking\n', 8, memory_dir=memory)
        assert a == b

    def test_different_text_is_a_different_record(self, tmp_path):
        memory = str(tmp_path / 'm')
        a = save_text_rating('quiet fingerpicking', 8, memory_dir=memory)
        b = save_text_rating('loud drumming', 8, memory_dir=memory)
        assert a != b

    def test_empty_text_is_refused(self, tmp_path):
        """An empty training example is noise with a score attached."""
        for empty in ('', '   ', '\n\n'):
            with pytest.raises(ValueError, match='empty'):
                save_text_rating(empty, 5, memory_dir=str(tmp_path / 'm'))

    def test_nothing_partial_is_left_behind(self, tmp_path):
        memory = tmp_path / 'm'
        save_text_rating('something', 5, memory_dir=str(memory))
        leftovers = [f for d in memory.iterdir() for f in d.iterdir()
                     if f.name.endswith('.partial')]
        assert not leftovers

    def test_the_key_is_stable(self):
        assert text_memory_key('abc') == text_memory_key('abc')
        assert text_memory_key('abc') != text_memory_key('abd')

    def test_the_api_wrapper_uses_the_configured_memory(self, cfg, tmp_path):
        written = api.rate_text('a folk ballad', 7, cfg=cfg,
                                memory_dir=cfg.main.memory_path)
        assert written.startswith(cfg.main.memory_path)


class TestScoringText:
    """The number has to be comparable with the ones files get, or it means
    nothing. That comes down to embedding the text the same way."""

    def _stub(self, monkeypatch, cfg, recorder):
        class FakeEmbedder:
            def embed_long_text(self, text):
                recorder['embedded'] = text
                return [np.ones(4, dtype=np.float32)]

        class FakeEvaluator:
            def predict(self, embeddings):
                recorder['predicted'] = embeddings
                return [7.25]

        monkeypatch.setattr('anagnorisis_core.models.embedder.get_omni_embedder',
                            lambda cfg: FakeEmbedder())
        monkeypatch.setattr(api, '_load_evaluator', lambda cfg: FakeEvaluator())

    def test_it_returns_the_models_number(self, cfg, monkeypatch):
        rec = {}
        self._stub(monkeypatch, cfg, rec)
        assert api.score_text('a quiet acoustic recording', cfg=cfg) == 7.25

    def test_it_embeds_as_a_document_not_a_query(self, cfg, monkeypatch):
        """The embedder is asymmetric and the evaluator was trained on document
        vectors. Scoring through the query tower would give a number from a
        different space — quietly wrong rather than obviously so."""
        rec = {}
        self._stub(monkeypatch, cfg, rec)
        api.score_text('  a quiet acoustic recording  ', cfg=cfg)
        assert rec['embedded'] == 'a quiet acoustic recording'

    def test_empty_text_is_refused(self, cfg, monkeypatch):
        rec = {}
        self._stub(monkeypatch, cfg, rec)
        with pytest.raises(ValueError, match='nothing to score'):
            api.score_text('   ', cfg=cfg)

    def test_no_trained_model_says_what_to_run(self, cfg):
        with pytest.raises(FileNotFoundError, match='anagnorisis train'):
            api.score_text('anything', cfg=cfg)

    def test_an_embedder_that_returns_nothing_is_an_error(self, cfg, monkeypatch):
        class Empty:
            def embed_long_text(self, text):
                return []
        monkeypatch.setattr('anagnorisis_core.models.embedder.get_omni_embedder',
                            lambda cfg: Empty())
        monkeypatch.setattr(api, '_load_evaluator',
                            lambda cfg: type('E', (), {'predict': lambda s, e: [1]})())
        with pytest.raises(RuntimeError, match='returned nothing'):
            api.score_text('anything', cfg=cfg)


class TestTheSizeRule:
    """Short text is its own best description; long text is worth compressing.
    The same rule has to apply wherever text arrives — a rated paragraph, a rated
    .txt, and the scoring of either — or the model is taught one thing and asked
    about another.
    """

    def test_short_text_is_kept_as_it_stands(self, cfg):
        body, summarised = description.body_for_text(
            'a short note', cfg=cfg, summarise=lambda t: 'SUMMARY')
        assert body == 'a short note' and summarised is False

    def test_long_text_is_summarised(self, cfg):
        long_text = 'x' * (description.text_verbatim_limit(cfg) + 1)
        body, summarised = description.body_for_text(
            long_text, cfg=cfg, summarise=lambda t: 'SUMMARY')
        assert body == 'SUMMARY' and summarised is True

    def test_the_limit_is_configurable(self, cfg):
        cfg.omni.text_verbatim_max_chars = 5
        body, summarised = description.body_for_text(
            'more than five', cfg=cfg, summarise=lambda t: 'SUMMARY')
        assert summarised is True and body == 'SUMMARY'

    def test_long_text_without_a_summariser_keeps_the_opening(self, cfg):
        """Rather than nothing: the opening is where a document usually says
        what it is."""
        cfg.omni.text_verbatim_max_chars = 10
        body, summarised = description.body_for_text(
            'abcdefghijKLMNOP', cfg=cfg, summarise=None)
        assert body == 'abcdefghij' and summarised is False

    def test_rating_short_text_loads_no_model(self, cfg, monkeypatch):
        """The common case must cost nothing — no descriptor, no GPU."""
        def explode(*a, **k):
            raise AssertionError('a model was loaded for short text')
        monkeypatch.setattr(
            'anagnorisis_core.models.descriptor.OmniDescriptor', explode)
        api.rate_text('a short note', 6, cfg=cfg,
                      memory_dir=cfg.main.memory_path)

    def test_rating_long_text_stores_the_summary(self, cfg, monkeypatch):
        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): pass
            def unload(self): FakeDescriptor.unloaded = True
            def describe_text(self, content): return 'the gist of it'
        FakeDescriptor.unloaded = False
        monkeypatch.setattr(
            'anagnorisis_core.models.descriptor.OmniDescriptor', FakeDescriptor)

        long_text = 'y' * (description.text_verbatim_limit(cfg) + 1)
        path = api.rate_text(long_text, 4, cfg=cfg,
                             memory_dir=cfg.main.memory_path)

        rating, body = _parse_memory_file(open(path).read())
        assert rating == 4.0 and body == 'the gist of it'
        assert FakeDescriptor.unloaded, 'the descriptor was left resident'

    def test_the_key_follows_the_text_not_the_summary(self, cfg, monkeypatch):
        """Rating the same passage twice must land on one file whether or not a
        model was involved."""
        class FakeDescriptor:
            def __init__(self, cfg): pass
            def initiate(self, models_folder): pass
            def unload(self): pass
            def describe_text(self, content): return 'gist one'
        monkeypatch.setattr(
            'anagnorisis_core.models.descriptor.OmniDescriptor', FakeDescriptor)
        long_text = 'z' * (description.text_verbatim_limit(cfg) + 1)

        first = api.rate_text(long_text, 4, cfg=cfg,
                              memory_dir=cfg.main.memory_path)
        assert first == os.path.join(
            os.path.dirname(first), text_memory_key(long_text) + '.md')


class TestTheCommands:
    def test_score_is_registered_and_remember_is_not(self):
        commands = cli._build_parser()._subparsers._group_actions[0].choices
        assert 'score' in commands
        assert 'remember' not in commands, (
            'rating text is rating; it does not need a command of its own')

    def test_rate_takes_a_file_or_text(self):
        parser = cli._build_parser()
        by_file = parser.parse_args(['rate', '/m/song.mp3', '8.5'])
        assert by_file.file == '/m/song.mp3' and by_file.text is None
        by_text = parser.parse_args(['rate', '--text', 'some text', '8.5'])
        assert by_text.file is None and by_text.text == 'some text'

    def test_score_takes_a_file_or_text(self):
        parser = cli._build_parser()
        assert parser.parse_args(['score', '/m/song.mp3']).file == '/m/song.mp3'
        assert parser.parse_args(['score', '--text', 'hi']).text == 'hi'

    def test_both_at_once_is_refused(self, capsys):
        args = cli._build_parser().parse_args(['score', '/m/f.mp3'])
        args.text = 'also text'
        assert cli._one_target(args, 'score') is None
        assert 'not both' in capsys.readouterr().err

    def test_neither_is_refused(self, capsys):
        args = cli._build_parser().parse_args(['score'])
        assert cli._one_target(args, 'score') is None
        assert 'needs a file or --text' in capsys.readouterr().err

    def test_a_dash_means_standard_input(self, monkeypatch):
        """The interesting text usually has newlines in it, which is miserable to
        pass as a shell argument and natural to pipe."""
        import io
        monkeypatch.setattr('sys.stdin', io.StringIO('piped text\nsecond line\n'))
        assert cli._read_text('-') == 'piped text\nsecond line\n'

    def test_anything_else_is_taken_literally(self):
        assert cli._read_text('a literal argument') == 'a literal argument'
