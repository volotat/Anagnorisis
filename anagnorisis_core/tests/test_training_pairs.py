"""
Tests for reading memory files in anagnorisis_core/training.py.

This exists because evaluator training was completely broken and silent about
it: the gather loop unpacked three values from a parser that returns two, so it
raised ValueError on the very first memory file. Nothing exercised the path, so
the model simply could not be retrained and no test failed.

The invariant: the rating comes off line 1 and never reaches the embedder. If it
did, the evaluator would learn to read the score instead of judging the file.
"""
import os
import pytest

from anagnorisis_core.ratings.training import _gather_from_memory, _parse_memory_file


class TestParse:
    def test_returns_rating_and_description(self):
        rating, desc = _parse_memory_file(
            'Rating: 7.5\nFile Name: cat.jpg\n\nTags: cat, sunlight\n')
        assert rating == 7.5
        assert 'Tags: cat, sunlight' in desc

    def test_the_rating_never_reaches_the_description(self):
        _, desc = _parse_memory_file('Rating: 9\nTags: cat\n')
        assert 'Rating' not in desc, 'the score must not be in what gets embedded'

    @pytest.mark.parametrize('text', [
        'no rating here\nTags: cat\n',
        'Rating: not-a-number\nTags: cat\n',
        '',
    ])
    def test_unusable_files_are_rejected_not_guessed(self, text):
        assert _parse_memory_file(text) == (None, None)


class TestEpochStatus:
    """The training loop runs for thousands of epochs. A status that never
    changes gives no sign of whether the model is improving or still moving."""

    def _status(self, **kw):
        from anagnorisis_core.ratings.training import _epoch_status
        args = dict(elapsed=60.0, time_budget_seconds=None,
                    best_accuracy=0.7831, best_epoch=388, current_accuracy=0.7790)
        args.update(kw)
        return _epoch_status(args.pop('epoch', 412), args.pop('total_epochs', 5001), **args)

    def test_reports_progress_and_the_running_best(self):
        s = self._status()
        assert 'Epoch 412/5001' in s, s
        assert '4589 left' in s, s
        # The best is the number worth watching: it is the epoch whose weights
        # are kept, not whatever the current epoch happened to score.
        assert 'Best 78.31% at epoch 388' in s, s
        assert 'now 77.90%' in s, s

    def test_a_time_budget_caps_the_estimate(self):
        """The loop stops at whichever limit comes first, so an estimate built
        only from the epoch rate would promise time the run will not take."""
        unbounded = self._status()
        bounded = self._status(time_budget_seconds=120)
        assert '11m' in unbounded, unbounded
        assert '1m 0s' in bounded, bounded

    def test_no_negative_time_when_the_budget_is_already_spent(self):
        s = self._status(elapsed=200.0, time_budget_seconds=120)
        assert '~0s' in s, s
        assert '-' not in s.split('(~')[1], s

    def test_the_last_epoch_has_nothing_left(self):
        s = self._status(epoch=5001)
        assert '0 left' in s, s


class TestGather:
    """The loop that used to raise on its first file."""

    @pytest.fixture
    def memory(self, tmp_path):
        day = tmp_path / '2026-08-18'
        day.mkdir()
        for i, rating in enumerate([3.0, 8.0]):
            (day / f'{"a" * 8}{i}.md').write_text(
                f'Rating: {rating}\nFile Name: f{i}.jpg\n\n'
                f'Tags: something descriptive number {i}\n', encoding='utf-8')
        (day / 'notes.md').write_text('not a memory file\n', encoding='utf-8')
        return tmp_path

    def test_reads_every_usable_file_without_raising(self, memory, monkeypatch):
        from omegaconf import OmegaConf
        cfg = OmegaConf.create({'main': {
            'memory_path': str(memory),
            'cache_path': str(memory / 'cache'),
        }})

        # Mirrors the real embedder, including the embed_text alias, and asserts
        # that it does: a fake with a different surface than the real class will
        # happily vouch for a call that cannot work in production.
        class FakeEmbedder:
            def embed_long_text(self, long_text):
                import numpy as np
                return [np.ones(8, dtype='float32')]

            embed_text = embed_long_text

        from anagnorisis_core.models.embedder import OmniEmbedder
        for name in ('embed_long_text', 'embed_text'):
            assert hasattr(OmniEmbedder, name), (
                f'the real embedder has no {name}(); this fake is lying')

        embeddings, scores = _gather_from_memory(cfg, FakeEmbedder())
        assert len(scores) == 2, f'expected both memory files, got {scores}'
        assert sorted(scores) == [3.0, 8.0]

    def test_progress_is_reported_on_every_item(self, tmp_path):
        """A status that moves once per hundred items cannot be shown once per
        second, however well the transport throttles — and a long pass then
        looks frozen. The producer reports freely; the consumers throttle."""
        from omegaconf import OmegaConf
        import numpy as np

        day = tmp_path / '2026-08-18'
        day.mkdir()
        for i in range(12):
            (day / f'{i:032x}.md').write_text(
                f'Rating: 5.0\nFile Name: f{i}.jpg\n\nTags: description {i}\n',
                encoding='utf-8')

        cfg = OmegaConf.create({'main': {
            'memory_path': str(tmp_path),
            'cache_path': str(tmp_path / 'cache'),
        }})

        class FakeEmbedder:
            model_hash = 'testhash'

            def embed_long_text(self, long_text):
                return [np.ones(8, dtype='float32')]

        messages = []
        _gather_from_memory(cfg, FakeEmbedder(), status_callback=messages.append)

        embedding_msgs = [m for m in messages if m.startswith('Embedding memory')]
        assert len(embedding_msgs) == 12, (
            f'expected one status per item, got {len(embedding_msgs)}: {messages}')
        assert any(m.startswith('Reading memory files') for m in messages), messages
        # The count has to say what it is counting towards, or it is just a
        # number that stops at an unknowable point.
        assert '(1/12' in embedding_msgs[0], embedding_msgs[0]
        assert '(12/12' in embedding_msgs[-1], embedding_msgs[-1]

    def test_embeddings_are_reused_on_the_second_run(self, tmp_path):
        """Memory files are immutable: the filename is the soft hash and a new
        opinion is a new file. Re-embedding the whole corpus on every training
        run was the bulk of the wait."""
        from omegaconf import OmegaConf
        import numpy as np

        day = tmp_path / '2026-08-18'
        day.mkdir()
        for i in range(5):
            (day / f'{i:032x}.md').write_text(
                f'Rating: 5.0\nFile Name: f{i}.jpg\n\nTags: description {i}\n',
                encoding='utf-8')

        cfg = OmegaConf.create({'main': {
            'memory_path': str(tmp_path),
            'cache_path': str(tmp_path / 'cache'),
        }})

        class CountingEmbedder:
            model_hash = 'testhash'

            def __init__(self):
                self.calls = 0

            def embed_long_text(self, long_text):
                self.calls += 1
                return [np.ones(8, dtype='float32')]

        first = CountingEmbedder()
        embeddings_a, scores_a = _gather_from_memory(cfg, first)
        assert first.calls == 5, first.calls

        second = CountingEmbedder()
        embeddings_b, scores_b = _gather_from_memory(cfg, second)
        assert second.calls == 0, (
            f'the second run re-embedded {second.calls} descriptions that had '
            f'not changed')

        assert scores_a == scores_b
        assert all(np.array_equal(a, b)
                   for a, b in zip(embeddings_a, embeddings_b))

    def test_a_different_model_does_not_reuse_the_old_vectors(self, tmp_path):
        """Vectors from one model are meaningless to another, so the model
        fingerprint has to be part of the key."""
        from omegaconf import OmegaConf
        import numpy as np

        day = tmp_path / '2026-08-18'
        day.mkdir()
        (day / f'{0:032x}.md').write_text(
            'Rating: 5.0\nFile Name: f.jpg\n\nTags: a description\n',
            encoding='utf-8')

        cfg = OmegaConf.create({'main': {
            'memory_path': str(tmp_path),
            'cache_path': str(tmp_path / 'cache'),
        }})

        class Embedder:
            def __init__(self, model_hash):
                self.model_hash = model_hash
                self.calls = 0

            def embed_long_text(self, long_text):
                self.calls += 1
                return [np.ones(8, dtype='float32')]

        first = Embedder('model-a')
        _gather_from_memory(cfg, first)
        second = Embedder('model-b')
        _gather_from_memory(cfg, second)
        assert second.calls == 1, (
            'a new model reused vectors produced by the old one')

    def test_the_filename_supplies_the_soft_hash(self, memory):
        """The parser does not return it; the file is named after it."""
        names = [f for f in os.listdir(memory / '2026-08-18') if f.endswith('.md')]
        assert any(len(os.path.splitext(n)[0]) == 9 for n in names)
