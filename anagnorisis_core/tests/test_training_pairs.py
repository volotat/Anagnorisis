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
        cfg = OmegaConf.create({'main': {'memory_path': str(memory)}})

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

    def test_the_filename_supplies_the_soft_hash(self, memory):
        """The parser does not return it; the file is named after it."""
        names = [f for f in os.listdir(memory / '2026-08-18') if f.endswith('.md')]
        assert any(len(os.path.splitext(n)[0]) == 9 for n in names)
