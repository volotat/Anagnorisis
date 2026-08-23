"""Tag matrices are several megabytes each and a new one is written whenever a
vocabulary is edited or the embedding model changes. Nothing ever read the old
ones again, and nothing removed them, so a cache directory accumulated one file
per change forever.

These tests pin what gets deleted and — more importantly — what does not.
"""
import os

import numpy as np
import pytest

from anagnorisis_core.search.proxy import EmbeddingProxyGenerator


class FakeSource:
    """Just enough of the proxy's source protocol to name and embed."""

    def __init__(self, name='images', model_hash='aaaaaaaaaaaa'):
        self.name = name
        self.model_hash = model_hash

    def embedding_key(self, file_path):
        return 'k'

    def get_embedding(self, file_path):
        return None


def _proxy(tmp_path, tags, name='images', model_hash='aaaaaaaaaaaa'):
    return EmbeddingProxyGenerator(
        source=FakeSource(name, model_hash),
        tags=tags,
        threshold=0.2,
        cache_path=str(tmp_path),
    )


def _embed(tags):
    """Deterministic unit vectors, one row per tag."""
    return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(tags), 1))


def _tag_files(tmp_path):
    return sorted(f for f in os.listdir(tmp_path) if f.startswith('tag_embeddings_'))


class TestSupersededFilesAreRemoved:
    def test_editing_the_vocabulary_removes_the_previous_matrix(self, tmp_path):
        _proxy(tmp_path, ['cat', 'dog'])._compute_tag_embeddings_batched(_embed)
        first = _tag_files(tmp_path)
        assert len(first) == 1, first

        # a third tag: new vocabulary hash, so a new file
        _proxy(tmp_path, ['cat', 'dog', 'bird'])._compute_tag_embeddings_batched(_embed)

        after = _tag_files(tmp_path)
        assert len(after) == 1, f'the old vocabulary was left behind: {after}'
        assert after != first

    def test_changing_the_model_removes_the_previous_matrix(self, tmp_path):
        tags = ['cat', 'dog']
        _proxy(tmp_path, tags, model_hash='aaaaaaaaaaaa')._compute_tag_embeddings_batched(_embed)
        _proxy(tmp_path, tags, model_hash='bbbbbbbbbbbb')._compute_tag_embeddings_batched(_embed)

        after = _tag_files(tmp_path)
        assert len(after) == 1, after
        assert 'bbbbbbbbbbbb' in after[0]

    def test_the_old_nameless_files_are_cleared_out(self, tmp_path):
        """Files from before the media type was part of the name cannot be read
        any more, so keeping them is pure cost."""
        legacy = tmp_path / 'tag_embeddings_0123456789ab_aaaaaaaaaaaa.pt'
        legacy.write_bytes(b'stale')

        _proxy(tmp_path, ['cat'])._compute_tag_embeddings_batched(_embed)

        assert not legacy.exists(), 'the legacy file survived'
        assert len(_tag_files(tmp_path)) == 1


class TestWhatMustSurvive:
    def test_another_media_types_matrix_is_left_alone(self, tmp_path):
        """The prune runs per media type. Deleting audio's current file while
        writing images' would make every run recompute the other's vocabulary."""
        _proxy(tmp_path, ['guitar'], name='audio')._compute_tag_embeddings_batched(_embed)
        _proxy(tmp_path, ['cat'], name='images')._compute_tag_embeddings_batched(_embed)

        after = _tag_files(tmp_path)
        assert len(after) == 2, after
        assert any('audio' in f for f in after) and any('images' in f for f in after)

    def test_the_file_just_written_is_never_the_one_deleted(self, tmp_path):
        p = _proxy(tmp_path, ['cat', 'dog'])
        assert p._compute_tag_embeddings_batched(_embed) is True
        path = p._tag_embs_path_for('aaaaaaaaaaaa')
        assert os.path.exists(path), 'it deleted what it had just saved'

    def test_unrelated_files_in_the_cache_are_untouched(self, tmp_path):
        other = tmp_path / 'content'
        other.mkdir()
        (other / 'shard.pkl').write_bytes(b'x')
        keep = tmp_path / 'model_identity.json'
        keep.write_text('{}')

        _proxy(tmp_path, ['cat'])._compute_tag_embeddings_batched(_embed)

        assert keep.exists() and (other / 'shard.pkl').exists()


class TestNaming:
    def test_the_media_type_leads_the_filename(self, tmp_path):
        p = _proxy(tmp_path, ['cat'], name='images')
        assert os.path.basename(p._tag_embs_path_for('aaaaaaaaaaaa')).startswith(
            'tag_embeddings_images_')

    def test_a_name_with_separators_cannot_break_the_parsing(self, tmp_path):
        """The filename is split on '_' to recognise the old format, so a media
        type called 'my_type' must not smuggle an extra separator in."""
        p = _proxy(tmp_path, ['cat'], name='my_odd type')
        base = os.path.basename(p._tag_embs_path_for('aaaaaaaaaaaa'))
        assert base[:-len('.pt')].count('_') == 4, base
