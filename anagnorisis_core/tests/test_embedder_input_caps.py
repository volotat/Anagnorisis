"""
Tests for the input caps both copies of the embedding tower must share.

The project runs the same model twice: a worker process on the GPU builds the
file embeddings that are indexed, and an in-process CPU copy embeds search
queries. The CPU copy is not text-only — hand the search box an image, a clip or
a song and it embeds *that file* to find things like it — so its vectors are
compared directly against the indexed ones. If one copy caps the pixels it feeds
the model and the other does not, the same image is resized two different ways
and "find things like this one" silently stops matching.

These tests pin the caps to both load paths without loading a model: the model
directory is faked, SentenceTransformer is recorded, and the two cap functions
are replaced with recorders.
"""
import os
import sys
import types

import pytest
from omegaconf import OmegaConf

import anagnorisis_core.models.embedder as embedder


class _FakeSentenceTransformer:
    """Stands in for the real class: only what the load paths touch."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.eval_called = False
        self.processor = None        # nothing for _cap_visual_pixels to mutate

    def eval(self):
        self.eval_called = True
        return self

    def encode_query(self, item, truncate_dim=None):
        return [0.0] * 8

    def encode_document(self, item, truncate_dim=None):
        return [0.0] * 8


class _Caps:
    """Records the calls both load paths make to the cap functions."""

    def __init__(self):
        self.audio = []
        self.visual = []

    def audio_decode(self, seconds=30.0):
        self.audio.append(seconds)

    def visual_pixels(self, model, max_side=512):
        self.visual.append((model, max_side))


@pytest.fixture()
def cfg(tmp_path):
    return OmegaConf.create({
        'embedder': {
            'model_name': 'test/model',
            'audio_seconds': 12.5,
            'video_frame_max_size': 384,
            'embedding_dimension': 8,
        },
        'main': {'embedding_models_path': str(tmp_path / 'models')},
    })


@pytest.fixture()
def fakes(monkeypatch, cfg):
    """Every external thing the load paths touch, replaced by a recorder."""
    local_path = os.path.join(cfg.main.embedding_models_path,
                              cfg.embedder.model_name.replace('/', '__'))
    os.makedirs(local_path, exist_ok=True)
    with open(os.path.join(local_path, 'config.json'), 'w') as f:
        f.write('{}')

    st_module = types.ModuleType('sentence_transformers')
    st_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, 'sentence_transformers', st_module)

    caps = _Caps()
    monkeypatch.setattr(embedder, '_cap_audio_decode', caps.audio_decode)
    monkeypatch.setattr(embedder, '_cap_visual_pixels', caps.visual_pixels)
    return caps


def test_query_tower_applies_the_configured_caps(cfg, fakes):
    """The CPU copy must cap what it feeds the model, like the worker does."""
    query = embedder.QueryEmbedder(cfg)

    assert query._ensure_loaded() is True

    assert fakes.audio == [cfg.embedder.audio_seconds]
    assert [max_side for _, max_side in fakes.visual] == [cfg.embedder.video_frame_max_size]
    assert fakes.visual[0][0] is query._model, 'capped something other than the loaded model'


def test_caps_fall_back_to_the_same_defaults(cfg, fakes, monkeypatch):
    """A config without the two keys must still cap, at the documented defaults.

    30 s and 512px are what the worker's own expressions fall back to
    (`audio_seconds`, `_visual_max_side()`); a different default here would put
    the towers back out of step on a config that omits them.
    """
    bare = OmegaConf.create({'embedder': {'model_name': 'test/model'},
                             'main': cfg.main})

    embedder._apply_input_caps(_FakeSentenceTransformer(), bare)

    assert fakes.audio == [30.0]
    assert [max_side for _, max_side in fakes.visual] == [512]


def test_reloading_does_not_reapply_caps(cfg, fakes):
    """A second call is a no-op, so the caps are applied once per load."""
    query = embedder.QueryEmbedder(cfg)
    query._ensure_loaded()
    query._ensure_loaded()

    assert len(fakes.visual) == 1
    assert len(fakes.audio) == 1


def test_caps_are_not_applied_when_the_model_is_absent(cfg, fakes):
    """No model downloaded: nothing is capped and nothing is claimed."""
    import shutil
    shutil.rmtree(os.path.join(cfg.main.embedding_models_path,
                               cfg.embedder.model_name.replace('/', '__')))
    query = embedder.QueryEmbedder(cfg)

    assert query._ensure_loaded() is False
    assert fakes.visual == []
    assert fakes.audio == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
