"""
The description the application shows must be the description the engine indexed.

This lives outside anagnorisis_core deliberately: it is the *seam* between the
engine and the Flask application, and it is the only test that reaches across it.
The engine's own section-by-section tests are in
anagnorisis_core/tests/test_description.py.

The invariant: the text metadata search embeds, the text shown as "full search
description", and the body of a memory file are the same string. When those
drifted apart, the displayed description stopped matching the indexed one and
search results looked random for weeks.
"""
import os
import pytest
from omegaconf import OmegaConf

from anagnorisis_core.media import description
from anagnorisis_core.media.media_types import bundled_media_types_dir


@pytest.fixture
def cfg(tmp_path):
    """Minimal config: the contract the core actually reads."""
    mt = bundled_media_types_dir()
    conf = OmegaConf.load(os.path.join(mt, 'media_types.yaml'))
    conf = OmegaConf.merge(conf, OmegaConf.create({
        'main': {
            'cache_path': str(tmp_path / 'cache'),
            'embedding_models_path': str(tmp_path / 'models'),
            'media_types_path': mt,
        },
        'embedder': {'model_name': 'test/model', 'embedding_dimension': 1024},
    }))
    os.makedirs(conf.main.cache_path, exist_ok=True)
    return conf


@pytest.fixture
def a_text_file(tmp_path):
    p = tmp_path / 'notes.txt'
    p.write_text('A note about a cat asleep on a mat.', encoding='utf-8')
    return str(p)


class TestOneTextForEveryConsumer:
    """The invariant: search, the UI and the memory body are the same string."""

    def test_memory_body_equals_the_search_text(self, cfg, a_text_file, monkeypatch):
        from src.memory_system import MemorySystem

        MemorySystem._instance = None
        ms = MemorySystem.__new__(MemorySystem)          # skip model loading
        ms.cfg = cfg
        from anagnorisis_core.media.media_types import get_registry
        ms.types = get_registry(cfg)
        monkeypatch.setattr(ms, '_ensure_initialized', lambda: None)
        monkeypatch.setattr(ms, '_describe_now', lambda fp, mt: 'A sleeping cat.')

        memory_text = ms.build_memory_text(a_text_file, soft_hash='abc123', rating=8.0)
        search_text = description.build_description(
            a_text_file, cfg=cfg, describe=lambda fp: 'A sleeping cat.')

        rating_line, _, body = memory_text.partition('\n')
        assert rating_line == 'Rating: 8.0'
        assert body == search_text, (
            'the memory body and the search text have diverged; the evaluator '
            'would train on one shape and score another'
        )

    def test_the_rating_is_only_on_line_one(self, cfg, a_text_file, monkeypatch):
        """Training strips line 1. A rating anywhere else leaks the answer."""
        from src.memory_system import MemorySystem
        ms = MemorySystem.__new__(MemorySystem)
        ms.cfg = cfg
        from anagnorisis_core.media.media_types import get_registry
        ms.types = get_registry(cfg)
        monkeypatch.setattr(ms, '_ensure_initialized', lambda: None)
        monkeypatch.setattr(ms, '_describe_now', lambda fp, mt: 'A sleeping cat.')

        body = ms.build_memory_text(a_text_file, 'abc123', 8.0).partition('\n')[2]
        assert 'Rating:' not in body
