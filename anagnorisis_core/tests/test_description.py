"""
Tests for anagnorisis_core/description.py — the one description of a file.

The invariant worth defending: the text metadata search embeds, the text the UI
shows as "full search description", and the body of the memory file are the same
string. When those drifted apart, the displayed description stopped matching the
indexed one and search results looked random for weeks.

No models here. The auto-description is injected as a callable, which is exactly
why it can be tested without a GPU.
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


class TestSections:
    def test_name_and_path_lead(self, cfg, a_text_file):
        text = description.build_description(a_text_file, cfg=cfg)
        lines = text.split('\n')
        assert lines[0] == f'File Name: {os.path.basename(a_text_file)}'
        assert lines[1] == f'File Path: {a_text_file}'

    def test_a_short_text_file_is_kept_verbatim(self, cfg, a_text_file):
        """Its own words are a better description than any summary of them, and
        keeping them costs nothing — so the injected describer is not consulted."""
        text = description.build_description(
            a_text_file, cfg=cfg, describe=lambda fp: 'A sleeping cat.')
        assert '# Content:' in text
        assert 'A sleeping cat.' not in text, (
            'a model was consulted for text short enough to use as it stands')

    def test_a_long_text_file_uses_the_injected_describer(self, cfg, tmp_path):
        big = tmp_path / 'big.txt'
        big.write_text('w' * (description.text_verbatim_limit(cfg) + 1),
                       encoding='utf-8')
        text = description.build_description(
            str(big), cfg=cfg, describe=lambda fp: 'A sleeping cat.')
        assert '# Automatic description:\nA sleeping cat.' in text

    def test_a_non_text_file_always_uses_the_describer(self, cfg, tmp_path):
        """The rule is about text; an image has no words to keep."""
        img = tmp_path / 'photo.jpg'
        img.write_bytes(b'\xff\xd8\xff')
        text = description.build_description(
            str(img), cfg=cfg, describe=lambda fp: 'A sleeping cat.')
        assert '# Automatic description:\nA sleeping cat.' in text

    def test_no_describer_means_no_description_section(self, cfg, a_text_file):
        text = description.build_description(a_text_file, cfg=cfg)
        assert '# Automatic description:' not in text

    def test_internal_metadata_is_included(self, cfg, a_text_file):
        text = description.build_description(a_text_file, cfg=cfg)
        assert '# Internal metadata:' in text
        assert 'file_size:' in text

    def test_meta_sidecar_is_included(self, cfg, a_text_file):
        with open(a_text_file + '.meta', 'w', encoding='utf-8') as fh:
            fh.write('This is my grandfather, photographed in 1963.\n')
        text = description.build_description(a_text_file, cfg=cfg)
        assert "# External metadata from 'notes.txt.meta' file:" in text
        assert 'my grandfather' in text

    def test_a_remote_file_gets_the_note_and_nothing_that_needs_reading(self, cfg):
        text = description.build_description(
            'webdav://198.51.100.7:6001/movies/clip.mp4', cfg=cfg,
            describe=lambda fp: 'should never be called')
        assert '# Remote file note:' in text
        # Every other section would mean downloading it.
        assert '# Automatic description:' not in text
        assert '# Internal metadata:' not in text
        assert 'should never be called' not in text


class TestDataServerVariant:
    """The annotator's two omissions are forced, not stylistic."""

    def test_path_can_be_withheld(self, cfg, a_text_file):
        text = description.build_description(a_text_file, cfg=cfg, include_path=False)
        assert text.startswith('File Name: notes.txt\n')
        assert 'File Path:' not in text, 'a published description must not leak the layout'

    def test_sidecar_can_be_withheld(self, cfg, a_text_file):
        """Writing the sidecar must not fold the previous run back into itself."""
        with open(a_text_file + '.meta', 'w', encoding='utf-8') as fh:
            fh.write('previous output\n')
        text = description.build_description(
            a_text_file, cfg=cfg, include_meta_sidecar=False)
        assert 'previous output' not in text


class TestSidecarReader:
    def test_missing_sidecar_is_not_an_error(self, tmp_path):
        text, truncated = description.read_meta_snippet(str(tmp_path / 'nope.meta'))
        assert (text, truncated) == ('', False)

    def test_long_sidecar_is_truncated(self, tmp_path):
        p = tmp_path / 'big.meta'
        p.write_text('x' * 80 + '\n' * 1, encoding='utf-8')
        with open(p, 'w', encoding='utf-8') as fh:
            for _ in range(description.MAX_META_LINES + 50):
                fh.write('a line of sidecar text\n')
        text, truncated = description.read_meta_snippet(str(p))
        assert truncated
        assert text.count('\n') <= description.MAX_META_LINES

    def test_char_cap_applies_even_with_few_lines(self, tmp_path):
        p = tmp_path / 'wide.meta'
        p.write_text('y' * (description.MAX_META_CHARS + 5_000) + '\n', encoding='utf-8')
        text, truncated = description.read_meta_snippet(str(p))
        assert truncated and len(text) <= description.MAX_META_CHARS
