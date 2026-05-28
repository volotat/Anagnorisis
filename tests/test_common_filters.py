"""
Tests for src/common_filters.py

Covers:
  - _normalize_text(): accent stripping, separator normalisation, case folding,
                       whitespace collapse
  - filter_by_text(mode='file-name'): fuzzy filename scoring, priority boosting
    for exact matches, empty query, unicode filenames

Note: semantic-content and semantic-metadata modes require live ML models and
are tested separately via Docker (Tier 2 tests).
"""
import numpy as np
import pytest

# Import the module-level helpers we want to test
from src.common_filters import _normalize_text, CommonFilters


# ===========================================================================
# _normalize_text
# ===========================================================================

class TestNormalizeText:
    def test_lowercase(self):
        assert _normalize_text('Hello World') == 'hello world'

    def test_accent_stripping(self):
        assert _normalize_text('résumé') == 'resume'
        assert _normalize_text('naïve') == 'naive'
        assert _normalize_text('über') == 'uber'

    def test_separator_to_space(self):
        assert _normalize_text('hello_world') == 'hello world'
        assert _normalize_text('hello-world') == 'hello world'
        assert _normalize_text('hello.world') == 'hello world'
        assert _normalize_text('hello/world') == 'hello world'
        assert _normalize_text('hello\\world') == 'hello world'

    def test_multiple_separators_collapse(self):
        assert _normalize_text('a___b') == 'a b'
        assert _normalize_text('a - b') == 'a b'

    def test_whitespace_collapse(self):
        assert _normalize_text('  lots   of   spaces  ') == 'lots of spaces'

    def test_empty_string(self):
        assert _normalize_text('') == ''

    def test_combined(self):
        # Real-world music filename pattern
        result = _normalize_text('Björk - Jóga (Remaster).flac')
        assert 'bjork' in result
        assert 'joga' in result
        assert 'remaster' in result


# ===========================================================================
# filter_by_text(mode='file-name') via a lightweight stub of CommonFilters
# ===========================================================================

class _MinimalEngine:
    """Stub engine — file-name mode does not call any engine methods."""
    pass

class _MinimalSocketEvents:
    def show_search_status(self, *args, **kwargs):
        pass

def _make_filters(media_directory='/media'):
    return CommonFilters(
        engine=_MinimalEngine(),
        metadata_engine=None,
        common_socket_events=_MinimalSocketEvents(),
        media_directory=media_directory,
        db_schema=None,
    )


class TestFilterByTextFileName:
    def _scores(self, files, query):
        cf = _make_filters()
        return cf.filter_by_text(files, query, mode='file-name')

    def test_returns_one_score_per_file(self):
        files = ['/media/cat.jpg', '/media/dog.jpg', '/media/flower.jpg']
        scores = self._scores(files, 'cat')
        assert len(scores) == len(files)

    def test_exact_match_highest_score(self):
        files = ['/media/vacation_photo.jpg', '/media/cat.jpg', '/media/cats_in_garden.jpg']
        scores = self._scores(files, 'cat.jpg')
        # 'cat.jpg' should outrank others
        idx_exact = files.index('/media/cat.jpg')
        assert scores[idx_exact] == max(scores), (
            f"Exact match should have max score. Scores: {list(zip(files, scores))}"
        )

    def test_fuzzy_match_scores_above_zero(self):
        files = ['/media/photograph_of_a_cat.jpg']
        scores = self._scores(files, 'cat')
        assert scores[0] > 0

    def test_empty_query_returns_scores(self):
        # Should not crash; scores may all be equal or low
        files = ['/media/a.jpg', '/media/b.jpg']
        scores = self._scores(files, '')
        assert len(scores) == len(files)
        assert all(np.isfinite(s) for s in scores)

    def test_scores_are_finite(self):
        files = ['/media/image1.jpg', '/media/image2.jpg']
        scores = self._scores(files, 'image')
        assert all(np.isfinite(s) for s in scores)

    def test_unicode_filename_no_crash(self):
        files = ['/media/фото_кот.jpg', '/media/Ünïcödé_fïlé.jpg']
        scores = self._scores(files, 'кот')
        assert len(scores) == len(files)
        assert all(np.isfinite(s) for s in scores)

    def test_accent_in_query_matches_normalised_filename(self):
        # Query 'resume' should match 'résumé.txt' after normalisation
        files = ['/media/résumé.txt', '/media/other.txt']
        scores = self._scores(files, 'resume')
        assert scores[0] > scores[1], (
            "Accent-stripped filename should match accent-free query better"
        )

    def test_path_substring_boost(self):
        # A file whose path contains the query exactly should outscore one that
        # only fuzzy-matches.
        files = ['/media/cats/my_image.jpg', '/media/dogs/my_image.jpg']
        scores = self._scores(files, 'cats')
        assert scores[0] > scores[1], (
            f"Path with exact segment 'cats' should outscore 'dogs'. Scores: {scores}"
        )

    def test_single_file_no_crash(self):
        scores = self._scores(['/media/solo.jpg'], 'solo')
        assert len(scores) == 1
        assert np.isfinite(scores[0])

    def test_unknown_mode_raises(self):
        cf = _make_filters()
        with pytest.raises(ValueError, match='Unknown mode'):
            cf.filter_by_text(['/media/a.jpg'], 'query', mode='nonexistent')


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
