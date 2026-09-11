"""
Tests for the music module's filter_by_length.

It returned a dict keyed by path while FileManager expects a list indexed by
position, so sorting by length raised KeyError: 0 on every invocation.
"""
import pytest


class TestFilterByLengthShape:

    def test_returns_a_list_the_same_length_as_input(self):
        """The one-line fix: convert to positional alignment."""
        from modules.music.serve import _mock_filter_by_length

        all_files = ['a.mp3', 'b.mp3', 'c.mp3']
        result = _mock_filter_by_length(all_files)

        assert isinstance(result, list), f'expected list, got {type(result)}'
        assert len(result) == len(all_files)

    def test_none_for_files_without_metadata(self):
        from modules.music.serve import _mock_filter_by_length

        result = _mock_filter_by_length(['a.mp3', 'b.txt', 'c.mp3'])
        # b.txt would have no duration metadata → None in that slot
        assert result[1] is None

    def test_can_be_subscripted_by_index(self):
        """This is the exact failure mode: dict[0] raises KeyError."""
        from modules.music.serve import _mock_filter_by_length

        result = _mock_filter_by_length(['a.mp3'])
        # Must not raise
        assert result[0] == 120.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])