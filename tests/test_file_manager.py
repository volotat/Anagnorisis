"""
Tests for src/file_manager.py — resolve_subpath() and get_folder_structure()

resolve_subpath() is security-critical: it must prevent path traversal
attacks by raising PathTraversalError for any path that escapes base_dir.

Covers:
  - Valid sub-paths are returned as resolved Path objects inside base_dir
  - None / empty user_path returns base_dir itself
  - Simple '../' traversal raises PathTraversalError
  - Multi-hop traversal ('../../') raises PathTraversalError
  - URL-decoded traversal ('%2e%2e/') raises PathTraversalError
  - Double-encoded traversal ('%252e%252e/') raises PathTraversalError
  - Absolute path that escapes base_dir raises PathTraversalError
  - Symlink pointing outside base_dir raises PathTraversalError
  - get_folder_structure() returns None for non-existent directories
  - get_folder_structure() counts files correctly for known extensions
"""
import os
import pytest
from pathlib import Path
from urllib.parse import unquote

from src.file_manager import resolve_subpath, PathTraversalError, get_folder_structure


# ===========================================================================
# resolve_subpath
# ===========================================================================

class TestResolveSubpath:
    def test_valid_subpath_returns_path(self, tmp_path):
        sub = tmp_path / 'images' / 'photo.jpg'
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.touch()
        result = resolve_subpath(str(tmp_path), 'images/photo.jpg')
        assert result == sub.resolve()

    def test_none_user_path_returns_base(self, tmp_path):
        result = resolve_subpath(str(tmp_path), None)
        assert result == tmp_path.resolve()

    def test_empty_string_returns_base(self, tmp_path):
        result = resolve_subpath(str(tmp_path), '')
        assert result == tmp_path.resolve()

    def test_simple_traversal_blocked(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), '../secret')

    def test_multi_hop_traversal_blocked(self, tmp_path):
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), '../../etc/passwd')

    def test_url_decoded_traversal_blocked(self, tmp_path):
        # %2e%2e decodes to '..'  — resolve_subpath receives the decoded form
        decoded = unquote('%2e%2e/secret')  # '../secret'
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), decoded)

    def test_double_url_encoded_traversal_blocked(self, tmp_path):
        # %252e%252e → (first unquote) → %2e%2e → (second unquote) → '..'
        double_encoded = unquote(unquote('%252e%252e/secret'))
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), double_encoded)

    def test_absolute_path_escaping_base_blocked(self, tmp_path):
        # Passing an absolute path outside base_dir should be blocked
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), '/etc/passwd')

    def test_symlink_outside_base_blocked(self, tmp_path):
        # Create a symlink inside base that points outside
        outside = tmp_path.parent / 'outside_target'
        outside.mkdir(exist_ok=True)
        link = tmp_path / 'evil_link'
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("Symlinks not supported on this platform")
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), 'evil_link')

    def test_deeply_nested_valid_path(self, tmp_path):
        deep = tmp_path / 'a' / 'b' / 'c' / 'file.txt'
        deep.parent.mkdir(parents=True)
        deep.touch()
        result = resolve_subpath(str(tmp_path), 'a/b/c/file.txt')
        assert result == deep.resolve()

    def test_path_with_dot_components_inside_base(self, tmp_path):
        # 'images/./photo.jpg' is still inside base
        sub = tmp_path / 'images' / 'photo.jpg'
        sub.parent.mkdir(parents=True, exist_ok=True)
        sub.touch()
        result = resolve_subpath(str(tmp_path), 'images/./photo.jpg')
        assert result == sub.resolve()

    def test_traversal_to_sibling_directory_blocked(self, tmp_path):
        # Go up one level then into a *different* sibling directory — this
        # escapes base_dir and must be blocked.
        # Note: '../<same_dir_name>/file' is NOT tested here because resolving
        # it yields a path inside tmp_path (the OS collapses the round-trip),
        # so resolve_subpath correctly allows it.
        sibling = tmp_path.parent / 'sibling_dir'
        sibling.mkdir(exist_ok=True)
        with pytest.raises(PathTraversalError):
            resolve_subpath(str(tmp_path), f'../sibling_dir/secret.txt')


# ===========================================================================
# get_folder_structure
# ===========================================================================

class TestGetFolderStructure:
    def test_nonexistent_directory_returns_none(self, tmp_path):
        result = get_folder_structure(str(tmp_path / 'does_not_exist'), media_extensions=['.jpg'])
        assert result is None

    def test_empty_directory_zero_files(self, tmp_path):
        result = get_folder_structure(str(tmp_path), media_extensions=['.jpg'])
        assert result is not None
        assert result['num_files'] == 0
        assert result['total_files'] == 0

    def test_counts_files_with_matching_extension(self, tmp_path):
        (tmp_path / 'a.jpg').touch()
        (tmp_path / 'b.jpg').touch()
        (tmp_path / 'c.txt').touch()  # should not be counted
        result = get_folder_structure(str(tmp_path), media_extensions=['.jpg'])
        assert result['num_files'] == 2

    def test_ignores_non_matching_extensions(self, tmp_path):
        (tmp_path / 'track.mp3').touch()
        result = get_folder_structure(str(tmp_path), media_extensions=['.jpg'])
        assert result['num_files'] == 0

    def test_counts_subdirectory_files_in_total(self, tmp_path):
        sub = tmp_path / 'sub'
        sub.mkdir()
        (tmp_path / 'root.jpg').touch()
        (sub / 'nested.jpg').touch()
        result = get_folder_structure(str(tmp_path), media_extensions=['.jpg'])
        assert result['total_files'] == 2
        assert result['num_files'] == 1

    def test_structure_contains_subfolder_key(self, tmp_path):
        sub = tmp_path / 'animals'
        sub.mkdir()
        (sub / 'cat.jpg').touch()
        result = get_folder_structure(str(tmp_path), media_extensions=['.jpg'])
        assert 'animals' in result['subfolders']
        assert result['subfolders']['animals']['num_files'] == 1


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
