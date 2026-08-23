"""
The application's re-exports still resolve.

Several things moved into anagnorisis_core while `src` kept importing them under
their old names, so existing callers and module code did not all have to change at
once. Those re-exports are an application-side promise, not an engine one, which
is why this test lives here — and it is worth having, because deleting a
"redundant" import line is exactly the kind of tidying that breaks nine module
files at startup.
"""
import os

import pytest


def test_file_manager_still_exposes_path_helpers():
    """Path handling moved to anagnorisis_core.storage.file_paths."""
    from src import file_manager
    for name in ('resolve_subpath', 'PathTraversalError',
                 'get_folder_structure', 'open_file_in_folder'):
        assert hasattr(file_manager, name), f'src.file_manager lost {name}'


def test_utils_still_exposes_progress_adapters():
    """The progress adapters moved to anagnorisis_core.progress."""
    from src import utils
    for name in ('SortingProgressCallback', 'EmbeddingGatheringCallback',
                 'ArbitraryProgressCallback'):
        assert hasattr(utils, name), f'src.utils lost {name}'


def test_utils_kept_its_own_display_helpers():
    """Formatting is presentation and stayed in the application."""
    from src import utils
    for name in ('convert_size', 'convert_length', 'time_difference'):
        assert hasattr(utils, name), f'src.utils lost {name}'


def test_event_manager_hashes_identically_to_the_core():
    """Two definitions of content identity would orphan every memory file."""
    from anagnorisis_core.storage.soft_hash import get_file_soft_hash, SOFT_HASH_ALGORITHM
    from src.app_factory.event_manager import EventManager

    assert EventManager.soft_hash_algorithm == SOFT_HASH_ALGORITHM

    # An absolute path: the hash reads the file, so a relative one depends on
    # whichever directory pytest happened to start in.
    target = os.path.abspath(__file__)
    assert EventManager.get_file_soft_hash(target) == get_file_soft_hash(target)
