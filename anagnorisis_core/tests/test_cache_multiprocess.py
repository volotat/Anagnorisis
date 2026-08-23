"""
Two processes sharing one cache directory.

The application and the `anagnorisis` command can both be pointed at the same
cache. Shard writes were already atomic, so corruption was never the risk — the
risk was a silent read-modify-write race: both processes read the same shard, each
merges its own additions, and whichever writes last erases the other's entries.

Lost cache entries only ever cost time, because the data recomputes. But they are
invisible, so they are exactly the kind of bug worth a test.
"""
import multiprocessing
import os
import pytest

from anagnorisis_core.storage.caching import DiskCache


def _writer(cache_dir, prefix, count):
    """Write `count` entries from a separate process and flush them."""
    cache = DiskCache(cache_dir=cache_dir, ttl_seconds=3600)
    for i in range(count):
        cache.set(f'{prefix}-{i}', f'value-{prefix}-{i}')
    cache.close()


class TestConcurrentWriters:
    def test_neither_process_loses_its_entries(self, tmp_path):
        """The race this guards: last writer wins the whole shard."""
        cache_dir = str(tmp_path / 'shared')
        n = 60

        ctx = multiprocessing.get_context('spawn')
        procs = [ctx.Process(target=_writer, args=(cache_dir, name, n))
                 for name in ('alpha', 'beta')]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0, f'writer failed with {p.exitcode}'

        reader = DiskCache(cache_dir=cache_dir, ttl_seconds=3600)
        missing = [k for name in ('alpha', 'beta') for k in
                   (f'{name}-{i}' for i in range(n))
                   if reader.get(k) is None]
        assert not missing, (
            f'{len(missing)} of {2 * n} entries were lost to a concurrent write, '
            f'e.g. {missing[:3]}'
        )

    def test_a_single_process_still_round_trips(self, tmp_path):
        """Guard the guard: if writing is broken outright the test above is moot."""
        cache_dir = str(tmp_path / 'solo')
        cache = DiskCache(cache_dir=cache_dir, ttl_seconds=3600)
        cache.set('k', 'v')
        cache.close()
        assert DiskCache(cache_dir=cache_dir, ttl_seconds=3600).get('k') == 'v'

    def test_the_lock_file_does_not_become_a_cache_entry(self, tmp_path):
        """A stray .lock must not be mistaken for a shard on the next read."""
        cache_dir = str(tmp_path / 'locks')
        cache = DiskCache(cache_dir=cache_dir, ttl_seconds=3600)
        cache.set('k', 'v')
        cache.close()
        fresh = DiskCache(cache_dir=cache_dir, ttl_seconds=3600)
        assert fresh.get('k') == 'v'
        assert any(f.endswith('.lock') for f in os.listdir(cache_dir)), \
            'expected a lock file to exist alongside the shard'
