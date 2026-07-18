"""
Tests for src/caching.py

Covers:
  - RAMCache: basic get/set, TTL expiration, thread-safety
  - DiskCache: write/read round-trip, TTL expiration, corrupted shard recovery,
               atomic write (no partial files), warming callback
  - TwoLevelCache: RAM-first lookup, disk fallback, write-through to both tiers,
                   write-back coalescing (pending entries readable before flush)
"""
import os
import pickle
import tempfile
import threading
import time
import pytest
from src.caching import RAMCache, DiskCache, TwoLevelCache


# ===========================================================================
# RAMCache
# ===========================================================================

class TestRAMCache:
    def test_set_and_get(self):
        cache = RAMCache(ttl_seconds=60)
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

    def test_missing_key_returns_none(self):
        cache = RAMCache(ttl_seconds=60)
        assert cache.get('nonexistent') is None

    def test_ttl_expiry(self):
        cache = RAMCache(ttl_seconds=1)
        cache.set('expiring', 'soon')
        assert cache.get('expiring') == 'soon'
        time.sleep(1.1)
        assert cache.get('expiring') is None

    def test_overwrite_key(self):
        cache = RAMCache(ttl_seconds=60)
        cache.set('k', 'v1')
        cache.set('k', 'v2')
        assert cache.get('k') == 'v2'

    def test_thread_safety(self):
        cache = RAMCache(ttl_seconds=60)
        errors = []

        def writer(i):
            try:
                cache.set(f'k{i}', i)
            except Exception as e:
                errors.append(e)

        def reader(i):
            try:
                cache.get(f'k{i}')
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(50)]
        threads += [threading.Thread(target=reader, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread safety errors: {errors}"

    def test_stores_various_types(self):
        cache = RAMCache(ttl_seconds=60)
        cache.set('list', [1, 2, 3])
        cache.set('dict', {'a': 1})
        cache.set('none_val', None)  # None value is allowed to store
        assert cache.get('list') == [1, 2, 3]
        assert cache.get('dict') == {'a': 1}
        # None is indistinguishable from missing — document that behaviour
        # (get() returns None for both missing and stored-None)


# ===========================================================================
# DiskCache
# ===========================================================================

class TestDiskCache:
    def _make_cache(self, tmpdir, **kwargs):
        return DiskCache(
            cache_dir=str(tmpdir),
            ttl_seconds=kwargs.pop('ttl_seconds', 3600),
            write_back=kwargs.pop('write_back', False),  # immediate writes for tests
            **kwargs,
        )

    def test_set_and_get(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set('hello', 'world')
        assert cache.get('hello') == 'world'

    def test_missing_key_returns_none(self, tmp_path):
        cache = self._make_cache(tmp_path)
        assert cache.get('no_such_key') is None

    def test_ttl_expiry(self, tmp_path):
        cache = self._make_cache(tmp_path, ttl_seconds=1)
        cache.set('expiring', 42)
        assert cache.get('expiring') == 42
        time.sleep(1.1)
        assert cache.get('expiring') is None

    def test_persists_across_instances(self, tmp_path):
        c1 = self._make_cache(tmp_path)
        c1.set('persistent', 'data')
        # Create a fresh instance pointing at the same directory
        c2 = self._make_cache(tmp_path)
        assert c2.get('persistent') == 'data'

    def test_corrupted_shard_recovers_gracefully(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set('before_corrupt', 'ok')

        # Corrupt the shard file for that key
        shard = DiskCache._shard_for_key('before_corrupt')
        shard_path = os.path.join(str(tmp_path), f"{shard:02x}.pkl")
        with open(shard_path, 'wb') as f:
            f.write(b'THIS IS NOT VALID PICKLE DATA!!!')

        # Should not raise; missing data is acceptable after corruption
        result = cache.get('before_corrupt')
        assert result is None  # shard reset to empty

    def test_no_partial_files_on_write(self, tmp_path):
        """Atomic writes: no .pkl.tmp or similar leftover files after set()."""
        cache = self._make_cache(tmp_path)
        cache.set('atomic', 'write')
        leftovers = [f for f in os.listdir(str(tmp_path)) if not f.endswith('.pkl')]
        assert not leftovers, f"Unexpected leftover files: {leftovers}"

    def test_warming_callback_called(self, tmp_path):
        warmed = {}

        def on_shard_loaded(mapping):
            warmed.update(mapping)

        c1 = self._make_cache(tmp_path)
        c1.set('w1', 'val1')
        c1.set('w2', 'val2')

        c2 = DiskCache(
            cache_dir=str(tmp_path),
            ttl_seconds=3600,
            write_back=False,
            on_shard_loaded=on_shard_loaded,
            warm_interval_seconds=0,  # always warm
        )
        c2.get('w1')
        assert 'w1' in warmed

    def test_write_back_readable_before_flush(self, tmp_path):
        """With write_back=True, a set() value is readable immediately from pending."""
        cache = DiskCache(
            cache_dir=str(tmp_path),
            ttl_seconds=3600,
            write_back=True,
        )
        cache.set('wb_key', 'wb_value')
        assert cache.get('wb_key') == 'wb_value'
        cache.close()

    def test_overwrite_existing_key(self, tmp_path):
        cache = self._make_cache(tmp_path)
        cache.set('k', 'v1')
        cache.set('k', 'v2')
        assert cache.get('k') == 'v2'

    def test_disk_cache_shard_cache_survives_corrupted_disk(self, tmp_path):
        """After a shard is read once, subsequent get() must NOT touch disk.
        Demonstrated by corrupting the on-disk file mid-run and checking that
        cached entries are still served correctly."""
        cache = DiskCache(
            cache_dir=str(tmp_path),
            ttl_seconds=3600,
            write_back=False,
        )
        cache.set('k1', 'v1')
        cache.set('k2', 'v2')

        # First read populates the in-memory shard cache.
        assert cache.get('k1') == 'v1'
        shard = DiskCache._shard_for_key('k1')
        assert shard in cache._shard_data, (
            "Expected shard to be cached in memory after first read"
        )

        # Corrupt the disk file. If get() re-reads from disk it would either
        # raise (pickle error) or return None — both would fail the test.
        shard_path = os.path.join(str(tmp_path), f"{shard:02x}.pkl")
        with open(shard_path, 'wb') as f:
            f.write(b'this is not valid pickle data')

        # Subsequent reads must still work because they hit the in-memory cache.
        assert cache.get('k1') == 'v1'
        assert cache.get('k2') == 'v2'


# ===========================================================================
# TwoLevelCache
# ===========================================================================

class TestTwoLevelCache:
    def test_set_and_get_from_ram(self, tmp_path):
        cache = TwoLevelCache(cache_dir=str(tmp_path))
        cache.set('r', 'ram_val')
        assert cache.get('r') == 'ram_val'

    def test_disk_fallback_after_ram_miss(self, tmp_path):
        cache1 = TwoLevelCache(cache_dir=str(tmp_path))
        cache1.set('d', 'disk_val')
        cache1.disk.flush()  # ensure written to disk

        # New instance: cold RAM, but disk has the value
        cache2 = TwoLevelCache(cache_dir=str(tmp_path), ram_ttl_seconds=3600)
        assert cache2.get('d') == 'disk_val'

    def test_write_through_both_tiers(self, tmp_path):
        cache = TwoLevelCache(cache_dir=str(tmp_path))
        cache.set('both', 'value')
        cache.disk.flush()

        assert cache.ram.get('both') == 'value'
        # Verify disk independently with a fresh DiskCache
        disk = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600, write_back=False)
        assert disk.get('both') == 'value'

    def test_save_to_disk_false_skips_disk(self, tmp_path):
        cache = TwoLevelCache(cache_dir=str(tmp_path))
        cache.set('ram_only', 'ephemeral', save_to_disk=False)
        assert cache.get('ram_only') == 'ephemeral'

        # Fresh instance — disk never written, so miss expected
        cache2 = TwoLevelCache(cache_dir=str(tmp_path))
        assert cache2.get('ram_only') is None

    def test_missing_key_returns_none(self, tmp_path):
        cache = TwoLevelCache(cache_dir=str(tmp_path))
        assert cache.get('absent') is None


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])
