"""
Tests for anagnorisis_core/caching.py

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
from anagnorisis_core.storage.caching import RAMCache, DiskCache, TwoLevelCache


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

class TestRAMCacheIsBounded:
    """The tier used to grow without limit, which is what killed the container.

    Entries only expired when the same key was read again, so anything never
    read again stayed for the life of the process.
    """

    def test_evicts_least_recently_used_when_over_budget(self):
        cache = RAMCache(ttl_seconds=3600, max_bytes=4000)
        for i in range(20):
            cache.set(f'k{i}', 'x' * 500)

        assert cache._total_bytes <= 4000, cache.stats()
        assert cache._evictions > 0
        # The newest survive, the oldest are gone.
        assert cache.get('k19') is not None
        assert cache.get('k0') is None

    def test_reading_a_key_protects_it_from_eviction(self):
        """Least *recently used*, not least recently written."""
        # Size the budget from what an entry actually costs, so the test is
        # about eviction order rather than about guessing the accounting.
        probe = RAMCache(ttl_seconds=3600)
        probe.set('k0', 'x' * 500)
        per_entry = probe._total_bytes

        cache = RAMCache(ttl_seconds=3600, max_bytes=per_entry * 6)
        for i in range(6):
            cache.set(f'k{i}', 'x' * 500)
        assert cache._evictions == 0, 'the budget should fit six entries'

        cache.get('k0')                 # k0 becomes the most recently used
        cache.set('k6', 'x' * 500)      # one over: something must go

        assert cache.get('k0') is not None, 'the key just read was evicted'
        assert cache.get('k1') is None, 'the least recently used key survived'

    def test_a_value_larger_than_the_budget_is_still_served(self):
        """Better to exceed the budget briefly than to store nothing at all."""
        cache = RAMCache(ttl_seconds=3600, max_bytes=100)
        cache.set('big', 'x' * 10_000)
        assert cache.get('big') is not None

    def test_unbounded_when_no_budget_is_given(self):
        cache = RAMCache(ttl_seconds=3600)
        for i in range(500):
            cache.set(f'k{i}', 'x' * 100)
        assert len(cache._data) == 500
        assert cache._evictions == 0


class TestBudgetSizing:
    def test_budget_comes_from_the_process_limit_not_the_machine(self):
        """In a container the cache must size itself for the cgroup limit, or
        it happily fills 30 GB while being killed at 16."""
        from anagnorisis_core.storage.caching import (
            memory_limit_bytes, default_ram_budget_bytes,
            MIN_RAM_BUDGET_BYTES, MAX_RAM_BUDGET_BYTES)

        limit = memory_limit_bytes()
        assert limit > 0
        budget = default_ram_budget_bytes(1.0)
        assert MIN_RAM_BUDGET_BYTES <= budget <= MAX_RAM_BUDGET_BYTES
        assert budget < limit, 'the cache must not claim the whole limit'

    def test_shares_divide_the_budget(self):
        from anagnorisis_core.storage.caching import default_ram_budget_bytes
        whole = default_ram_budget_bytes(1.0)
        part = default_ram_budget_bytes(0.25)
        assert part <= whole

    def test_sizes_track_what_is_actually_stored(self):
        """Entry counts cannot work here: measured on a real library, entries
        range from 296 bytes to 60 KB depending on the cache."""
        import numpy as np
        from anagnorisis_core.storage.caching import estimate_bytes

        assert estimate_bytes(np.zeros(1024, dtype=np.float32)) >= 4096
        assert estimate_bytes('x' * 5000) >= 5000
        assert estimate_bytes([np.zeros(256, dtype=np.float32)] * 4) >= 4096
        assert estimate_bytes(None) < 100


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

    def _corrupt_shard_for(self, tmp_path, key):
        shard = DiskCache._shard_for_key(key)
        shard_path = os.path.join(str(tmp_path), f"{shard:02x}.pkl")
        with open(shard_path, 'wb') as f:
            f.write(b'THIS IS NOT VALID PICKLE DATA!!!')
        return shard_path

    def test_corrupted_shard_does_not_break_the_live_cache(self, tmp_path):
        """A cache that already knows the value keeps serving it.

        This used to assert the value became None, on the assumption that every
        get() re-reads the shard from disk. It does not — the shard stays in
        memory — so the value survives the file being destroyed. That is the
        better outcome, and worth pinning down rather than treating as a bug.
        """
        cache = self._make_cache(tmp_path)
        cache.set('before_corrupt', 'ok')
        self._corrupt_shard_for(tmp_path, 'before_corrupt')

        assert cache.get('before_corrupt') == 'ok'

    def test_a_fresh_reader_recovers_from_a_corrupt_shard(self, tmp_path):
        """The case that actually matters: nothing in memory to fall back on.

        A restart after a bad shutdown reads the file cold. It must return
        nothing rather than raising — a cache miss costs recomputation, an
        exception here would take down whatever was searching.
        """
        cache = self._make_cache(tmp_path)
        cache.set('before_corrupt', 'ok')
        self._corrupt_shard_for(tmp_path, 'before_corrupt')

        fresh = self._make_cache(tmp_path)
        assert fresh.get('before_corrupt') is None

    def test_the_cache_is_usable_again_after_corruption(self, tmp_path):
        """Recovery has to be more than not crashing once."""
        cache = self._make_cache(tmp_path)
        cache.set('before_corrupt', 'ok')
        self._corrupt_shard_for(tmp_path, 'before_corrupt')

        fresh = self._make_cache(tmp_path)
        assert fresh.get('before_corrupt') is None
        fresh.set('after_corrupt', 'written')
        assert fresh.get('after_corrupt') == 'written'
        assert self._make_cache(tmp_path).get('after_corrupt') == 'written', \
            'the replacement shard must be readable by a later instance'

    def test_no_partial_files_on_write(self, tmp_path):
        """Atomic writes: no .pkl.tmp or similar leftover files after set()."""
        cache = self._make_cache(tmp_path)
        cache.set('atomic', 'write')
        leftovers = [f for f in os.listdir(str(tmp_path)) if not f.endswith('.pkl')]
        assert not leftovers, f"Unexpected leftover files: {leftovers}"

    def test_a_shard_read_does_not_flood_the_ram_tier(self, tmp_path):
        """Reading one key must not pull its shard-mates into RAM.

        Shard warming used to do exactly that. It bought nothing measurable,
        because TwoLevelCache.get already promotes the key it read, and against
        a bounded RAM tier it is harmful: one shard load floods the budget with
        entries nobody asked for and evicts the ones in use.
        """
        from anagnorisis_core.storage.caching import TwoLevelCache

        writer = TwoLevelCache(cache_dir=str(tmp_path), name='writer')
        for i in range(50):
            writer.set(f'key{i}', f'value{i}')
        writer.disk.close()

        reader = TwoLevelCache(cache_dir=str(tmp_path), name='reader')
        assert reader.get('key0') == 'value0'
        assert len(reader.ram._data) == 1, (
            f'reading one key put {len(reader.ram._data)} entries in RAM')

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

class TestDiskCacheIsBounded:
    """Resident shards and total disk are both capped now.

    A pass over a large library reads every shard, so without a cap the whole
    cache ended up in memory and stayed there.
    """

    def test_resident_shards_are_held_to_the_byte_budget(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600,
                          write_back=False, max_resident_bytes=8192)
        for i in range(200):
            cache.set(f'key{i}', 'x' * 400)
        for i in range(200):
            cache.get(f'key{i}')

        assert cache._resident_bytes <= 8192 or len(cache._shard_data) == 1, (
            f'{cache._resident_bytes} bytes resident against an 8192 budget')
        assert cache._shard_evictions > 0, 'nothing was ever evicted'

    def test_shards_are_not_evicted_when_they_fit(self, tmp_path):
        """Capping by shard *count* rather than bytes evicted shards moments
        before they were needed again: measured on a real library it turned
        256 disk reads into 4,031 and ran 15x slower on an external drive."""
        cache = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600,
                          write_back=False, max_resident_bytes=64 * 1024 * 1024)
        for i in range(500):
            cache.set(f'key{i}', 'x' * 200)
        reads_after_writes = cache._stats['reads']
        for _ in range(3):
            for i in range(500):
                cache.get(f'key{i}')

        assert cache._stats['reads'] == reads_after_writes, (
            'shards were re-read from disk despite fitting in the budget')
        assert cache._shard_evictions == 0

    def test_an_evicted_shard_is_re_read_not_lost(self, tmp_path):
        """Eviction is a memory decision, never a correctness one."""
        cache = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600,
                          write_back=False, max_resident_bytes=1)
        for i in range(100):
            cache.set(f'key{i}', f'value{i}')
        for i in range(100):
            assert cache.get(f'key{i}') == f'value{i}', f'key{i} was lost'

    def test_a_shard_with_unflushed_writes_is_not_evicted(self, tmp_path):
        """Dropping it would throw away writes that are not on disk yet."""
        cache = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600,
                          write_back=True, max_resident_bytes=1,
                          flush_interval_seconds=3600)
        for i in range(50):
            cache.set(f'key{i}', f'value{i}')
        for i in range(50):
            assert cache.get(f'key{i}') == f'value{i}'
        cache.close()

    def test_disk_budget_prunes_least_recently_used_entries(self, tmp_path):
        """The mechanism that clears out orphaned model generations.

        Keys carry the model that produced them, so a model change orphans
        everything written under the old one. Nothing reads those again, so
        pruning by last use removes them without the cache needing to know what
        a model hash is.
        """
        # A long TTL on purpose: the point is that pruning removes entries the
        # TTL would happily keep for another three months, which is exactly the
        # situation an orphaned model generation is in.
        ninety_days = 90 * 24 * 3600
        writer = DiskCache(cache_dir=str(tmp_path), ttl_seconds=ninety_days,
                           write_back=False)
        old = time.time() - 10_000
        # Big enough entries that a shard is worth pruning: with 256 shards,
        # tiny values leave each one below the floor and nothing is dropped.
        for i in range(300):
            writer.set(f'stale-generation-key{i}', 'x' * 8000)
        # Age them: this is what a superseded generation looks like on disk.
        for shard in list(writer._shard_data):
            data = writer._shard_data[shard]
            for key in list(data):
                value, _ts = data[key]
                data[key] = (value, old)
            writer._atomic_write_pickle(writer._shard_paths[shard], data)

        before = sum(os.path.getsize(p) for p in writer._shard_paths
                     if os.path.exists(p))

        # A second cache over the same directory, with a budget far below what
        # is there, prunes as it reads.
        reader = DiskCache(cache_dir=str(tmp_path), ttl_seconds=ninety_days,
                           write_back=False, max_disk_bytes=before // 8,
                           cleanup_write_threshold=1)
        for i in range(300):
            reader.get(f'stale-generation-key{i}')

        assert reader._pruned_entries > 0, 'nothing was pruned under budget'

    def test_no_pruning_without_a_budget(self, tmp_path):
        cache = DiskCache(cache_dir=str(tmp_path), ttl_seconds=3600,
                          write_back=False)
        for i in range(100):
            cache.set(f'key{i}', 'x' * 400)
        for i in range(100):
            assert cache.get(f'key{i}') is not None
        assert cache._pruned_entries == 0


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
