"""
This module provides a small, fast two‑tier cache with a simple get/set API. The idea is
to keep hot data in memory for instant access while backing everything on disk for
persistence across runs. RAMCache is a tiny, thread‑safe store with per‑key TTL; DiskCache
is a filesystem store that shards keys across many small pickle files to keep each read and
write cheap. Sharding is based on a fast hash so keys are evenly spread without any
central index.

Disk I/O is minimized by design. On a disk miss, the whole shard is loaded once and its
live entries are “warmed” into RAM in one shot, so subsequent lookups for nearby keys
are served from memory. Expired entries are cleaned lazily when a shard is read, and the
shard is only rewritten when it matters (or when we actually modify a value), avoiding
write‑amplification during steady reads. When a key is read, its lifetime is optionally
refreshed on disk after a configurable interval, and always refreshed in RAM, so frequently
accessed items stay hot without rewriting on every access.

All writes to disk are atomic (tempfile + replace) to avoid partial files on crashes, and
a simple per‑shard lock guards concurrent access within a process. The public surface area
is intentionally tiny and predictable: callers only use get and set; the TwoLevelCache
composes RAM and disk as a write‑through, read‑through facade and hides the inner policy
so modules don’t need to think about tiers, warming, or cleanup. This keeps the cache
reliable, fast enough for typical workloads, and easy to evolve without touching callers.
"""

import os
import tempfile
import threading
import time
import pickle
import hashlib
import atexit
from typing import Tuple, Dict, Any, Optional

three_months_in_seconds = 90 * 24 * 60 * 60
one_hour_in_seconds = 3600
ten_minutes_in_seconds = 60 * 10
five_minutes_in_seconds = 60 * 5

class RAMCache:
    """
    A simple thread-safe in-memory cache with a Time-To-Live (TTL) for each item.
    """
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        # key -> (timestamp, value)
        self._data: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        """
        Retrieves an item from the cache. Returns the item if it exists and has not
        expired, otherwise returns None.
        """
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            
            ts, value = item

            # check if item has expired
            if self.ttl and now - ts > self.ttl:
                # expired
                del self._data[key]
                return None
            return value
        
    def set(self, key: str, value):
        """
        Adds an item to the cache with the current timestamp.
        """
        now = time.time()
        with self._lock:
            self._data[key] = (now, value)

class DiskCache:
    """
    Disk-backed cache with sharded pickle files and TTL.
    Exposes: get(key), set(key, value)

    on_shard_loaded: optional callback(dict[key]=value) to warm a higher-level cache.
    warm_interval_seconds: how often to invoke on_shard_loaded per shard (limits repeated warms).
    refresh_after_seconds: on get(), if (now - saved_ts) >= refresh_after_seconds, bump ts and persist.
    """
    def __init__(
        self,
        cache_dir: str,
        ttl_seconds: int,
        on_shard_loaded: Optional[callable] = None,
        warm_interval_seconds: int = 60,
        refresh_after_seconds: Optional[int] = None,
        cleanup_write_threshold: int = 64,
        *,
        write_back: bool = True,                               # coalesce writes in memory
        flush_interval_seconds: int = five_minutes_in_seconds,  # periodic flush cadence
        flush_batch_size: int = 5 * 1024 * 1024,               # flush when shard buffer reaches this (5mb)
        max_pending_per_shard: int = 4096,                     # safety cap per shard
        name: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.ttl = ttl_seconds
        self._locks: Dict[int, threading.Lock] = {}
        self._locks_master = threading.Lock()

        # Warming and refresh policy
        self._on_shard_loaded = on_shard_loaded
        self._warm_interval_seconds = max(0, int(warm_interval_seconds))
        self._last_warm: Dict[int, float] = {}
        # Default refresh interval: 1/4 of TTL (at least 1s) to avoid writing on every read
        self._refresh_after = (
            int(refresh_after_seconds) if refresh_after_seconds is not None
            else (max(1, int(ttl_seconds * 0.25)) if ttl_seconds else 0)
        )
        self._cleanup_write_threshold = max(0, int(cleanup_write_threshold))

        # Write-back settings
        self._write_back = bool(write_back)
        self._flush_interval = max(1, int(flush_interval_seconds))
        self._flush_batch_size = max(1, int(flush_batch_size))
        self._max_pending_per_shard = max(1, int(max_pending_per_shard))
        self._pending: Dict[int, Dict[str, Tuple[Any, float]]] = {}
        self._pending_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._flusher_thread: Optional[threading.Thread] = None
        if self._write_back:
            self._flusher_thread = threading.Thread(target=self._flusher_loop, daemon=True)
            self._flusher_thread.start()
            atexit.register(self.close)

        # I/O statistics
        self._stats_lock = threading.Lock()
        self._stats = {
            "reads": 0,            # shard file loads
            "writes": 0,           # shard file writes
            "bytes_read": 0,
            "bytes_written": 0,
        }

        # Optional name for logging
        self.name = name or "NoName"

        self.stat_time = time.time()

        # Track last printed snapshot to avoid noisy logs
        self._last_stats_snapshot = {"reads": 0, "writes": 0, "bytes_read": 0, "bytes_written": 0}
        self._last_pending_snapshot = 0

    @staticmethod
    def _shard_for_key(key: str) -> int:
        b = key.encode('utf-8', 'surrogatepass')
        return hashlib.blake2b(b, digest_size=16).digest()[0]  # 0..255

    def _get_shard_path_and_lock(self, key: str) -> Tuple[int, str, threading.Lock]:
        shard = self._shard_for_key(key)
        path = os.path.join(self.cache_dir, f"{shard:02x}.pkl")
        with self._locks_master:
            lock = self._locks.get(shard)
            if lock is None:
                lock = threading.Lock()
                self._locks[shard] = lock
        return shard, path, lock

    def _load_and_clean_shard(self, shard_path: str) -> Tuple[Dict[str, Tuple[Any, float]], int]:
        """
        Return (data, removed_count). We track how many expired entries we dropped so
        callers can decide whether it's worth rewriting the shard.
        """
        data: Dict[str, Tuple[Any, float]] = {}
        removed = 0
        now = time.time()

        if os.path.exists(shard_path):
            size = 0
            try:
                try:
                    size = os.path.getsize(shard_path)
                except OSError:
                    size = 0
                try:
                    with open(shard_path, 'rb') as f:
                        obj = pickle.load(f)
                        if isinstance(obj, dict):
                            data = obj
                        else:
                            data = {}
                except Exception:
                    # Corrupt shard; reset to empty
                    data = {}
            finally:
                with self._stats_lock:
                    self._stats["reads"] += 1
                    self._stats["bytes_read"] += int(size)

        if self.ttl and data:
            expired = [k for k, (_, ts) in data.items() if now - ts > self.ttl]
            for k in expired:
                data.pop(k, None)
            removed = len(expired)

        return data, removed

    def _atomic_write_pickle(self, path: str, data: Dict[str, Tuple[Any, float]]) -> None:
        d = os.path.dirname(path)
        fd, tmp = tempfile.mkstemp(dir=d, prefix='.shard_', suffix='.pkl')
        wrote_bytes = 0
        try:
            with os.fdopen(fd, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
                try:
                    wrote_bytes = f.tell()
                except Exception:
                    wrote_bytes = 0
            os.replace(tmp, path)  # atomic on POSIX      
        finally:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            finally:
                with self._stats_lock:
                    self._stats["writes"] += 1
                    self._stats["bytes_written"] += int(wrote_bytes)

    def _log_stats(self) -> None:
        with self._stats_lock:
            snap = dict(self._stats)
        pending = 0
        if self._write_back:
            with self._pending_lock:
                pending = sum(len(p) for p in self._pending.values())

        # Print only if something changed since the last attempt
        last = getattr(self, "_last_stats_snapshot", None) or {}
        last_pending = getattr(self, "_last_pending_snapshot", 0)
        changed = (
            snap.get("reads") != last.get("reads") or
            snap.get("writes") != last.get("writes") or
            snap.get("bytes_read") != last.get("bytes_read") or
            snap.get("bytes_written") != last.get("bytes_written") or
            pending != last_pending
        )
        if not changed:
            return

        pid = os.getpid()
        now = time.time()
        elapsed = now - self.stat_time
        print(
            f"[DiskCache pid={pid} name={self.name}] reads="
            f"{snap['reads']} ({snap['bytes_read']/1e6:.1f}MB), "
            f"writes={snap['writes']} ({snap['bytes_written']/1e6:.1f}MB), "
            f"pending={pending} after {elapsed:.1f}s"
        )

        # Update snapshots and reset elapsed window
        self._last_stats_snapshot = {
            "reads": snap["reads"],
            "writes": snap["writes"],
            "bytes_read": snap["bytes_read"],
            "bytes_written": snap["bytes_written"],
        }
        self._last_pending_snapshot = pending
        self.stat_time = now

    def stats(self, reset: bool = False) -> Dict[str, int]:
        with self._stats_lock:
            snap = dict(self._stats)
            if reset:
                for k in self._stats:
                    self._stats[k] = 0
        return snap

    def reset_stats(self) -> None:
        self.stats(reset=True)

    def _maybe_warm_ram(self, shard: int, data: Dict[str, Tuple[Any, float]]) -> None:
        if not self._on_shard_loaded:
            return
        now = time.time()
        last = self._last_warm.get(shard, 0.0)
        if now - last < self._warm_interval_seconds:
            return
        # Prepare a compact dict of unexpired entries key -> value
        if self.ttl:
            to_warm = {k: v for k, (v, ts) in data.items() if now - ts <= self.ttl}
        else:
            to_warm = {k: v for k, (v, _) in data.items()}
        if to_warm:
            try:
                self._on_shard_loaded(to_warm)
            finally:
                self._last_warm[shard] = now

    def _flush_shard(self, shard: int) -> None:
        """Drain pending updates for shard to disk (merge + atomic write)."""
        with self._pending_lock:
            pending = self._pending.get(shard)
            if not pending:
                return
            updates = pending
            self._pending[shard] = {}

        # Serialize with the shard's file lock while merging and writing
        path = os.path.join(self.cache_dir, f"{shard:02x}.pkl")
        with self._locks_master:
            lock = self._locks.get(shard) or threading.Lock()
            self._locks[shard] = lock
        with lock:
            data, _ = self._load_and_clean_shard(path)
            data.update(updates)  # pending wins
            if data:
                self._atomic_write_pickle(path, data)
            else:
                # Only rewrite empty shard if file exists (to clear expired/corrupt content)
                if os.path.exists(path):
                    self._atomic_write_pickle(path, data)

    def _flusher_loop(self) -> None:
        """Periodically flush all shards with pending updates."""
        while not self._stop_event.wait(self._flush_interval):
            with self._pending_lock:
                shards = [s for s, pend in self._pending.items() if pend]
            for shard in shards:
                self._flush_shard(shard)
            
            # Print stats after each flush cycle
            self._log_stats()

    def get(self, key: str) -> Optional[Any]:
        shard, shard_path, lock = self._get_shard_path_and_lock(key)

        # Serve from pending write-back if available (no disk read)
        if self._write_back:
            with self._pending_lock:
                entry = self._pending.get(shard, {}).get(key)
            if entry is not None:
                value, ts = entry
                now = time.time()
                if self.ttl and now - ts > self.ttl:
                    # Drop expired pending item
                    with self._pending_lock:
                        self._pending.get(shard, {}).pop(key, None)
                    return None
                # Refresh timestamp in pending if needed
                if self.ttl and self._refresh_after and (now - ts >= self._refresh_after):
                    with self._pending_lock:
                        # Re-check in case it changed
                        curr = self._pending.get(shard, {}).get(key)
                        if curr is not None:
                            self._pending[shard][key] = (value, now)
                return value

        with lock:
            data, removed = self._load_and_clean_shard(shard_path)

            # Warm RAM with the whole shard (throttled)
            self._maybe_warm_ram(shard, data)

            entry = data.get(key)
            if entry is None:
                if removed >= self._cleanup_write_threshold and data:
                    self._atomic_write_pickle(shard_path, data)
                return None

            value, ts = entry
            now = time.time()

            if self.ttl and now - ts > self.ttl:
                data.pop(key, None)
                self._atomic_write_pickle(shard_path, data)
                return None

            wrote = False
            if self.ttl and self._refresh_after and (now - ts >= self._refresh_after):
                data[key] = (value, now)
                wrote = True

            if wrote or (removed >= self._cleanup_write_threshold):
                if self._write_back:
                    # Defer write by placing refreshed entry into pending
                    with self._pending_lock:
                        self._pending.setdefault(shard, {})[key] = (value, now)
                else:
                    self._atomic_write_pickle(shard_path, data)

            return value

    def set(self, key: str, value: Any) -> None:
        shard, shard_path, lock = self._get_shard_path_and_lock(key)
        if self._write_back:
            now = time.time()
            flush_now = False
            with self._pending_lock:
                bucket = self._pending.setdefault(shard, {})
                bucket[key] = (value, now)
                if len(bucket) >= self._flush_batch_size or len(bucket) >= self._max_pending_per_shard:
                    flush_now = True
            if flush_now:
                # Flush this shard synchronously to bound memory/durability window
                self._flush_shard(shard)
            return

        # Immediate write-through (existing behavior)
        with lock:
            data, _ = self._load_and_clean_shard(shard_path)
            data[key] = (value, time.time())
            self._atomic_write_pickle(shard_path, data)

    def flush(self) -> None:
        """Flush all pending shards to disk."""
        if not self._write_back:
            return
        with self._pending_lock:
            shards = [s for s, pend in self._pending.items() if pend]
        for shard in shards:
            self._flush_shard(shard)

    def close(self) -> None:
        """Stop flusher and flush pending updates."""
        if self._write_back:
            self._stop_event.set()
            if self._flusher_thread and self._flusher_thread.is_alive():
                self._flusher_thread.join(timeout=self._flush_interval + 1)
            self.flush()


class TwoLevelCache:
    """
    Simple two-level cache (RAM + Disk) that hides inner complexity.
    Exposes: get(key), set(key, value)
    - Warms RAM with entire shards on disk reads (throttled).
    - Refreshes TTL in RAM on get(); disk refresh happens inside DiskCache.get().
    """
    def __init__(
        self,
        cache_dir: str,
        disk_ttl_seconds: int = three_months_in_seconds,
        ram_ttl_seconds: int = one_hour_in_seconds,
        warm_interval_seconds: int = 60,
        refresh_after_seconds: Optional[int] = None,
        name: Optional[str] = None,
    ):
        self.ram = RAMCache(ram_ttl_seconds)

        # When DiskCache loads a shard, push all its values into RAM (one-hot timestamps).
        def _on_shard_loaded(mapping: Dict[str, Any]):
            for k, v in mapping.items():
                self.ram.set(k, v)

        self.disk = DiskCache(
            cache_dir=cache_dir,
            ttl_seconds=disk_ttl_seconds,
            on_shard_loaded=_on_shard_loaded,
            warm_interval_seconds=warm_interval_seconds,
            refresh_after_seconds=refresh_after_seconds,
            name=name,
        )

    def get(self, key: str) -> Optional[Any]:
        # RAM first
        val = self.ram.get(key)
        if val is not None:
            return val

        # Disk next. Disk will warm RAM with the whole shard.
        val = self.disk.get(key)
        if val is not None:
            # Ensure RAM TTL is refreshed for the accessed key as well.
            self.ram.set(key, val)
        return val

    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        # Write-through both tiers
        self.ram.set(key, value)
        if save_to_disk:
            self.disk.set(key, value)