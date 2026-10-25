"""
This module provides a small, fast two‑tier cache with a simple get/set API. The idea is
to keep hot data in memory for instant access while backing everything on disk for
persistence across runs. RAMCache is a tiny, thread‑safe store with per‑key TTL; DiskCache
is a filesystem store that shards keys across many small pickle files to keep each read and
write cheap. Sharding is based on a fast hash so keys are evenly spread without any
central index.

Disk I/O is minimized by design. On a disk miss the whole shard is loaded once and kept,
so subsequent lookups for nearby keys need no further reads. Expired entries are cleaned
lazily when a shard is read, and the shard is only rewritten when it matters (or when we
actually modify a value), avoiding write‑amplification during steady reads. When a key is
read, its lifetime is optionally refreshed on disk after a configurable interval, and
always refreshed in RAM, so frequently accessed items stay hot without rewriting on every
access.

Both tiers are bounded, and that is load‑bearing rather than tidiness. Cache keys carry
the identity of the model that produced the value, so every model change orphans whatever
was written under the old one; nothing reads those entries again. Left unbounded, a pass
over a large library pulled the whole cache — live and orphaned alike — into memory and
kept it there until the process was killed. The RAM tier now evicts least‑recently‑used
entries against a byte budget derived from the process's actual memory limit, the shard
cache keeps only the most recently used shards resident, and an optional disk budget
prunes the least recently used entries of a shard as it is read. Since an orphaned
generation is precisely the part nothing touches, pruning by last use removes it without
this file needing to know anything about model hashes or key formats.

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
import contextlib
from collections import OrderedDict

try:
    import fcntl                      # POSIX advisory locks
except ImportError:                    # pragma: no cover - Windows
    fcntl = None
from typing import Tuple, Dict, Any, Optional

three_months_in_seconds = 90 * 24 * 60 * 60
one_hour_in_seconds = 3600
ten_minutes_in_seconds = 60 * 10
five_minutes_in_seconds = 60 * 5

# Share of the process's memory limit that all caches together may hold in RAM.
# The rest belongs to the models, which need several GB while resident.
DEFAULT_RAM_BUDGET_FRACTION = 0.12
MIN_RAM_BUDGET_BYTES = 64 * 1024 * 1024
MAX_RAM_BUDGET_BYTES = 4 * 1024 * 1024 * 1024

# Applied to caches created after this is set. Every cache is reached through
# get_two_level_cache(), which does not take a config, so the settings are put
# here once at startup rather than threaded through eight call sites.
_BUDGETS = {
    'ram_fraction': DEFAULT_RAM_BUDGET_FRACTION,
    # Off unless asked for. Pruning is by least-recent use, and a live entry
    # that simply has not been searched lately looks exactly like an orphaned
    # one, so too small a budget quietly throws away embeddings that then cost
    # real time to rebuild. The right number depends on how big the library is,
    # which only the person running it knows.
    'max_disk_bytes': None,
    # Also off by default, and for a sharper reason: capping the shard tier on
    # a cache bigger than the cap means re-reading shards from disk constantly.
    # Only worth setting once the cache has been pruned to its live size.
    'shard_resident_bytes': None,
}


def configure_cache_budgets(ram_fraction: Optional[float] = None,
                            max_disk_bytes: Optional[int] = None,
                            shard_resident_bytes: Optional[int] = None) -> None:
    """Set cache limits for caches created from now on.

    Call once at startup. Existing caches keep the limits they were built with.
    """
    if ram_fraction is not None:
        _BUDGETS['ram_fraction'] = max(0.01, min(float(ram_fraction), 0.9))
    if max_disk_bytes is not None:
        _BUDGETS['max_disk_bytes'] = int(max_disk_bytes) or None
    if shard_resident_bytes is not None:
        _BUDGETS['shard_resident_bytes'] = int(shard_resident_bytes) or None


def memory_limit_bytes() -> int:
    """How much memory this process is actually allowed, in bytes.

    Inside a container that is the cgroup limit, not the size of the machine:
    reading the host's total would let the cache size itself for 30 GB while
    the container is killed at 16 GB. cgroup v2 first, then v1, then the host
    as a last resort.
    """
    for path in ('/sys/fs/cgroup/memory.max',
                 '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
            if raw and raw != 'max':
                value = int(raw)
                # cgroup v1 reports a sentinel near 2^63 when unlimited.
                if 0 < value < (1 << 62):
                    return value
        except (OSError, ValueError):
            pass

    # Not in a limited cgroup, or the files are not where we looked.
    try:
        return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    except (OSError, ValueError, AttributeError):
        return 2 * 1024 * 1024 * 1024


def default_ram_budget_bytes(share: float = 1.0) -> int:
    """A cache's slice of the total RAM budget.

    *share* is that cache's fraction of the whole budget, so several caches can
    divide one allowance rather than each claiming it in full.
    """
    total = memory_limit_bytes() * _BUDGETS['ram_fraction']
    budget = int(total * max(share, 0.0))
    return max(MIN_RAM_BUDGET_BYTES, min(budget, MAX_RAM_BUDGET_BYTES))


def estimate_bytes(value: Any, _depth: int = 0) -> int:
    """Roughly what *value* costs to keep in memory.

    Deliberately cheap and approximate: it exists to keep a budget honest, not
    to be exact. Numpy arrays and strings are the overwhelming majority of what
    is cached and both report their size directly. Containers are sampled
    rather than walked, because this runs on every write.
    """
    if value is None:
        return 16
    nbytes = getattr(value, 'nbytes', None)   # numpy arrays and friends
    if isinstance(nbytes, int):
        return nbytes + 96
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) + 48
    if isinstance(value, (int, float, bool)):
        return 32
    if _depth >= 2:
        return 256
    if isinstance(value, (list, tuple)):
        if not value:
            return 56
        sample = value[:8]
        per_item = sum(estimate_bytes(v, _depth + 1) for v in sample) / len(sample)
        return int(per_item * len(value)) + 56
    if isinstance(value, dict):
        if not value:
            return 64
        sample = list(value.items())[:8]
        per_item = sum(estimate_bytes(k, _depth + 1) + estimate_bytes(v, _depth + 1)
                       for k, v in sample) / len(sample)
        return int(per_item * len(value)) + 64
    return 256

@contextlib.contextmanager
def _shard_file_lock(path: str):
    """Hold an exclusive, cross-process lock while a shard is rewritten.

    More than one process can share a cache directory — the application and the
    `anagnorisis` command are the obvious pair. Shard writes are already atomic,
    so nothing can be corrupted, but a read-modify-write from two processes at
    once would silently drop one side's entries: both read the same shard, each
    merges its own additions, and whichever writes second wins the whole file.

    Locking the read-modify-write, and re-reading inside the lock, turns that
    into a merge instead of a race. Lost cache entries are only ever a
    performance bug — the data recomputes — but they are the kind that hides for
    months, so it is worth the few lines.

    Falls back to doing nothing where fcntl is unavailable (Windows), which is no
    worse than the behaviour this replaces.
    """
    if fcntl is None:
        yield
        return
    lock_path = path + '.lock'
    fh = None
    try:
        fh = open(lock_path, 'a+b')
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fh is not None:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            finally:
                fh.close()


class RAMCache:
    """
    A thread-safe in-memory cache with a per-key TTL and a size budget.

    Bounded because it used to not be. Entries only expired when the same key
    was read again, so anything never read again stayed for the life of the
    process, and a pass over a large library pulled the whole cache into memory
    until the process was killed. The budget is in bytes rather than entries
    because what is stored varies by two orders of magnitude between caches:
    measured on a real library, 296 bytes per entry for file listings against
    60 KB for audio embeddings. Counting entries would either waste most of the
    budget or blow through it, depending on which cache you looked at.

    Eviction is least-recently-used: reads move a key to the back, and writes
    drop from the front until the total fits.
    """
    def __init__(self, ttl_seconds: int, max_bytes: Optional[int] = None):
        self.ttl = ttl_seconds
        self.max_bytes = max_bytes
        # key -> (timestamp, value), in least-recently-used order
        self._data: "OrderedDict[str, Tuple[float, Any]]" = OrderedDict()
        self._sizes: Dict[str, int] = {}
        self._total_bytes = 0
        self._evictions = 0
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
                self._forget(key)
                return None

            self._data.move_to_end(key)   # most recently used
            return value

    def set(self, key: str, value):
        """
        Adds an item to the cache with the current timestamp.
        """
        now = time.time()
        size = estimate_bytes(value) + len(key) + 64
        with self._lock:
            if key in self._data:
                self._total_bytes -= self._sizes.pop(key, 0)
            self._data[key] = (now, value)
            self._data.move_to_end(key)
            self._sizes[key] = size
            self._total_bytes += size
            self._evict_if_needed()

    # -- internals ------------------------------------------------------

    def _forget(self, key: str) -> None:
        """Drop one key. Caller holds the lock."""
        self._data.pop(key, None)
        self._total_bytes -= self._sizes.pop(key, 0)

    def _evict_if_needed(self) -> None:
        """Drop least-recently-used entries until inside the budget.

        Caller holds the lock. The newest entry is never evicted, so a single
        value larger than the whole budget is still served rather than dropped
        the instant it is written.
        """
        if not self.max_bytes:
            return
        while self._total_bytes > self.max_bytes and len(self._data) > 1:
            oldest = next(iter(self._data))
            self._forget(oldest)
            self._evictions += 1

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'entries': len(self._data),
                'bytes': self._total_bytes,
                'max_bytes': self.max_bytes or 0,
                'evictions': self._evictions,
            }

class DiskCache:
    """
    Disk-backed cache with sharded pickle files and TTL.
    Exposes: get(key), set(key, value)

    refresh_after_seconds: on get(), if (now - saved_ts) >= refresh_after_seconds, bump ts and persist.
    """
    def __init__(
        self,
        cache_dir: str,
        ttl_seconds: int,
        refresh_after_seconds: Optional[int] = None,
        cleanup_write_threshold: int = 64,
        *,
        write_back: bool = True,                               # coalesce writes in memory
        flush_interval_seconds: int = five_minutes_in_seconds,  # periodic flush cadence
        flush_batch_size: int = 5 * 1024 * 1024,               # flush when shard buffer reaches this (5mb)
        max_pending_per_shard: int = 4096,                     # safety cap per shard
        max_resident_bytes: Optional[int] = None,              # None = unbounded
        max_disk_bytes: Optional[int] = None,                  # None = unbounded
        name: Optional[str] = None,
    ):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.ttl = ttl_seconds

        # There are only ever 256 shards, so their file paths and their locks
        # are built once here and indexed by shard number.
        #
        # Every read used to rebuild `cache_dir/<shard>.pkl` with os.path.join
        # and then take a global lock purely to find that shard's own lock in a
        # dict. Both answers depend on nothing but the shard number, and a
        # search over a large library does this once per file: measured 17% of
        # the per-file cost with everything else already cached.
        self._shard_paths = [
            os.path.join(self.cache_dir, f"{shard:02x}.pkl") for shard in range(256)
        ]
        self._shard_locks = [threading.Lock() for _ in range(256)]

        # Warming and refresh policy
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

        # In-memory cache of loaded shard data dicts.
        # Populated on the first disk read of a shard, then updated by every
        # set() and _flush_shard(). Eliminates repeated disk I/O when the same
        # shard is queried many times in a row (the typical case for
        # process_files looping over thousands of paths that all miss cache).
        # Bounded, least-recently-used. This used to grow without limit: a pass
        # over a large library reads every shard, so the whole cache ended up
        # resident and stayed that way, which is what exhausted the container.
        #
        # Bounded by *bytes*, not by a count of shards. Keys hash uniformly
        # across all 256, so a sweep touches every one of them; capping the
        # count means a shard is evicted just before it is needed again.
        # Measured on a real library, a 48-of-256 cap turned 256 disk reads
        # into 4,031 and ran 15x slower on an external drive. A byte budget
        # keeps the whole working set resident whenever it fits, and degrades
        # to re-reading only when it genuinely cannot.
        self._shard_data: "OrderedDict[int, Dict[str, Tuple[Any, float]]]" = OrderedDict()
        self._shard_bytes: Dict[int, int] = {}
        self._resident_bytes = 0
        self._max_resident_bytes = int(max_resident_bytes) if max_resident_bytes else None
        self._shard_evictions = 0
        self._shard_data_lock = threading.Lock()

        # Disk budget. Cache keys carry the identity of the model that produced
        # the value, so every model or algorithm change orphans the entries
        # written under the old one. They are never read again but nothing ever
        # removed them: measured on a real library, only 9% of the metadata
        # cache and 25% of the content cache belonged to the live generation.
        # Rather than special-casing model hashes, which would tie this file to
        # the key formats its callers invent, the budget prunes by last use:
        # an orphaned generation is simply the part nothing touches.
        self._max_disk_bytes = int(max_disk_bytes) if max_disk_bytes else None
        self._disk_bytes_checked_at = 0.0
        self._disk_bytes = 0
        self._pruned_entries = 0
        # Identity of the shard file as we last saw it, so a flush can tell
        # whether another process has written since and a re-read is needed.
        self._shard_disk_stamp: Dict[int, tuple] = {}

    @staticmethod
    def _shard_for_key(key: str) -> int:
        b = key.encode('utf-8', 'surrogatepass')
        return hashlib.blake2b(b, digest_size=16).digest()[0]  # 0..255

    def _get_shard_path_and_lock(self, key: str) -> Tuple[int, str, threading.Lock]:
        shard = self._shard_for_key(key)
        return shard, self._shard_paths[shard], self._shard_locks[shard]

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

        removed += self._prune_to_budget(data)
        return data, removed

    # -- disk budget ----------------------------------------------------

    def _disk_usage_bytes(self, max_age_seconds: float = 60.0) -> int:
        """Total size of the shard files, remeasured at most once a minute.

        256 stat calls, so cheap, but not cheap enough to repeat on every
        shard load during a pass over a large library.
        """
        now = time.time()
        if now - self._disk_bytes_checked_at < max_age_seconds:
            return self._disk_bytes
        total = 0
        for path in self._shard_paths:
            try:
                total += os.path.getsize(path)
            except OSError:
                pass
        self._disk_bytes = total
        self._disk_bytes_checked_at = now
        return total

    def _prune_to_budget(self, data: Dict[str, Tuple[Any, float]]) -> int:
        """Drop the least recently used entries of one shard, if over budget.

        Called as a shard is read, so pruning is spread across the shards a
        workload actually touches instead of needing a stop-the-world sweep.
        Each shard is held to its equal share of the budget.

        Entries carry the time they were last written or refreshed, and reads
        refresh them (see *refresh_after_seconds*), so the oldest entries are
        the ones nothing has asked for. That is what makes this remove orphaned
        model generations without knowing anything about model hashes.
        """
        if not self._max_disk_bytes or not data:
            return 0
        if self._disk_usage_bytes() <= self._max_disk_bytes:
            return 0

        # Each shard is held to its equal slice of the budget. The floor is only
        # there to stop a tiny budget pruning shards down to nothing; the real
        # guard is the total-usage check above, which is why it is small. A
        # larger floor would silently disable the budget entirely: at 64 KB no
        # budget under 16 MB could ever have had an effect.
        share = max(self._max_disk_bytes // 256, 2048)
        total = sum(estimate_bytes(value) + len(key)
                    for key, (value, _ts) in data.items())
        if total <= share:
            return 0

        # Oldest first, which is least recently used.
        by_age = sorted(data.items(), key=lambda kv: kv[1][1])
        dropped = 0
        for key, (value, _ts) in by_age:
            if total <= share or len(data) <= 1:
                break
            total -= estimate_bytes(value) + len(key)
            data.pop(key, None)
            dropped += 1

        if dropped:
            self._pruned_entries += dropped
            # The measurement is now stale by roughly what we just removed.
            self._disk_bytes_checked_at = 0.0
        return dropped
    
    def _get_or_load_shard(self, shard: int, shard_path: str) -> Tuple[Dict[str, Tuple[Any, float]], int]:
        """Return shard data from the in-memory cache, loading from disk on first access.

        After the first read of a shard, subsequent lookups are served from RAM
        without touching the disk. The cached dict is shared with callers; do not
        mutate it — use ``set()`` / ``_flush_shard()`` for that.
        """
        with self._shard_data_lock:
            cached = self._shard_data.get(shard)
            if cached is not None:
                self._shard_data.move_to_end(shard)   # most recently used
                return cached, 0

        data, removed = self._load_and_clean_shard(shard_path)
        # The file's size stands in for what the unpickled shard costs. Walking
        # thousands of entries to weigh them precisely would cost more than the
        # eviction decision is worth; measured expansion is roughly 1-2x.
        try:
            resident = os.path.getsize(shard_path) * 2
        except OSError:
            resident = 0
        with self._shard_data_lock:
            self._shard_data[shard] = data
            self._shard_data.move_to_end(shard)
            self._resident_bytes += resident - self._shard_bytes.get(shard, 0)
            self._shard_bytes[shard] = resident
            self._evict_resident_shards()
        return data, removed

    def _note_shard_size(self, shard: int, shard_path: str) -> None:
        """Re-weigh a shard after it changed on disk, and evict if that pushed
        the resident set over budget.

        Writes grow a shard, so accounting only at load time would leave every
        shard recorded at whatever it weighed when first read — zero, for one
        created by this process.
        """
        try:
            resident = os.path.getsize(shard_path) * 2
        except OSError:
            resident = 0
        with self._shard_data_lock:
            if shard not in self._shard_data:
                return
            self._resident_bytes += resident - self._shard_bytes.get(shard, 0)
            self._shard_bytes[shard] = resident
            self._evict_resident_shards()

    def _evict_resident_shards(self) -> None:
        """Drop least-recently-used shards until inside the budget.

        Caller holds the lock. A shard with writes still pending is kept:
        dropping it would lose them, and it becomes evictable once the flusher
        has written it out. The most recent shard is never evicted, so a cache
        whose single shard exceeds the budget still works.
        """
        if not self._max_resident_bytes:
            return
        while (self._resident_bytes > self._max_resident_bytes
               and len(self._shard_data) > 1):
            for candidate in list(self._shard_data.keys())[:-1]:
                with self._pending_lock:
                    has_pending = bool(self._pending.get(candidate))
                if not has_pending:
                    self._shard_data.pop(candidate, None)
                    self._shard_disk_stamp.pop(candidate, None)
                    self._resident_bytes -= self._shard_bytes.pop(candidate, 0)
                    self._shard_evictions += 1
                    break
            else:
                return   # every evictable shard still has writes waiting

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

    def _flush_shard(self, shard: int) -> None:
        """Drain pending updates for shard to disk (merge + atomic write)."""
        with self._pending_lock:
            pending = self._pending.get(shard)
            if not pending:
                return
            updates = pending
            self._pending[shard] = {}

        # Serialize with the shard's file lock while merging and writing. Same
        # path and same lock object the read path uses, both indexed by shard.
        path = self._shard_paths[shard]
        with self._shard_locks[shard], _shard_file_lock(path):
            # Prefer the in-memory copy, but only while it still matches what is
            # on disk. If another process has written this shard since we loaded
            # it, merging into our stale copy would erase their entries.
            with self._shard_data_lock:
                cached = self._shard_data.get(shard)
            if cached is not None and self._disk_stamp(path) == self._shard_disk_stamp.get(shard):
                # Copy so the flush doesn't mutate the live cache mid-read.
                data = dict(cached)
            else:
                data, _ = self._load_and_clean_shard(path)
            data.update(updates)  # pending wins

            if data:
                self._atomic_write_pickle(path, data)
            else:
                # Only rewrite empty shard if file exists (to clear expired/corrupt content)
                if os.path.exists(path):
                    self._atomic_write_pickle(path, data)

            # Keep the cache in sync with what we just wrote to disk, and record
            # the file's identity so the next flush can detect a foreign write.
            with self._shard_data_lock:
                self._shard_data[shard] = data
            self._shard_disk_stamp[shard] = self._disk_stamp(path)
        self._note_shard_size(shard, path)

    @staticmethod
    def _disk_stamp(path: str) -> tuple:
        """Cheap identity of a shard file: (mtime_ns, size), or () if absent."""
        try:
            st = os.stat(path)
            return (st.st_mtime_ns, st.st_size)
        except OSError:
            return ()

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
            # Prefer in-memory shard cache over reading from disk on every call.
            data, removed = self._get_or_load_shard(shard, shard_path)

            # No shard warming. It used to copy every entry of a freshly loaded
            # shard into the RAM tier, but TwoLevelCache.get already promotes
            # the key it read, so warming only ever pre-loaded neighbours
            # nobody had asked for. Measured on a real cache it bought nothing
            # (0.61 vs 0.59 µs per repeated read), and against a *bounded* RAM
            # tier it is actively harmful: one shard load would flood the
            # budget with unrequested entries and evict the ones in use.

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
            data, _ = self._get_or_load_shard(shard, shard_path)
            data[key] = (value, time.time())
            self._atomic_write_pickle(shard_path, data)
        self._note_shard_size(shard, shard_path)

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
        refresh_after_seconds: Optional[int] = None,
        ram_max_bytes: Optional[int] = None,
        ram_budget_share: float = 0.15,
        max_disk_bytes: Optional[int] = None,
        name: Optional[str] = None,
    ):
        # The per-key tier is bounded; the shard tier is not, unless asked for.
        #
        # They hold the same values in different shapes, and only one of them
        # can be capped cheaply. The shard cache is what stands between a lookup
        # and the disk, so on a spinning drive capping it is punishing: measured
        # against a real 2.15 GB metadata cache, a 1 GB shard budget turned 256
        # disk reads into 4,043 and ran 21x slower, and a 216 MB budget 41x.
        #
        # The reason such a cache needs 2 GB in the first place is that most of
        # it is orphaned: entries written under models that have since been
        # replaced, which nothing will ever read. Pruning those (max_disk_bytes)
        # shrinks the cache to its live working set, and only then is capping
        # the shard tier something other than a way to make searching slow.
        budget = (ram_max_bytes if ram_max_bytes is not None
                  else default_ram_budget_bytes(ram_budget_share))
        self.ram = RAMCache(ram_ttl_seconds, max_bytes=budget)

        if cache_dir is None:
            # None is used to indicate a RAM-only cache (no disk persistence)
            self.disk = None
        else:
            # The RAM tier is filled by get() below, one key at a time, as keys
            # are actually asked for. See the note in DiskCache.get for why
            # pre-loading a whole shard into it was removed.
            self.disk = DiskCache(
                cache_dir=cache_dir,
                ttl_seconds=disk_ttl_seconds,
                refresh_after_seconds=refresh_after_seconds,
                max_resident_bytes=_BUDGETS['shard_resident_bytes'],
                max_disk_bytes=(max_disk_bytes if max_disk_bytes is not None
                                else _BUDGETS['max_disk_bytes']),
                name=name,
            )

    def get(self, key: str) -> Optional[Any]:
        # RAM first
        val = self.ram.get(key)
        if val is not None:
            return val

        # Disk next. Disk will warm RAM with the whole shard.
        if self.disk is not None:
            val = self.disk.get(key)
            if val is not None:
                # Ensure RAM TTL is refreshed for the accessed key as well.
                self.ram.set(key, val)
            return val
        
        return None

    def set(self, key: str, value: Any, save_to_disk: bool = True) -> None:
        # Write-through both tiers
        self.ram.set(key, value)
        if save_to_disk and self.disk is not None:
            self.disk.set(key, value)

# ---------------------------------------------------------------------------
# Process-wide instance registry
# ---------------------------------------------------------------------------

_TLC_SINGLETONS: Dict[str, "TwoLevelCache"] = {}
_TLC_LOCK = threading.Lock()


def get_two_level_cache(cache_dir: str, **kwargs) -> "TwoLevelCache":
    """
    Return a shared TwoLevelCache instance for this cache_dir within the process.
    First caller's kwargs win; later calls ignore differing kwargs.

    Two instances over one directory would each keep their own RAM tier, so a
    value written through one would only be visible to the other after a disk
    read. Always go through here rather than constructing TwoLevelCache directly
    for a directory someone else may also use.
    """
    abs_dir = os.path.abspath(cache_dir)
    with _TLC_LOCK:
        inst = _TLC_SINGLETONS.get(abs_dir)
        if inst is not None:
            return inst
        inst = TwoLevelCache(cache_dir=abs_dir, **kwargs)
        _TLC_SINGLETONS[abs_dir] = inst
        return inst
