"""Measure how fast and how well search works, reproducibly.

Speed is reported per search mode, split into the phases that actually scale, so
a number can be extrapolated honestly instead of guessed:

    list    walking the filesystem for candidate files
    query   embedding the query text (a constant, not per-file)
    score   cache lookups + similarity (this is what scales with file count)

Cold means the cache has to come off disk into RAM; warm is the average of
several repeats with it already there. Both matter: cold is what you feel after a
restart, warm is what you feel while working.

Quality is measured by self-retrieval: take a file's own description as the
query and check whether that file comes back first. It needs no labelled data,
it is reproducible, and it is the check a person does by hand when they paste a
file's tags into the search box and expect that file at the top. A search engine
that fails this is broken in a way no timing will reveal.

    python3 anagnorisis_core/benchmarks/search_benchmark.py --media /mnt/media \
        [--project-config DIR] [--index] [--describe-sample 3] [--target 100000]

Needs anagnorisis-core installed — it imports the engine like anything else does,
and nothing else. There is no dependency on the application here, which is why
this lives beside the package rather than at the repository root.
"""
import argparse
import json
import os
import random
import statistics
import tempfile
import time

from anagnorisis_core import api
from anagnorisis_core.config import load_config
from anagnorisis_core.progress import NullProgress

WARM_RUNS = 5
QUERIES = {
    # Deliberately ordinary phrases: a benchmark that only uses queries which
    # happen to work would flatter the engine.
    'name': 'winter',
    'semantic': 'a quiet acoustic recording',
    'metadata': 'a quiet acoustic recording',
}


def _clear_ram_tier(cache):
    """Force the next read to come off disk.

    Clears the RAM tier and the record of which shards were warmed, so the disk
    tier reloads them. Pending writes are flushed first — measuring a cold read
    should not also lose data.
    """
    try:
        if getattr(cache, 'disk', None) is not None:
            cache.disk.flush()
            cache.disk._last_warm.clear()
        if getattr(cache, 'ram', None) is not None:
            with cache.ram._lock:
                cache.ram._data.clear()
    except Exception as exc:
        print(f'  ! could not cool the cache ({exc}); cold numbers will be warm')


def _caches_for(mode, cfg):
    """The caches a given mode reads, so they can be cooled between runs."""
    caches = []
    if mode == 'metadata':
        from anagnorisis_core.search.metadata_search import get_metadata_search
        caches.append(get_metadata_search(cfg)._fast_cache)
    elif mode == 'semantic':
        from anagnorisis_core.search.content_search import get_content_search
        from anagnorisis_core.media.media_types import get_registry
        for name in get_registry(cfg).names:
            try:
                caches.append(get_content_search(cfg, name)._fast_cache)
            except Exception:
                pass
    return caches


def _time_search(mode, files, cfg):
    """One search, timed in two parts: (query_seconds, score_seconds, hits).

    Separated because they scale differently and mixing them makes any
    extrapolation wrong. Embedding the query is a *constant* — one short piece of
    text through the model on the CPU — while scoring is per-file. Divide the
    combined figure by the file count and you attribute a fixed cost to every
    file, which at 100k files inflates the estimate by orders of magnitude.
    """
    import math

    if mode == 'name':
        # No model involved; the query costs nothing to prepare.
        t0 = time.perf_counter()
        scored = [(fp, api._name_score(QUERIES[mode], fp)) for fp in files]
        score_s = time.perf_counter() - t0
        query_s = 0.0
    elif mode == 'semantic':
        from anagnorisis_core.search.content_search import get_content_search
        from anagnorisis_core.media.media_types import get_registry

        registry = get_registry(cfg)
        by_type = {}
        for fp in files:
            mt = registry.for_file(fp)
            if mt and api.models.has_content_embedding(mt.name):
                by_type.setdefault(mt.name, []).append(fp)

        first_engine = get_content_search(cfg, next(iter(by_type))) if by_type else None
        t0 = time.perf_counter()
        query_vec = first_engine.process_text(QUERIES[mode]) if first_engine else None
        query_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        scored = []
        for type_name, group in by_type.items():
            engine = get_content_search(cfg, type_name)
            embeddings = engine.process_files(group, generate_embs_if_not_in_cache=False)
            scored.extend(zip(group, engine.compare(embeddings, query_vec)))
        score_s = time.perf_counter() - t0
    else:
        from anagnorisis_core.search.metadata_search import get_metadata_search

        engine = get_metadata_search(cfg)
        t0 = time.perf_counter()
        query_vec = engine.process_query(QUERIES[mode])
        query_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        embeddings = engine.process_files(files, generate_embs_if_not_in_cache=False)
        scored = list(zip(files, engine.compare(embeddings, query_vec)))
        score_s = time.perf_counter() - t0

    hits = [p for p, sc in scored if sc is not None and not math.isnan(sc)]
    return query_s, score_s, len(hits)


def measure_speed(files, cfg, target):
    rows = []
    for mode in ('name', 'semantic', 'metadata'):
        caches = _caches_for(mode, cfg)

        # Load the model first. Model loading is a one-off with nothing to do
        # with how long a search takes, and folding it in would hide every other
        # number behind it.
        t0 = time.perf_counter()
        _time_search(mode, files[:1], cfg)
        warmup = time.perf_counter() - t0

        for c in caches:
            _clear_ram_tier(c)
        cold_q, cold_s, cold_hits = _time_search(mode, files, cfg)

        warm_q, warm_s = [], []
        for _ in range(WARM_RUNS):
            q, sc, _ = _time_search(mode, files, cfg)
            warm_q.append(q)
            warm_s.append(sc)

        warm_score = statistics.mean(warm_s)
        rows.append({
            'mode': mode,
            'files': len(files),
            'indexed_hits': cold_hits,
            'model_warmup_s': round(warmup, 3),
            'query_s': round(statistics.mean(warm_q), 4),
            'cold_score_s': round(cold_s, 4),
            'warm_score_s': round(warm_score, 4),
            'warm_score_stdev_s': round(statistics.pstdev(warm_s), 4),
            'cold_per_file_us': round(cold_s / max(len(files), 1) * 1e6, 2),
            'warm_per_file_us': round(warm_score / max(len(files), 1) * 1e6, 2),
            # Extrapolation: per-file scoring times the target, plus the query
            # constant once. The constant does not scale, so it is added, not
            # multiplied.
            f'est_cold_{target}_s': round(cold_s / max(len(files), 1) * target
                                          + statistics.mean(warm_q), 1),
            f'est_warm_{target}_s': round(warm_score / max(len(files), 1) * target
                                          + statistics.mean(warm_q), 1),
        })
        print(f"  {mode:9} query {statistics.mean(warm_q):6.3f}s | score cold "
              f"{cold_s:7.4f}s warm {warm_score:7.4f}s | "
              f"{cold_hits}/{len(files)} indexed", flush=True)
    return rows


def measure_listing(media_root, cfg, target):
    """How long it takes to find the files, in the three regimes that exist.

    Scanning is cached per directory, keyed by the directory's mtime, so there are
    three distinct costs and they differ by a lot:

      uncached  the filesystem is actually read. What you pay once per folder,
                and again whenever a folder changes.
      cold      the listing is cached but on disk; shards load into RAM.
      warm      the listing is in RAM.

    The command line pays `uncached` every run because it walks directly; the
    application goes through this cache, which is why browsing feels instant
    after the first visit. Both numbers are worth having — they are what a user
    experiences in the two different tools.
    """
    from omegaconf import OmegaConf

    from anagnorisis_core.storage.file_walker import FileWalker
    from anagnorisis_core.media.media_types import get_registry

    exts = set(get_registry(cfg).all_extensions())
    servers = OmegaConf.create([{'name': 'Local', 'url': f'osfs://{media_root}/'}])
    walker = FileWalker(servers, cfg.main.cache_path)
    root = f'osfs://{media_root}/'

    # uncached: a scratch cache directory, so this really is a first-ever scan.
    # Clearing the RAM tier of the *shared* cache is not enough — main() has
    # already called collect_files by this point, which filled the disk tier, so
    # the "uncached" row measured a cache hit and was published as if it were a
    # filesystem walk.
    # Not a TemporaryDirectory: the cache keeps a background flush thread that
    # writes after this function returns, and removing the directory under it
    # raises FileNotFoundError on a lock file at exit. A temp dir inside the
    # container costs nothing and disappears with it.
    scratch = tempfile.mkdtemp(prefix='bench-uncached-')
    scratch_walker = FileWalker(servers, scratch)
    t0 = time.perf_counter()
    files = scratch_walker.walk(root, exts)
    uncached = time.perf_counter() - t0

    # cold: the listings are now cached on disk, but not in RAM
    _clear_ram_tier(walker._fast_cache)
    t0 = time.perf_counter()
    walker.walk(root, exts)
    cold = time.perf_counter() - t0

    warm_times = []
    for _ in range(WARM_RUNS):
        t0 = time.perf_counter()
        walker.walk(root, exts)
        warm_times.append(time.perf_counter() - t0)
    warm = statistics.mean(warm_times)

    n = max(len(files), 1)
    row = {
        'files': len(files),
        'uncached_s': round(uncached, 4),
        'cold_s': round(cold, 4),
        'warm_s': round(warm, 4),
        'warm_stdev_s': round(statistics.pstdev(warm_times), 4),
        'uncached_per_file_us': round(uncached / n * 1e6, 2),
        'cold_per_file_us': round(cold / n * 1e6, 2),
        'warm_per_file_us': round(warm / n * 1e6, 2),
        f'est_uncached_{target}_s': round(uncached / n * target, 1),
        f'est_cold_{target}_s': round(cold / n * target, 1),
        f'est_warm_{target}_s': round(warm / n * target, 1),
    }
    print(f"  listing   uncached {uncached:7.3f}s  cold {cold:7.3f}s  "
          f"warm {warm:7.4f}s  ({len(files)} files)", flush=True)
    return row


def measure_quality(files, cfg, sample=25, seed=7):
    """Self-retrieval: does a file's own description find that file first?"""
    from anagnorisis_core.search.metadata_search import get_metadata_search

    search = get_metadata_search(cfg)
    rng = random.Random(seed)
    candidates = [f for f in files]
    rng.shuffle(candidates)

    results = {}
    for mode in ('metadata', 'semantic'):
        ranks = []
        checked = 0
        for fp in candidates:
            if checked >= sample:
                break
            description = search.generate_full_description(
                fp, generate_desc_if_not_in_cache=False)
            # Use the description's distinctive middle, not the filename line —
            # otherwise this measures filename matching in disguise.
            body = '\n'.join(description.split('\n')[2:])[:600].strip()
            if len(body) < 40:
                continue
            print(f'    {mode} quality sample {checked + 1}/{sample}...',
                  end='\r', flush=True)
            hits = api.search(body, files, cfg=cfg, mode=mode, limit=20)
            if not hits:
                continue
            checked += 1
            paths = [p for p, _ in hits]
            ranks.append(paths.index(fp) + 1 if fp in paths else None)

        found = [r for r in ranks if r]
        results[mode] = {
            'sampled': len(ranks),
            'recall_at_1': round(sum(1 for r in found if r == 1) / len(ranks), 3) if ranks else None,
            'recall_at_5': round(sum(1 for r in found if r <= 5) / len(ranks), 3) if ranks else None,
            'mrr': round(sum(1 / r for r in found) / len(ranks), 3) if ranks else None,
        }
        print(f"  {mode:9} recall@1 {results[mode]['recall_at_1']}  "
              f"recall@5 {results[mode]['recall_at_5']}  MRR {results[mode]['mrr']} "
              f"(n={results[mode]['sampled']})", flush=True)
    return results


def _by_media_type(files, cfg):
    """Group files by their media type, because the costs are nothing alike.

    A 4K video and a text file both count as "one file" and differ by two orders
    of magnitude in what they cost to index or describe. A single average over a
    mixed set therefore describes no real library; the per-type figures plus a
    stated mix do.
    """
    from anagnorisis_core.media.media_types import get_registry

    registry = get_registry(cfg)
    groups: dict[str, list[str]] = {}
    for fp in files:
        media_type = registry.for_file(fp)
        groups.setdefault(media_type.name if media_type else 'unknown', []).append(fp)
    return dict(sorted(groups.items()))


def _time_model_load(cfg, models_folder):
    """The one-off cost of getting the embedder onto the card.

    Kept out of the per-file numbers for the same reason the query is kept out of
    the score: it is paid once per pass, not once per file, and multiplying it by
    a file count is how a plausible estimate becomes a fantasy.
    """
    from anagnorisis_core.models.embedder import get_omni_embedder

    embedder = get_omni_embedder(cfg)
    embedder.unload()
    t0 = time.perf_counter()
    embedder.initiate(models_folder)
    load_s = time.perf_counter() - t0
    embedder.unload()
    return load_s


def measure_resume(files, cfg, target):
    """A pass over a library that is already indexed — the recurring cost.

    The scheduled background passes run this shape over and over: walk the
    library, find every embedding already cached, do nothing. It is the cost the
    app pays almost all the time, so it gets its own number.

    Nothing is subtracted for model loading here, and that is the point: with
    every file a cache hit, ``process_files`` never needs the model, so no load
    happens at all. Subtracting one anyway is how this figure first came out as a
    flat 0.000s — a load that never happened, deducted from real work.
    """
    from anagnorisis_core.progress import NullProgress

    t0 = time.perf_counter()
    api.index(files, cfg=cfg, ctx=NullProgress())
    elapsed = time.perf_counter() - t0
    n = max(len(files), 1)
    row = {
        'files': len(files),
        'seconds': round(elapsed, 3),
        'per_file_ms': round(elapsed / n * 1e3, 3),
        f'est_{target}_s': round(elapsed / n * target, 1),
    }
    print(f"  re-index (all cache hits): {elapsed:.3f}s for {n} files "
          f"({elapsed / n * 1e3:.2f} ms/file, "
          f"est. {row[f'est_{target}_s']}s @{target:,})", flush=True)
    return row


def measure_indexing(files, cfg, target, models_folder):
    """How long it takes to make a library searchable — the cheap half.

    There are two costs to getting a library searchable and they are orders of
    magnitude apart, so this harness measures them separately:

      index     embed the file's content, and embed its description *text*.
                Fractions of a second per file.
      describe  have the descriptor write that description in the first place.
                Seconds per file — see ``measure_describing``.

    ``api.index`` does the first only: it reads descriptions from the cache and
    never generates one (``generate_desc_if_not_in_cache=False``). So what is
    measured here is the cost of indexing the descriptions a library *already
    has* — which is exactly what it costs on a data-server folder that arrived
    with its ``.meta`` files, and exactly what it does not cost on a fresh
    library of undescribed files.

    Each phase pays one model load, which is measured once and subtracted, so the
    per-file figures are per-file work. Extrapolation adds it back once.
    """
    from anagnorisis_core.progress import NullProgress

    load_s = _time_model_load(cfg, models_folder)
    print(f"  embedder load (one-off, excluded below): {load_s:.2f}s", flush=True)

    groups = _by_media_type(files, cfg)
    phases = (('content', dict(content=True, metadata=False)),
              ('descriptions', dict(content=False, metadata=True)))

    per_type: dict[str, dict] = {}
    for type_name, group in groups.items():
        row = {'files': len(group)}
        for phase, flags in phases:
            t0 = time.perf_counter()
            report = api.index(group, cfg=cfg, ctx=NullProgress(), **flags)
            raw = time.perf_counter() - t0
            # Each call ends by unloading, so each paid one load. Subtract it.
            net = max(raw - load_s, 0.0)
            row[f'{phase}_raw_s'] = round(raw, 3)
            row[f'{phase}_s'] = round(net, 3)
            row[f'{phase}_per_file_s'] = round(net / max(len(group), 1), 4)
            if report.errors:
                row.setdefault('errors', []).extend(report.errors[:3])
        per_type[type_name] = row
        print(f"  {type_name:<10} {len(group):>5} files   "
              f"content {row['content_per_file_s']:.4f}s/file   "
              f"descriptions {row['descriptions_per_file_s']:.4f}s/file", flush=True)

    resume_row = measure_resume(files, cfg, target)
    resume = resume_row['seconds']
    n = max(len(files), 1)

    totals = {}
    for phase, _ in phases:
        secs = sum(r[f'{phase}_s'] for r in per_type.values())
        totals[phase] = {
            'seconds': round(secs, 2),
            'per_file_s': round(secs / n, 4),
            f'est_{target}_s': round(secs / n * target + load_s, 1),
        }

    return {
        'files': len(files),
        'model_load_s': round(load_s, 2),
        'per_type': per_type,
        'totals': totals,
        'resume_s': resume_row['seconds'],
        'resume_per_file_ms': resume_row['per_file_ms'],
        f'est_resume_{target}_s': resume_row[f'est_{target}_s'],
        f'est_total_{target}_s': round(
            sum(t['per_file_s'] for t in totals.values()) * target + 2 * load_s, 1),
    }


def measure_describing(files, cfg, target, models_folder, sample_per_type=3, seed=7):
    """The expensive half: what it costs the descriptor to write a description.

    This is the number that decides whether indexing a large library is an
    afternoon or a fortnight, and it is per media type because the spread is
    enormous — a short text file and a long video are not comparable units.

    Only a sample is measured, chosen with a fixed seed so runs are comparable.
    The descriptor is loaded once and that load is reported separately.
    """
    from anagnorisis_core.models.descriptor import OmniDescriptor
    from anagnorisis_core.media.media_types import get_registry

    registry = get_registry(cfg)
    groups = _by_media_type(files, cfg)
    rng = random.Random(seed)

    descriptor = OmniDescriptor(cfg)
    t0 = time.perf_counter()
    try:
        descriptor.initiate(models_folder)
    except Exception as exc:
        print(f"  ! descriptor unavailable: {exc}", flush=True)
        return {'unavailable': str(exc)}
    load_s = time.perf_counter() - t0
    print(f"  descriptor load (one-off): {load_s:.2f}s", flush=True)

    per_type: dict[str, dict] = {}
    try:
        for type_name, group in groups.items():
            picked = rng.sample(group, min(sample_per_type, len(group)))
            times, failures = [], 0
            for fp in picked:
                t0 = time.perf_counter()
                try:
                    text = api._describe_one(descriptor, fp, cfg, registry.for_file(fp))
                except Exception as exc:
                    print(f"    ! {os.path.basename(fp)}: {exc}", flush=True)
                    failures += 1
                    continue
                elapsed = time.perf_counter() - t0
                if text:
                    times.append(elapsed)
                else:
                    failures += 1
            if not times:
                per_type[type_name] = {'sampled': 0, 'failed': failures,
                                       'population': len(group)}
                print(f"  {type_name:<10} no successful description "
                      f"({failures} failed)", flush=True)
                continue
            mean = statistics.mean(times)
            per_type[type_name] = {
                'sampled': len(times),
                'failed': failures,
                'population': len(group),
                'mean_s': round(mean, 2),
                'median_s': round(statistics.median(times), 2),
                'min_s': round(min(times), 2),
                'max_s': round(max(times), 2),
                f'est_all_{target}_hours': round(mean * target / 3600, 1),
            }
            print(f"  {type_name:<10} {mean:6.2f}s/file "
                  f"(median {statistics.median(times):.2f}s, n={len(times)})"
                  f"   {target} files would take "
                  f"{mean * target / 3600:.1f}h", flush=True)
    finally:
        descriptor.unload()

    # A blended figure using the measured mix of this set. The mix is what varies
    # between libraries, so the per-type rows above travel and this one does not.
    weighted = sum(r['mean_s'] * r['population']
                   for r in per_type.values() if 'mean_s' in r)
    covered = sum(r['population'] for r in per_type.values() if 'mean_s' in r)
    blended = weighted / covered if covered else None

    return {
        'model_load_s': round(load_s, 2),
        'sample_per_type': sample_per_type,
        'per_type': per_type,
        'blended_per_file_s': round(blended, 2) if blended else None,
        f'est_blended_{target}_hours': (
            round(blended * target / 3600, 1) if blended else None),
        'mix_note': ('blended at this set\'s media mix: '
                     + ', '.join(f'{k} {r["population"]}'
                                 for k, r in per_type.items())),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--media', required=True,
                    help='the folder to measure against')
    # Optional, like the command line's: the package carries defaults. Pass them
    # to measure against an existing library rather than a fresh one — which
    # changes what "cold" means, so it is worth being deliberate about.
    ap.add_argument('--project-config', default=None,
                    help='folder whose cache/ is measured against')
    ap.add_argument('--models', default=None)
    ap.add_argument('--config', action='append', default=[])
    ap.add_argument('--index', action='store_true', help='build the index first')
    ap.add_argument('--target', type=int, default=100_000,
                    help='extrapolate to this many files')
    ap.add_argument('--quality-sample', type=int, default=25)
    ap.add_argument('--speed-only', action='store_true',
                    help='skip the quality pass, which is much the slower half')
    ap.add_argument('--listing-only', action='store_true',
                    help='measure only the file listing regimes, and exit')
    ap.add_argument('--resume-only', action='store_true',
                    help='measure only the re-index pass over an already indexed '
                         'library, and exit')
    ap.add_argument('--describe-sample', type=int, default=0,
                    help='files per media type to time the descriptor on '
                         '(0 = skip; this is seconds per file, so keep it small)')
    ap.add_argument('--out', default=None, help='write results as JSON here')
    args = ap.parse_args()

    cfg = load_config(args.config,
                      project_config_path=args.project_config,
                      models_path=args.models)

    # Two calls, because the first one in a fresh process is dominated by imports
    # that have nothing to do with walking: pyfilesystem, fs.osfs, the media-type
    # taxonomy. Reporting the first as "the walk" overstates it by ~30x and is the
    # same mistake as folding the query embedding into the per-file score.
    t0 = time.perf_counter()
    files = api.collect_files([args.media], cfg=cfg)
    first_call = time.perf_counter() - t0
    t0 = time.perf_counter()
    api.collect_files([args.media], cfg=cfg)
    list_time = time.perf_counter() - t0
    startup = max(first_call - list_time, 0.0)
    print(f'{len(files)} files listed in {list_time:.4f}s '
          f'({list_time / max(len(files),1) * 1e6:.1f} us/file); '
          f'first call took {first_call:.3f}s, of which ~{startup:.3f}s is '
          f'one-time imports', flush=True)

    if args.listing_only:
        print('\nlisting (the cached directory walk the app uses):', flush=True)
        listing = measure_listing(args.media, cfg, args.target)
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump({'listing': listing, 'target': args.target,
                           'list_seconds': round(list_time, 4),
                           'list_per_file_us': round(
                               list_time / max(len(files), 1) * 1e6, 1),
                           'first_call_seconds': round(first_call, 4),
                           'startup_seconds': round(startup, 4)}, fh, indent=2)
            print(f'wrote {args.out}')
        return

    if args.resume_only:
        print('\nre-index over an already indexed library:', flush=True)
        row = measure_resume(files, cfg, args.target)
        if args.out:
            with open(args.out, 'w') as fh:
                json.dump({'resume': row, 'target': args.target}, fh, indent=2)
            print(f'wrote {args.out}')
        return

    indexing = {}
    if args.index:
        print('\nindexing (embedding content and description text):', flush=True)
        indexing = measure_indexing(files, cfg, args.target,
                                    cfg.main.embedding_models_path)

    describing = {}
    if args.describe_sample:
        print('\ndescribing (the descriptor writing a description, sampled):',
              flush=True)
        describing = measure_describing(files, cfg, args.target,
                                       cfg.main.embedding_models_path,
                                       sample_per_type=args.describe_sample)

    print('\nlisting (the cached directory walk the app uses):', flush=True)
    listing = measure_listing(args.media, cfg, args.target)

    print('\nspeed:', flush=True)
    speed = measure_speed(files, cfg, args.target)
    quality = {}
    if args.speed_only:
        print('\nquality: skipped (--speed-only)', flush=True)
    else:
        print('\nquality (self-retrieval):', flush=True)
        quality = measure_quality(files, cfg, sample=args.quality_sample)

    payload = {
        'files_measured': len(files),
        'list_seconds': round(list_time, 4),
        'list_per_file_us': round(list_time / max(len(files), 1) * 1e6, 1),
        'first_call_seconds': round(first_call, 4),
        'startup_seconds': round(startup, 4),
        'listing': listing,
        'indexing': indexing,
        'describing': describing,
        'target': args.target,
        'warm_runs': WARM_RUNS,
        'queries': QUERIES,
        'speed': speed,
        'quality': quality,
    }
    if args.out:
        with open(args.out, 'w') as fh:
            json.dump(payload, fh, indent=2)
        print(f'\nwrote {args.out}')
    else:
        print('\n' + json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
