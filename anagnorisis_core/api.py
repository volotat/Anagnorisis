"""What the engine can do, as plain functions.

Three callers use these: the Flask application, the ``anagnorisis`` command, and
the data server. They call the *same* functions — the
command line is argparse over this module, not a reimplementation — so the CLI
cannot drift from what the app does.

Nothing here opens a database or a socket. Ratings live in the application and
are passed in; progress is reported through the two-method context described in
:mod:`anagnorisis_core.progress`.
"""
import contextlib
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

from anagnorisis_core.media import description, models
from anagnorisis_core.storage import virtual_file_system as vfs
from anagnorisis_core.models.descriptor import OmniDescriptor
from anagnorisis_core.models.embedder import get_omni_embedder
from anagnorisis_core.media.media_types import get_registry
from anagnorisis_core.progress import NullProgress
from anagnorisis_core.storage.sinks import Sink


@dataclass
class DescribeReport:
    considered: int = 0
    already_done: int = 0
    written: int = 0
    failed: int = 0
    skipped_unstable: int = 0
    skipped_unsupported: int = 0
    errors: list = field(default_factory=list)

    def __str__(self):
        return (f"{self.written} written, {self.already_done} already had one, "
                f"{self.failed} failed, {self.skipped_unstable} still changing, "
                f"{self.skipped_unsupported} of a type nothing can describe "
                f"(of {self.considered} considered)")


def collect_files(paths: Iterable[str], *, cfg, recursive: bool = True) -> list[str]:
    """Every file under *paths* whose extension belongs to a known media type.

    Returns full VFS URLs (``osfs:///…``), which is what everything downstream
    now speaks: one spelling for a file whether it came from the command line, the
    application or a remote server.

    Goes through the shared directory-listing cache rather than walking the tree
    directly. The listing of each directory is cached and keyed by that
    directory's modification time, so a folder is re-read only when it actually
    changes. On the measured demo set that is ~8 µs per file against ~815 µs for a
    raw walk — for a 100k-file library, about a second instead of a minute and a
    half, on every single command.

    Hidden directories are skipped, so version-control folders, thumbnail caches
    and trash never get indexed or annotated.

    Local paths only. Describing a file means reading it, and reading someone
    else's file across a network is not something a sweep should decide to do.
    """
    from omegaconf import OmegaConf

    from anagnorisis_core.storage.file_walker import get_file_walker

    registry = get_registry(cfg)
    known = set(registry.all_extensions())
    cache_path = OmegaConf.select(cfg, 'main.cache_path', default=None)

    roots: list[str] = []
    single_files: list[str] = []
    for raw in paths:
        if not vfs.is_local_url(raw):
            # Refused rather than resolved. Resolving a remote URL here would
            # download it, which is precisely the automatic fetching this project
            # does not do — and doing it while merely *listing* files would be
            # the most surprising place of all to do it.
            raise ValueError(
                f'{raw!r} is on a remote server. This works on local paths only; '
                f'remote files are reached through the application, which reads '
                f'their .meta sidecars instead of downloading them.')
        local = vfs.resolve_to_local_path(raw)[0] if '://' in raw else raw
        local = os.path.abspath(local)
        if os.path.isfile(local):
            if os.path.splitext(local)[1].lower() in known:
                single_files.append(f'osfs://{local}')
        else:
            roots.append(local)

    found: list[str] = list(single_files)
    if roots:
        servers = OmegaConf.create(
            [{'name': os.path.basename(r) or 'root', 'url': f'osfs://{r}/'}
             for r in roots])
        walker = get_file_walker(servers, cache_path or os.path.join(roots[0], '.anagnorisis-cache'))
        for root in roots:
            root_url = f'osfs://{root}/'
            if recursive:
                found.extend(walker.walk(root_url, known))
            else:
                files, _subdirs = walker.list_dir(root_url, known)
                found.extend(files)

    # Sorted so a run is predictable and interruption resumes at the same place.
    return sorted(set(found))


def _is_stable(file_path: str, seen: dict) -> bool:
    """True once a file has stopped changing between passes.

    A daemon watching a folder people drop files into will meet half-copied
    ones. Describing a truncated video would be permanent, because a `.meta` is
    never overwritten — so wait until size and mtime hold still.
    """
    try:
        local = vfs.resolve_to_local_path(file_path)[0] if '://' in file_path else file_path
        st = os.stat(local)
    except OSError:
        return False
    signature = (st.st_size, st.st_mtime_ns)
    previous = seen.get(file_path)
    seen[file_path] = signature
    return previous == signature


def _describe_one(descriptor, file_path: str, cfg, media_type) -> str:
    method_name = models.describe_method_for(
        media_type.name if media_type else None)
    if method_name is None:
        return ''
    if method_name == 'describe_text':
        content = description.read_text_content(file_path)
        # Short text is its own description, so there is nothing to generate.
        # Checked here as well as in build_description because otherwise this
        # would spend a model run on a paragraph and the assembled description
        # would then ignore the result — the expensive half of a decision the
        # cheap half had already made.
        body, summarised = description.body_for_text(
            content, cfg=cfg,
            summarise=lambda text: getattr(descriptor, method_name)(text))
        return body if summarised else content.strip()
    method = getattr(descriptor, method_name)
    return (method(file_path) or '').strip()


class _NoDescriptorNeeded:
    """Stands in for the descriptor on a batch that cannot possibly need it.

    Reaching for an attribute here means the size check in
    :func:`_may_need_generation` was wrong, which should be impossible: it only
    answers "no" when the file is smaller in *bytes* than the verbatim limit is
    in characters, and UTF-8 never uses fewer bytes than characters. Failing
    loudly beats loading a model nobody expected to load.
    """

    def __getattr__(self, name):
        raise RuntimeError(
            f"descriptor.{name} was called for a batch judged not to need it")


def _write_described(described, sink, report, ctx) -> None:
    """Persist finished descriptions. Assumes no model is resident."""
    for fp, text in described:
        ctx.check()
        try:
            if sink.write(fp, text):
                report.written += 1
            else:
                report.failed += 1
        except Exception as exc:
            report.failed += 1
            report.errors.append(f'{fp}: {exc}')


def _may_need_generation(file_path: str, cfg, media_type) -> bool:
    """Whether describing this file could actually reach the model.

    Loading the descriptor costs about 13 seconds, and two kinds of file never
    call it: types with no describe method at all (a PDF today), and short text,
    which is kept verbatim because the words are already its best description.

    Deliberately biased towards True. A wrong True costs the load we pay today
    anyway; a wrong False would silently skip a description.
    """
    method_name = models.describe_method_for(
        media_type.name if media_type else None)
    if method_name is None:
        return False
    if method_name != 'describe_text':
        return True

    # Size in bytes is never smaller than the decoded character count, so a
    # file under the limit is certainly kept verbatim. Read nothing: this runs
    # over the whole batch, and remote files must not be fetched to answer it.
    if not vfs.is_local_url(file_path):
        return True
    try:
        # Safe for local URLs: this only strips the scheme and never downloads.
        local_path, _temp = vfs.resolve_to_local_path(file_path)
        return os.path.getsize(local_path) >= description.text_verbatim_limit(cfg)
    except Exception:
        return True


def describe(
    paths: Iterable[str],
    *,
    cfg,
    sink: Sink,
    ctx=None,
    recursive: bool = True,
    batch_size: int = 100,
    models_folder: Optional[str] = None,
    require_stable: bool = False,
    _stability_state: Optional[dict] = None,
) -> DescribeReport:
    """Describe files that have no description yet, and store it in *sink*.

    Runs in two phases per batch, because the embedder and the descriptor cannot
    both be resident on an 8 GB card: embed the batch (which is what gives the
    tags and fingerprint something to be derived from), release that model, then
    describe the batch, then release that one. Assembling the text afterwards
    loads nothing at all — the proxy is rebuilt from the cache.
    """
    ctx = ctx or NullProgress()
    models_folder = models_folder or cfg.main.embedding_models_path
    report = DescribeReport()
    registry = get_registry(cfg)

    files = collect_files(paths, cfg=cfg, recursive=recursive)
    report.considered = len(files)

    todo = []
    for fp in files:
        if sink.is_done(fp):
            report.already_done += 1
            continue
        if require_stable and not _is_stable(fp, _stability_state if _stability_state is not None else {}):
            report.skipped_unstable += 1
            continue
        # A type with no describe method can never produce anything. Dropping
        # it here rather than failing it per file keeps a folder of PDFs from
        # loading a 5.5B model once per batch to write nothing, and stops them
        # being counted as failures when nothing was ever attempted.
        media_type = registry.for_file(fp)
        if models.describe_method_for(
                media_type.name if media_type else None) is None:
            report.skipped_unsupported += 1
            continue
        todo.append(fp)

    if not todo:
        ctx.update(1.0, 'Nothing to describe.')
        return report

    embedder = get_omni_embedder(cfg)
    descriptor = OmniDescriptor(cfg)

    # Tag vocabularies first, while nothing large is resident.
    _warm_tag_vocabularies(todo, cfg=cfg, registry=registry, ctx=ctx)

    # batch_size 0 means "do not interleave": embed everything, then describe
    # everything. That is the fewest possible model switches — two loads for the
    # whole run — at the cost of producing no finished .meta files until the
    # describing pass starts.
    stride = batch_size if batch_size and batch_size > 0 else len(todo)

    for start in range(0, len(todo), stride):
        batch = todo[start:start + stride]
        ctx.check()

        # ---- phase 1: content embeddings, so tags and fingerprint exist ----
        ctx.update(start / len(todo), f'Embedding {len(batch)} files...')
        try:
            _embed_batch(batch, cfg=cfg, registry=registry, ctx=ctx)
        except Exception as exc:
            report.errors.append(f'embedding batch failed: {exc}')
        finally:
            embedder.unload()   # the descriptor needs almost the whole card

        # ---- phase 2: descriptions. Collect them; write nothing yet. ----
        # Writing here would assemble the full description while the descriptor is
        # still resident, and assembly can touch the embedding caches — so the two
        # large models would overlap for no reason.
        described: list[tuple[str, str]] = []

        # A batch of short text files is described from its own words and never
        # reaches the model, so loading 13 seconds of descriptor for it is pure
        # cost. Assemble those directly and skip the load when nothing is left.
        needs_model = [fp for fp in batch
                       if _may_need_generation(fp, cfg, registry.for_file(fp))]
        if not needs_model:
            for fp in batch:
                try:
                    text = _describe_one(_NoDescriptorNeeded(), fp, cfg,
                                         registry.for_file(fp))
                except Exception as exc:
                    report.errors.append(f'{fp}: {exc}')
                    report.failed += 1
                    continue
                if text:
                    described.append((fp, text))
                else:
                    report.failed += 1
            _write_described(described, sink, report, ctx)
            continue

        try:
            descriptor.initiate(models_folder)
        except Exception as exc:
            # initiate() spawns the worker before it can fail — the VRAM guard,
            # for instance, raises from inside the worker, which then keeps
            # looping on its input queue. Unload explicitly: the `finally` below
            # belongs to the block this `continue` skips, so without this the
            # worker survives holding VRAM and the process never exits.
            descriptor.unload()
            report.errors.append(f'descriptor unavailable: {exc}')
            report.failed += len(batch)
            continue

        try:
            for i, fp in enumerate(batch):
                ctx.check()
                ctx.update((start + i) / len(todo), f'Describing {os.path.basename(fp)}')
                try:
                    text = _describe_one(descriptor, fp, cfg, registry.for_file(fp))
                    if text:
                        described.append((fp, text))
                    else:
                        report.failed += 1
                except Exception as exc:
                    report.failed += 1
                    report.errors.append(f'{fp}: {exc}')
        finally:
            descriptor.unload()

        # ---- phase 3: assemble and write, with no model resident ----
        _write_described(described, sink, report, ctx)

    ctx.update(1.0, str(report))
    return report


def train_evaluator(*, cfg, ctx=None, max_steps=None, time_budget_seconds=None):
    """Train the evaluator on the memory files, and return where it was saved.

    Needs no database. Every memory file carries its rating on line 1 and the
    description below it, which is the whole training pair — so this reads a
    folder and writes a model, and nothing in between has to know about the
    application.

    The rating line is stripped before the description is embedded. That is not
    tidiness: leaving it in would teach the model to read the score off the page
    instead of judging the file.
    """
    from anagnorisis_core.ratings.training import train_universal_evaluator

    ctx = ctx or NullProgress()

    # The trainer's callback is (message, percent, *accuracies) — it reports a
    # percentage and, later in training, baseline and train/test accuracy. Accept
    # them all and translate to the engine's 0..1 progress, rather than making
    # the trainer speak a new dialect.
    def report(message, percent=0, *accuracies):
        ctx.check()
        try:
            fraction = max(0.0, min(1.0, float(percent) / 100.0))
        except (TypeError, ValueError):
            fraction = 0.0
        if accuracies:
            shown = ' '.join(f'{a:.3f}' for a in accuracies
                             if isinstance(a, (int, float)))
            message = f'{message} [{shown}]' if shown else message
        ctx.update(fraction, str(message))

    return train_universal_evaluator(
        cfg, callback=report, max_steps=max_steps,
        time_budget_seconds=time_budget_seconds)


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

@dataclass
class IndexReport:
    considered: int = 0
    content_embedded: int = 0
    descriptions_embedded: int = 0
    errors: list = field(default_factory=list)

    def __str__(self):
        return (f"{self.content_embedded} content embeddings, "
                f"{self.descriptions_embedded} description embeddings "
                f"(of {self.considered} files)")


def index(paths, *, cfg, ctx=None, recursive=True, batch_size=200,
          content=True, metadata=True) -> IndexReport:
    """Embed files so they can be searched, and cache the result.

    Two indexes, because there are two ways to search. *content* embeds the file
    itself, which is what semantic search compares against. *metadata* embeds the
    file's text description, which is what metadata search compares against and
    what the evaluator scores.

    **Safe to interrupt.** Every embedding is written to the cache as it is
    produced and keyed by the file and the model that made it, so stopping this
    halfway loses only the batch in flight. Run it again and it resumes: files
    already embedded are skipped, because a cache hit is what "already done"
    means here. Nothing is left in a half-written state — there is no index file
    to corrupt, only per-file entries that either exist or do not.
    """
    from anagnorisis_core.search.content_search import get_content_search
    from anagnorisis_core.search.metadata_search import get_metadata_search

    ctx = ctx or NullProgress()
    report = IndexReport()
    registry = get_registry(cfg)
    files = collect_files(paths, cfg=cfg, recursive=recursive)
    report.considered = len(files)
    if not files:
        ctx.update(1.0, 'Nothing to index.')
        return report

    if content:
        by_type: dict[str, list[str]] = {}
        for fp in files:
            media_type = registry.for_file(fp)
            if media_type and models.has_content_embedding(media_type.name):
                by_type.setdefault(media_type.name, []).append(fp)

        for type_name, group in by_type.items():
            engine = get_content_search(cfg, type_name)
            # No initiate(): see _embed_batch. process_files loads on demand.
            for start in range(0, len(group), batch_size):
                ctx.check()
                batch = group[start:start + batch_size]
                ctx.update(start / max(len(group), 1),
                           f'Embedding {type_name} content {start}/{len(group)}')
                try:
                    engine.process_files(batch, generate_embs_if_not_in_cache=True)
                    report.content_embedded += len(batch)
                except Exception as exc:
                    report.errors.append(f'{type_name} content batch: {exc}')
        # No unload here. Both passes use the same embedder singleton and the
        # same weights, and the descriptor is never resident during index, so
        # dropping it between them bought nothing and cost a 12.3s reload.

    if metadata:
        search = get_metadata_search(cfg)
        # The descriptions pass reads tag vocabularies through the proxy. If
        # they have not been embedded yet, the fallback path embeds them one
        # tag at a time on the CPU, about 1900 per media type. describe() warms
        # them up front for exactly this reason; index() needs it too.
        _warm_tag_vocabularies(files, cfg=cfg, registry=registry, ctx=ctx)
        for start in range(0, len(files), batch_size):
            ctx.check()
            batch = files[start:start + batch_size]
            ctx.update(start / max(len(files), 1),
                       f'Embedding descriptions {start}/{len(files)}')
            try:
                search.process_files(batch, generate_embs_if_not_in_cache=True)
                report.descriptions_embedded += len(batch)
            except Exception as exc:
                report.errors.append(f'description batch: {exc}')

    # One unload for the whole run, whichever passes ran.
    if content or metadata:
        get_omni_embedder(cfg).unload()

    ctx.update(1.0, str(report))
    return report


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------

SEARCH_MODES = ('name', 'semantic', 'metadata')


def search(query: str, paths, *, cfg, mode: str = 'metadata', limit: int = 20,
           recursive: bool = True, ctx=None) -> list[tuple[str, float]]:
    """Rank files against *query*, best first.

    Three modes, because they answer different questions:

    * ``name`` — fuzzy match on the filename and path. No model, no index, works
      on anything immediately. Use it when you remember what the file is called.
    * ``semantic`` — compares the query against the embedding of the file's own
      content. Finds a photo of a beach whatever it is named. Requires the file
      to have been indexed with ``index(content=True)``.
    * ``metadata`` — compares the query against the embedding of the file's
      *description*: its tags, its fingerprint, the model's sentences about it,
      its internal metadata and its ``.meta`` sidecar. This is the mode that can
      match on things a picture cannot show — a filename you gave it, a note you
      wrote in a sidecar.

    Files that have not been indexed score NaN and are dropped rather than
    ranked at the bottom: an unindexed file is unknown, not irrelevant, and
    sorting zeros to the end would quietly present them as bad matches.

    The query is embedded on the CPU, in-process. Searching never wakes the GPU.
    """
    ctx = ctx or NullProgress()
    if mode not in SEARCH_MODES:
        raise ValueError(f'unknown search mode {mode!r}; expected one of {SEARCH_MODES}')

    files = collect_files(paths, cfg=cfg, recursive=recursive)
    if not files:
        return []

    if mode == 'name':
        scored = [(fp, _name_score(query, fp)) for fp in files]
    elif mode == 'semantic':
        scored = _semantic_scores(query, files, cfg=cfg, ctx=ctx)
    else:
        scored = _metadata_scores(query, files, cfg=cfg, ctx=ctx)

    ranked = [(fp, float(sc)) for fp, sc in scored
              if sc is not None and not math.isnan(sc)]
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    return ranked[:limit]


def _name_score(query: str, file_path: str) -> float:
    """How well a filename matches, from 0 to 1.

    A substring hit outranks a fuzzy one, and a hit in the filename outranks a
    hit further up the path — matching how people actually remember files.
    """
    import difflib

    q = query.lower().strip()
    name = os.path.basename(file_path).lower()
    if not q:
        return 0.0
    if q in name:
        return 1.0
    if q in file_path.lower():
        return 0.85
    return difflib.SequenceMatcher(None, q, name).ratio() * 0.8


def _semantic_scores(query, files, *, cfg, ctx):
    from anagnorisis_core.search.content_search import get_content_search

    registry = get_registry(cfg)
    scores: list[tuple[str, float]] = []
    by_type: dict[str, list[str]] = {}
    for fp in files:
        media_type = registry.for_file(fp)
        if media_type and models.has_content_embedding(media_type.name):
            by_type.setdefault(media_type.name, []).append(fp)

    # Embed the query once, not once per media type. Every type shares the same
    # model and therefore the same vector space, so a second embedding of the same
    # text would be an identical vector bought at full price — and on the CPU,
    # where search deliberately runs, that price is most of the search.
    query_vec = None

    for type_name, group in by_type.items():
        engine = get_content_search(cfg, type_name)
        ctx.update(0.0, f'Searching {len(group)} {type_name} files')
        if query_vec is None:
            query_vec = engine.process_text(query)
        # generate_embs_if_not_in_cache=False keeps searching off the GPU: an
        # unindexed file is skipped, never embedded on the request.
        embeddings = engine.process_files(group, generate_embs_if_not_in_cache=False)
        for fp, score in zip(group, engine.compare(embeddings, query_vec)):
            scores.append((fp, score))
    return scores


def _metadata_scores(query, files, *, cfg, ctx):
    from anagnorisis_core.search.metadata_search import get_metadata_search

    search_engine = get_metadata_search(cfg)
    ctx.update(0.0, f'Searching {len(files)} descriptions')
    query_vec = search_engine.process_query(query)
    embeddings = search_engine.process_files(
        files, generate_embs_if_not_in_cache=False)
    return list(zip(files, search_engine.compare(embeddings, query_vec)))


# ---------------------------------------------------------------------------
# Ratings
# ---------------------------------------------------------------------------

def rate(file_path: str, rating: float, *, cfg, memory_dir: str,
         describe_now: bool = False, models_folder=None, ctx=None) -> str:
    """Record what you think of a file. Returns the memory file written.

    With *describe_now* the descriptor runs so the memory holds the model's
    sentences as well as the tags and metadata. Without it the memory is written
    from what is already known, which is instant.
    """
    from anagnorisis_core.models.descriptor import OmniDescriptor
    from anagnorisis_core.ratings.memory import save_rating

    ctx = ctx or NullProgress()
    describe = None
    descriptor = None
    if describe_now:
        descriptor = OmniDescriptor(cfg)
        descriptor.initiate(models_folder or cfg.main.embedding_models_path)
        registry = get_registry(cfg)

        def describe(fp):
            return _describe_one(descriptor, fp, cfg, registry.for_file(fp))

    try:
        ctx.update(0.5, f'Writing memory for {os.path.basename(file_path)}')
        written = save_rating(file_path, rating, cfg=cfg,
                              memory_dir=memory_dir, describe=describe)
    finally:
        if descriptor is not None:
            descriptor.unload()
    ctx.update(1.0, f'Wrote {written}')
    return written


def rank_by_rating(paths, *, cfg, memory_dir: str, predicted: bool = False,
                   limit=None, recursive: bool = True, ctx=None,
                   models_folder=None) -> list[tuple[str, float, str]]:
    """Order files by rating, best first.

    Returns ``(path, rating, source)`` where *source* is ``'user'`` for a rating
    you gave and ``'model'`` for one the evaluator predicted.

    Your own ratings always win where they exist — the model is imitating you, so
    where you have spoken there is nothing to predict. With *predicted* the
    evaluator fills in the rest; without it, unrated files are simply left out.
    """
    from anagnorisis_core.ratings.memory import load_ratings
    from anagnorisis_core.storage.soft_hash import get_file_soft_hash

    ctx = ctx or NullProgress()
    files = collect_files(paths, cfg=cfg, recursive=recursive)
    user_ratings = load_ratings(memory_dir)

    rated: list[tuple[str, float, str]] = []
    unrated: list[str] = []
    for i, fp in enumerate(files):
        ctx.check()
        ctx.update(i / max(len(files), 1), f'Reading ratings {i}/{len(files)}')
        try:
            digest = get_file_soft_hash(fp)
        except Exception:
            digest = None
        if digest and digest in user_ratings:
            rated.append((fp, user_ratings[digest], 'user'))
        else:
            unrated.append(fp)

    if predicted and unrated:
        for fp, score in _predict_ratings(unrated, cfg=cfg, ctx=ctx,
                                          models_folder=models_folder):
            rated.append((fp, score, 'model'))

    rated.sort(key=lambda row: row[1], reverse=True)
    return rated[:limit] if limit else rated


@contextlib.contextmanager
def _summariser_for(text: str, *, cfg, models_folder=None, ctx=None):
    """A summarise callable for text that is too long to use as it stands.

    Yields None for short text, which is the common case and costs nothing —
    no model is loaded and none is unloaded. For long text it loads the
    descriptor, yields a callable, and releases it afterwards however the caller
    leaves.

    One implementation because rating and scoring must agree: if a passage is
    stored summarised, it has to be scored summarised too, or the number would
    describe text the model was never taught.
    """
    from anagnorisis_core.media import description
    from anagnorisis_core.models.descriptor import OmniDescriptor

    if len(text) <= description.text_verbatim_limit(cfg):
        yield None
        return

    if ctx is not None:
        ctx.update(0.3, 'Text is long; summarising it first')
    descriptor = OmniDescriptor(cfg)
    descriptor.initiate(models_folder or cfg.main.embedding_models_path)
    try:
        yield lambda content: descriptor.describe_text(content)
    finally:
        descriptor.unload()


def _load_evaluator(cfg):
    """The trained evaluator, or a message naming what to run."""
    from anagnorisis_core.models.universal_evaluator import UniversalEvaluator

    model_path = os.path.join(
        cfg.main.get('personal_models_path', cfg.main.embedding_models_path),
        'universal_evaluator.pt')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f'no trained evaluator at {model_path}. Run `anagnorisis train` first.')
    evaluator = UniversalEvaluator()
    evaluator.load(model_path)
    return evaluator


def score_text(text: str, *, cfg, models_folder=None, ctx=None) -> float:
    """What the evaluator predicts you would rate this text.

    Scored exactly as a file's description would be — same embedder, same
    document tower, same model — so the number is comparable with the ones
    :func:`score_file` and ``sort --predicted`` give. That comparability is the
    whole reason this is worth having: a way to ask the model what it has
    learned, in your own words.

    Short text is scored as it stands; long text is summarised first, by the same
    rule that decides what gets *stored* when you rate it. Scoring the raw text
    while rating stores a summary would answer a question about something the
    model was never taught.

    Embedded as a *document*, not a query. The embedder is asymmetric and the
    evaluator was trained on document vectors; the query tower would produce a
    number from a different space — quietly wrong rather than obviously so.
    """
    import numpy as np

    from anagnorisis_core.media import description
    from anagnorisis_core.models.embedder import get_omni_embedder

    raw = (text or '').strip()
    if not raw:
        raise ValueError('nothing to score')

    evaluator = _load_evaluator(cfg)
    with _summariser_for(raw, cfg=cfg, models_folder=models_folder,
                         ctx=ctx) as summarise:
        body, _ = description.body_for_text(raw, cfg=cfg, summarise=summarise)
    if not body:
        raise RuntimeError('nothing left to score after summarising')

    vectors = get_omni_embedder(cfg).embed_long_text(body)
    if vectors is None or len(vectors) == 0:
        raise RuntimeError('the embedder returned nothing for that text')
    scores = evaluator.predict([np.array(vectors, dtype=np.float32)])
    return float(scores[0])


def score_file(file_path: str, *, cfg, ctx=None) -> float:
    """What the evaluator predicts you would rate this file.

    Scores the file's assembled description — its tags, fingerprint, metadata and
    whatever the descriptor has already written — using only what is cached, so
    this loads no descriptor and starts no generation. A file that has never been
    described is scored from what is known about it, which is thin but honest.
    """
    ctx = ctx or NullProgress()
    scored = _predict_ratings([file_path], cfg=cfg, ctx=ctx)
    if not scored:
        raise RuntimeError(
            f'nothing to score for {file_path} — it has no description yet. '
            f'Run `anagnorisis describe` or `index` over it first.')
    return scored[0][1]


def rate_text(text: str, rating: float, *, cfg, memory_dir: str,
              models_folder=None, when=None, ctx=None) -> str:
    """Record a rating for a piece of text. Returns the memory file written.

    The text counterpart of :func:`rate`, and deliberately the same thing: a
    memory file with no file behind it. Training cannot tell them apart, which is
    the point — anything you can describe, you can rate.

    Short text is stored as it stands; long text is summarised first, exactly as
    a long .txt would be. Only that second case loads the descriptor, so rating a
    sentence costs nothing and rating a novel costs one model run.
    """
    from anagnorisis_core.ratings.memory import save_text_rating

    ctx = ctx or NullProgress()
    body = (text or '').strip()
    if not body:
        raise ValueError('refusing to rate empty text')

    with _summariser_for(body, cfg=cfg, models_folder=models_folder,
                         ctx=ctx) as summarise:
        written = save_text_rating(body, rating, memory_dir=memory_dir,
                                   cfg=cfg, summarise=summarise, when=when)
    ctx.update(1.0, f'Wrote {written}')
    return written


def _predict_ratings(files, *, cfg, ctx, models_folder=None):
    """Score files with the trained evaluator, from their descriptions."""
    import numpy as np

    from anagnorisis_core.search.metadata_search import get_metadata_search

    search_engine = get_metadata_search(cfg)
    evaluator = _load_evaluator(cfg)

    embeddings = []
    kept = []
    for i, fp in enumerate(files):
        ctx.check()
        ctx.update(i / max(len(files), 1), f'Scoring {i}/{len(files)}')
        text = search_engine.generate_full_description(
            fp, generate_desc_if_not_in_cache=False)
        vectors = search_engine.embedder.embed_long_text(text)
        if vectors is None or len(vectors) == 0:
            continue
        embeddings.append(np.array(vectors, dtype=np.float32))
        kept.append(fp)

    if not embeddings:
        return []
    scores = evaluator.predict(embeddings)
    return list(zip(kept, (float(s) for s in scores)))



def _warm_tag_vocabularies(files, *, cfg, registry, ctx) -> None:
    """Embed the tag vocabularies for the media types we are about to touch.

    Done once, up front, and **in batches on the GPU** — the embedder is about to
    be loaded for the content pass anyway, so the tags ride along with it. Left to
    the lazy path these go one at a time through the CPU query embedder, which is
    correct on the search path and around a hundred times slower here: a
    two-thousand-tag vocabulary takes minutes rather than seconds, and it happens
    while the descriptor is resident.
    """
    embedder = get_omni_embedder(cfg)

    def embed_batch(tags):
        return embedder.embed_documents(tags)

    seen = set()
    for fp in files:
        media_type = registry.for_file(fp)
        if media_type is None or media_type.name in seen:
            continue
        seen.add(media_type.name)
        proxy = models.get_proxy(cfg, media_type)
        if proxy is None:
            continue
        ctx.update(0.0, f'Preparing {media_type.name} tag vocabulary...')
        try:
            if not proxy.ensure_tag_embeddings(embed_batch=embed_batch):
                # Fall back to the per-tag CPU route rather than leaving a media
                # type with no tags at all.
                proxy.ensure_tag_embeddings()
        except Exception as exc:
            print(f'[describe] Could not prepare {media_type.name} tags: {exc}')


def _embed_batch(batch, *, cfg, registry, ctx):
    """Fill the content-embedding cache for a batch, grouped by media type."""
    from anagnorisis_core.search.content_search import get_content_search

    by_type: dict[str, list[str]] = {}
    for fp in batch:
        media_type = registry.for_file(fp)
        if media_type is None or not models.has_content_embedding(media_type.name):
            continue
        by_type.setdefault(media_type.name, []).append(fp)

    for type_name, files in by_type.items():
        engine = get_content_search(cfg, type_name)
        # Deliberately no initiate() here. That call loads the model, reads its
        # identity and then *unloads* it again — which is right at app startup and
        # actively wasteful before doing work: it costs a load, an unload, and
        # then a reload on the first embed. process_files loads on demand, and the
        # model's identity is now readable from the weights on disk without
        # loading anything.
        engine.process_files(files, generate_embs_if_not_in_cache=True)


def watch(
    paths: Iterable[str],
    *,
    cfg,
    sink: Sink,
    interval_seconds: int = 600,
    ctx=None,
    recursive: bool = True,
    batch_size: int = 100,
    models_folder: Optional[str] = None,
    max_passes: Optional[int] = None,
):
    """Describe, then keep describing whatever appears.

    New files are picked up on the next pass. A file that moved is described
    again, and its old sidecar is left where it is: matching them up by content
    would save some GPU time at the risk of attaching the wrong description to
    the wrong file, which nothing would ever correct.
    """
    stability: dict = {}
    passes = 0
    while max_passes is None or passes < max_passes:
        report = describe(paths, cfg=cfg, sink=sink, ctx=ctx, recursive=recursive,
                          batch_size=batch_size, models_folder=models_folder,
                          require_stable=True, _stability_state=stability)
        print(f"[anagnorisis] pass {passes + 1}: {report}")
        passes += 1
        if max_passes is not None and passes >= max_passes:
            break
        time.sleep(interval_seconds)
