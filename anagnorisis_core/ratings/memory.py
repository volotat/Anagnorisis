"""Remembering what you thought of a file.

Rating something writes a small Markdown file into
``<memory>/<YYYY-MM-DD>/<soft_hash>.md``. It holds the rating on line 1 and the
file's description underneath — and that description is byte-for-byte the one
metadata search indexes, so the evaluator later trains on exactly the text it
will be asked to judge.

**Why a copy of the description rather than a reference to the file.** The file
is not a reliable place to keep this. It gets renamed, reorganised, moved to
another drive, deleted; if it lives on someone else's server it can vanish
without warning. The judgement — *this kind of thing is an 8 to me* — is the part
worth keeping, and it stays valid whether or not the original is still reachable.
So memory files are an archive, not a cache: nothing regenerates them.

The rating lives on line 1 and **only** line 1, because training strips that line
before embedding the rest. Move it anywhere else and the model learns to read the
score off the page instead of judging the file.
"""
import datetime
import os
import tempfile
from typing import Optional

import xxhash

from anagnorisis_core.media import description
from anagnorisis_core.media.media_types import get_registry
from anagnorisis_core.storage.soft_hash import SOFT_HASH_ALGORITHM, get_file_soft_hash


def memory_file_path(memory_dir: str, soft_hash: str, when=None) -> str:
    """Where today's memory for *soft_hash* goes."""
    day = (when or datetime.date.today()).isoformat()
    folder = os.path.join(memory_dir, day)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f'{soft_hash}.md')


def find_memory_files(memory_dir: str, soft_hash: str) -> list[str]:
    """Every memory file for a soft hash, oldest dated folder first.

    A file can be rated more than once. Each rating writes into its own dated
    folder, so the history is kept and the most recent one wins.
    """
    found = []
    if not os.path.isdir(memory_dir):
        return found
    for day in sorted(os.listdir(memory_dir)):
        candidate = os.path.join(memory_dir, day, f'{soft_hash}.md')
        if os.path.isfile(candidate):
            found.append(candidate)
    return found


def read_memory(path: str) -> tuple[Optional[float], str]:
    """Parse a memory file into (rating, description)."""
    from anagnorisis_core.ratings.training import _parse_memory_file
    with open(path, 'r', encoding='utf-8') as fh:
        return _parse_memory_file(fh.read())


def load_ratings(memory_dir: str) -> dict[str, float]:
    """Every rating you have given, by soft hash, latest wins.

    This is how a stateless caller learns your ratings: they are in the memory
    files, not only in the application's database.
    """
    ratings: dict[str, float] = {}
    if not os.path.isdir(memory_dir):
        return ratings
    for day in sorted(os.listdir(memory_dir)):       # sorted → latest wins
        day_dir = os.path.join(memory_dir, day)
        if not os.path.isdir(day_dir):
            continue
        for name in sorted(os.listdir(day_dir)):
            if not name.endswith('.md'):
                continue
            rating, _ = read_memory(os.path.join(day_dir, name))
            if rating is not None:
                ratings[os.path.splitext(name)[0]] = rating
    return ratings


def save_rating(file_path: str, rating: float, *, cfg, memory_dir: str,
                describe=None, soft_hash: Optional[str] = None,
                when=None) -> str:
    """Record a rating for a file and return the memory file written.

    *describe* is the same injected callable the description builder takes, so
    the caller decides whether to spend model time here. Without it the memory
    still holds the file's tags, metadata and sidecar — everything except the
    model's sentences.

    Written atomically: a half-written memory file would be a permanent, silently
    corrupt training example.
    """
    soft_hash = soft_hash or get_file_soft_hash(file_path)
    media_type = get_registry(cfg).for_file(file_path)
    body = description.build_description(
        file_path, cfg=cfg, media_type=media_type, describe=describe)
    text = f'Rating: {rating}\n' + body

    return _write_memory(memory_dir, soft_hash, text, when=when)


def _write_memory(memory_dir: str, key: str, text: str, when=None) -> str:
    """Write one memory file atomically and return its path.

    Atomic because a half-written memory file is a permanent, silently corrupt
    training example: nothing regenerates it, since the thing it describes may be
    gone.
    """
    target = memory_file_path(memory_dir, key, when=when)
    folder = os.path.dirname(target)
    fd, tmp = tempfile.mkstemp(dir=folder, suffix='.partial')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(text)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return target


def text_memory_key(text: str) -> str:
    """The filename a piece of text is remembered under.

    A file's memory is named by its soft hash, so that renaming or moving the
    file does not orphan the rating. Text has no path to be moved, so it is named
    by its own content: rating the same text twice replaces the earlier record
    instead of teaching the model the same example twice under two names.
    """
    return xxhash.xxh3_128(text.encode('utf-8')).hexdigest()


def save_text_rating(text: str, rating: float, *, memory_dir: str, cfg=None,
                     summarise=None, when=None) -> str:
    """Record a rating for a piece of text, and return the memory file written.

    The same shape as a file's memory — the rating on line 1, the body below it —
    because training reads nothing else: it strips line 1 and embeds the rest. So
    a rated paragraph is an ordinary training example, indistinguishable from a
    rated file's description. That is what lets you teach the model about things
    you have no file for: what you are in the mood for, a review you agree with,
    a genre you cannot stand.

    Short text is stored as it stands. Long text is summarised first if a
    *summarise* callable is given — the same rule a text file gets, since a
    rated paragraph and a rated .txt should not be treated differently for having
    arrived by a different route.
    """
    raw = (text or '').strip()
    if not raw:
        raise ValueError('refusing to remember empty text')

    body, _ = description.body_for_text(raw, cfg=cfg, summarise=summarise)
    if not body:
        raise ValueError('refusing to remember empty text')

    # Keyed on what the user gave, not on the summary: rating the same passage
    # twice must land on the same file whether or not a model was involved.
    return _write_memory(memory_dir, text_memory_key(raw),
                         f'Rating: {rating}\n{body}\n', when=when)
