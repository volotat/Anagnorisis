"""The one description of a file, in words.

Everything downstream reads this text and nothing else: metadata search embeds
it, the UI shows it as "full search description", the memory file stores it, and
the data server publishes it in a `.meta` sidecar. It is the project's universal
currency, because a sentence travels between machines running different models
and a vector does not.

**There is one text, not one per consumer.** When you open "full search
description" you are looking at precisely what was embedded and precisely what
will be written into the memory file — the same tags, the same fingerprint, the
same order. Not a rendering of it, the thing itself. This used to be assembled
in two places with two `.meta` readers and two sets of caps, and they drifted:
the displayed description stopped matching the indexed one and search results
looked random. `tests/test_description.py` now asserts the consumers agree.

The only sanctioned variation is the data server's, and both halves of it are
forced rather than stylistic: it omits the `.meta` sidecar because it is writing
that file, and it omits the path because that would publish the server's
directory layout to strangers.

The auto-description is injected as a callable rather than produced here. This
module assembles text; whose model wrote the description, and which cache it came
from, is the caller's business.
"""
import os
from typing import Callable, Optional

import fs

from anagnorisis_core.media import extractors, models
from anagnorisis_core.storage import virtual_file_system as vfs
from anagnorisis_core.media.media_types import MediaType

# Read only the head of a `.meta` sidecar. It is user-authored and can be any
# size; the embedding does not benefit from a novel, and the read happens on the
# search path.
MAX_META_LINES = 300
MAX_META_CHARS = 30_000
# Internal-metadata values longer than this are dropped — they are base64 cover
# art and similar, which is bulk with no meaning to a reader or a model.
MAX_META_VALUE_CHARS = 1000

# What a remote file gets instead of the sections that would require downloading
# it. Kept as a constant because it is part of the indexed text.
REMOTE_NOTE = (
    "# Remote file note:\n"
    "Original file content is not fetched automatically. "
    "Rate this file to generate a permanent description in memory.\n\n"
)


def read_meta_snippet(meta_path: str) -> tuple[str, bool]:
    """Read a `.meta` sidecar, capped. Returns (text, was_truncated).

    VFS-aware: *meta_path* may be a plain path or a full VFS URL, so this works
    for a local folder and for a remote server alike. A missing file is normal
    and returns empty rather than raising.
    """
    lines: list[str] = []
    total = 0
    truncated = False

    # Almost every file is asked this and almost none has a sidecar, and
    # fs.open_fs costs about 535µs against 1.6µs for os.path.exists. Answering
    # "no" the cheap way saves roughly half a millisecond per file on both the
    # describe-write and the metadata-index paths. Remote paths still go the
    # long way round, because only a filesystem can answer for them.
    if vfs.is_local_url(meta_path):
        try:
            local_path, _temp = vfs.resolve_to_local_path(meta_path)
            if not os.path.exists(local_path):
                return '', False
        except Exception:
            pass

    try:
        base_url, path_in_fs = vfs.resolve_base_and_path_from_url(meta_path)
        with fs.open_fs(base_url) as my_fs:
            if not my_fs.exists(path_in_fs):
                return '', False
            with my_fs.open(path_in_fs, 'rb') as fh:
                for i, raw in enumerate(fh):
                    line = raw.decode('utf-8', errors='ignore')
                    if i >= MAX_META_LINES or total + len(line) > MAX_META_CHARS:
                        truncated = True
                        break
                    lines.append(line)
                    total += len(line)
    except Exception as exc:
        print(f"Error reading metadata file {meta_path}: {exc}")
    return ''.join(lines), truncated


# Below this many characters a piece of text is its own best description, so it
# is kept verbatim. Above it, the text is summarised — which costs a model run,
# and loses detail, and is worth it only once the text is too long to be read as
# a whole. Adjustable as `omni.text_verbatim_max_chars`.
TEXT_VERBATIM_MAX_CHARS = 4000


def text_verbatim_limit(cfg) -> int:
    omni = getattr(cfg, 'omni', None)
    if omni is None:
        return TEXT_VERBATIM_MAX_CHARS
    return int(omni.get('text_verbatim_max_chars', TEXT_VERBATIM_MAX_CHARS))


def body_for_text(content: str, *, cfg, summarise=None) -> tuple[str, bool]:
    """The description of a piece of text, and whether it was summarised.

    Short text is returned as it stands. Summarising it would spend a model run
    to produce something *less* informative than what it already had — the words
    themselves are what a search should match and what the evaluator should learn
    from. Only once the text is too long to be taken in whole does compressing it
    start to pay.

    *summarise* is injected, so this decides *whether* to spend model time and
    never *how*. With no summariser and text over the limit, the opening is kept
    rather than nothing: it is where a document usually says what it is.
    """
    content = (content or '').strip()
    if not content:
        return '', False
    if len(content) <= text_verbatim_limit(cfg):
        return content, False
    if summarise is None:
        return content[:text_verbatim_limit(cfg)], False
    return (summarise(content) or '').strip(), True


def read_text_content(file_path: str, cap_chars: int = 30_000) -> str:
    """Read a text file so a model can summarise it, capped.

    The cap keeps a large document inside the model's context; anything longer
    is summarised from its opening, which is where a text file usually says what
    it is. VFS-aware, so this works for a local path or a remote URL.
    """
    base_url, path_in_fs = vfs.resolve_base_and_path_from_url(file_path)
    with fs.open_fs(base_url) as my_fs:
        with my_fs.open(path_in_fs, 'rb') as fh:
            return fh.read(cap_chars).decode('utf-8', errors='ignore')


def proxy_section(file_path: str, cfg, media_type: Optional[MediaType]) -> str:
    """Zero-shot tags and the quantised fingerprint, or ''.

    Rebuilt from the content-embedding cache, so this never loads a model and is
    safe to call whatever is currently in VRAM. A file that has not been embedded
    yet simply has no proxy section.
    """
    proxy = models.get_proxy(cfg, media_type)
    if proxy is None:
        return ''
    try:
        return proxy.get_proxy_text(file_path) or ''
    except Exception as exc:
        print(f"[description] Proxy section failed for {file_path}: {exc}")
        return ''


def internal_metadata_section(file_path: str, media_type: Optional[MediaType]) -> str:
    """EXIF / ID3 / filesystem facts as `key: value` lines, or ''."""
    internal = extractors.read(
        media_type.metadata_extractor if media_type else None, file_path
    )
    if not internal:
        return ''
    lines = [
        f"{key}: {value}"
        for key, value in internal.items()
        if isinstance(value, str) and len(value) <= MAX_META_VALUE_CHARS
    ]
    if not lines:
        return ''
    return "# Internal metadata:\n" + "\n".join(lines) + "\n\n"


def build_description(
    file_path: str,
    *,
    cfg,
    media_type: Optional[MediaType] = None,
    describe: Optional[Callable[[str], str]] = None,
    include_path: bool = True,
    include_meta_sidecar: bool = True,
) -> str:
    """Assemble the description of *file_path*.

    Args:
        cfg: the configuration; only the embedder/omni sections and the cache
            path are read.
        media_type: resolved type, or None to look it up.
        describe: callable returning the model-written description, or None to
            omit that section. Injected so this module never decides whether to
            spend GPU time.
        include_path: False publishes the file's name but not its location —
            for descriptions that leave the machine.
        include_meta_sidecar: False when the caller is *writing* the sidecar,
            which would otherwise fold the previous run's output back in.

    A remote file gets name, path and sidecar only: every other section would
    mean downloading it, and background passes must not.
    """
    file_name = os.path.basename(file_path)
    if media_type is None:
        from anagnorisis_core.media.media_types import get_registry
        media_type = get_registry(cfg).for_file(file_path)

    text = f"File Name: {file_name}\n"
    if include_path:
        text += f"File Path: {file_path}\n"
    text += "\n"

    if vfs.is_local_url(file_path):
        # A short text file is its own best description: keeping it verbatim
        # costs nothing, loses nothing, and skips a model run that would have
        # produced something less informative. Long ones are summarised as
        # before. The section is labelled for what it actually is, so nothing
        # downstream has to guess which it got.
        verbatim = _short_text_content(file_path, cfg, media_type)
        if verbatim is not None:
            text += "# Content:\n" + verbatim + "\n\n"
        elif describe is not None:
            auto_desc = describe(file_path)
            if auto_desc:
                text += "# Automatic description:\n" + auto_desc + "\n\n"

        proxy_text = proxy_section(file_path, cfg, media_type)
        if proxy_text:
            text += proxy_text + "\n"

        text += internal_metadata_section(file_path, media_type)
    else:
        text += REMOTE_NOTE

    if include_meta_sidecar:
        meta_content, _ = read_meta_snippet(file_path + '.meta')
        if meta_content:
            text += (f"# External metadata from '{file_name}.meta' file:\n"
                     + meta_content + "\n")

    return text


def _short_text_content(file_path: str, cfg, media_type) -> Optional[str]:
    """The file's own text when it is a text file short enough to keep, else None.

    None means "not applicable" — either this is not text, or it is long enough
    to be worth summarising — so the caller falls through to the descriptor.
    """
    from anagnorisis_core.media import models as media_models

    if media_models.describe_method_for(
            media_type.name if media_type else None) != 'describe_text':
        return None
    try:
        content = read_text_content(file_path)
    except Exception:
        return None
    body, summarised = body_for_text(content, cfg=cfg, summarise=None)
    if summarised or not body or len(content.strip()) > text_verbatim_limit(cfg):
        return None
    return body
