"""Stable identity for a locally downloaded model.

Every expensive thing this project computes — a content embedding, a metadata
embedding, a natural-language description — is cached under a key that contains
the identity of the model that produced it. That identity therefore has exactly
two jobs, and both matter:

  * **Reproduce.** The same model must yield the same string in every process,
    forever. If it drifts, every restart silently recomputes the whole library
    and no cache ever pays for itself.
  * **Change when the output would change.** Swap the weights, switch the task
    adapter, alter the prompt — anything that changes what the model returns
    must change the identity, or stale results are served as if they were fresh.

Fingerprinting the *files on disk* is what satisfies the first job. The obvious
alternative — hashing the loaded ``state_dict()`` — fails it, and did: the
jina-v5-omni checkpoint ships no trained audio LoRA, so PEFT fills in 384
``lora_A`` tensors randomly on every load. They are inert (their ``lora_B``
pairs are all zero) so embeddings were reproducible, but the hash was not, and
the content cache was rebuilt from scratch on every restart and every idle
respawn. Reading the files avoids the whole class of problem, and needs no model
loaded to do it.

The second job is the caller's: pass in every setting that changes the output.
"""
import hashlib
import os
from typing import Iterator

# Bytes sampled from each end of a weight file. Enough to catch a different
# checkpoint without reading gigabytes on every startup.
HASH_SAMPLE_BYTES = 65536

# Transient files a download leaves behind. A lock or a half-written shard would
# otherwise change the fingerprint of an unchanged model.
_SKIP_SUFFIXES = ('.lock', '.incomplete', '.tmp', '.pyc')


def model_files(local_path: str) -> Iterator[str]:
    """Relative paths of the files that define the model."""
    for root, dirs, files in os.walk(local_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        for name in files:
            if name.startswith('.') or name.endswith(_SKIP_SUFFIXES):
                continue
            yield os.path.relpath(os.path.join(root, name), local_path)


def fingerprint_model_dir(local_path: str, *settings) -> str:
    """Identity of the model in *local_path*, as configured by *settings*.

    Pass every setting that changes what the model outputs — the task adapter,
    the truncation dimension, the generation prompt. They are folded in as
    plain strings, so anything ``str()`` renders stably will do.

    If the directory cannot be read, the settings alone still produce a stable
    key; the fingerprint just stops noticing an in-place weight swap. That is
    the right failure mode: a consistent key serves cached work correctly, while
    a random one throws all of it away.
    """
    md5 = hashlib.md5()
    for setting in settings:
        md5.update(str(setting).encode('utf-8'))
    try:
        for rel in sorted(model_files(local_path)):
            path = os.path.join(local_path, rel)
            size = os.path.getsize(path)
            md5.update(rel.encode('utf-8'))
            md5.update(str(size).encode('utf-8'))
            with open(path, 'rb') as fh:  # sample the ends; never read GBs
                md5.update(fh.read(HASH_SAMPLE_BYTES))
                if size > HASH_SAMPLE_BYTES * 2:
                    fh.seek(-HASH_SAMPLE_BYTES, os.SEEK_END)
                    md5.update(fh.read(HASH_SAMPLE_BYTES))
    except OSError as exc:
        print(f"[model_identity] Could not fingerprint '{local_path}' ({exc}); "
              f"falling back to the configured settings alone.")
    return md5.hexdigest()
