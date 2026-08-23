"""Where content embeddings are kept, and under what name.

Two things read this cache and neither owns it: the search engine that writes the
embeddings, and the embedding proxy that turns one back into tags and a
fingerprint. So the key formula lives here rather than in either of them — a
second, hand-rolled copy of that string is exactly how a proxy silently stops
finding embeddings and every file quietly loses its tags.

It is deliberately tiny. The search engine itself is still application-side; only
the contract it shares with the proxy lives in the core.
"""

# Bump when the stored vectors change shape or meaning, so old entries are
# ignored rather than misread.
CONTENT_ALGORITHM_VERSION = "content-v1.0"

# Sub-directory of the cache holding content embeddings. One namespace for all
# media types — the model is the same, and the key already carries its hash.
CACHE_PREFIX = "content"


def embedding_cache_key(file_path: str, model_hash: str, version: str = '') -> str:
    """The one formula for a content-embedding cache key.

    The model hash is in the key because a different model means a different
    vector space: reusing a vector across models would compare things that were
    never measured the same way.
    """
    return f"{file_path}::{model_hash}{version}"
