"""The pinned embedding contract — the one thing in this project that must not drift.

Every vector already stored was produced by exactly one combination of embedding
model, model revision, e5 prefixes and fusion layout. Change any of them and new
vectors land in a *different* vector space from the old ones: nothing errors, but
cosine similarity between an old row and a new query becomes meaningless, so search
quality degrades in a way that looks like bad ranking rather than a bug.

So the combination is written down here rather than assembled from whatever
environment variables happen to be set, and `fingerprint()` reduces it to a short
string you can store next to an index and compare later.

Captioner and transcriber are deliberately *not* pinned here. They change the
`document` text a row carries, which means a re-index produces a better document,
but they do not change the geometry of the space, so old and new rows stay
comparable. Only what `fingerprint()` covers has that property.

## Changing a pinned value

Re-embed everything. Either drop the table and let the store recreate it, or run
`aerosearch worker --reindex-all` once the new values are in place. Bumping
`FUSION_VERSION` is the same kind of change: the text fed to the embedder differs,
so stored vectors no longer match what the current code would produce.
"""
from __future__ import annotations

import hashlib
import os

#: Embedding model. Strong on the Russian-heavy corpus, 1024-dim, ~560M params.
EMBED_MODEL = "intfloat/multilingual-e5-large"

#: Hugging Face revision (commit SHA) for EMBED_MODEL, resolved from the Hub on
#: 2026-10-28; upstream's own lastModified for this commit is 2026-04-02. Pinned
#: because the repo has been reuploaded before, and an unpinned re-download would
#: move the vector space under an existing index without any error.
#:
#: This is the baseline for everything indexed from now on. Rows embedded before it
#: was pinned may have used a different revision, so treat the current index as
#: fingerprint-unknown and re-embed if ranking looks off.
EMBED_REVISION: str | None = (
    os.getenv("AEROSEARCH_EMBED_REVISION") or "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3"
)

#: Output dimension of EMBED_MODEL. The pgvector column is created at this width,
#: so a mismatch is a hard error rather than silent drift.
EMBED_DIM = 1024

#: e5 is asymmetric: documents and queries get different prefixes. Dropping or
#: swapping them measurably hurts retrieval.
PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "

#: Layout `fusion.fuse` writes ("File: … / Visual: … / Speech: …"). Bump on any
#: change to that layout, since it changes the text the embedder sees.
FUSION_VERSION = "fuse-v1"


def fingerprint() -> str:
    """Short, stable id for the current embedding contract.

    Two indexes with the same fingerprint hold directly comparable vectors; two with
    different fingerprints do not, and mixing them silently degrades ranking.
    """
    parts = [
        EMBED_MODEL,
        EMBED_REVISION or "unpinned",
        str(EMBED_DIM),
        PASSAGE_PREFIX,
        QUERY_PREFIX,
        FUSION_VERSION,
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]


def describe() -> dict[str, str]:
    """The contract as plain data, for printing or storing beside an index."""
    return {
        "embed_model": EMBED_MODEL,
        "embed_revision": EMBED_REVISION or "(unpinned)",
        "embed_dim": str(EMBED_DIM),
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "fusion_version": FUSION_VERSION,
        "fingerprint": fingerprint(),
    }
