"""The pinned embedding contract — the one thing in this project that must not drift.

Every vector already stored was produced by exactly one combination of embedding
model, model revision, prefixes, how images are fed in, and the fusion layout. Change
any of them and new vectors land in a *different* vector space from the old ones:
nothing errors, but cosine similarity between an old row and a new query becomes
meaningless, so search quality degrades in a way that looks like bad ranking rather
than a bug.

So the combination is written down here rather than assembled from whatever
environment variables happen to be set, and `fingerprint()` reduces it to a short
string you can store next to an index and compare later.

## Two backends, one active

`gemma2` (**active**) is EmbeddingGemma 2, and it is the *only* model in the
retrieval path. Text, images, video and audio all go into one 768-dim
cosine-comparable space, so a photo is retrievable from a text query without being
captioned and a spoken clip without being transcribed. No captioner, no ASR, not even
ffmpeg: the media file is handed to the embedder as it is.

That is the whole baseline. A captioner (Gemma) and a transcriber (Whisper) remain
wired up behind `--enrich`, because they add *readable text* to a row and may retrieve
specific phrasing better than a 768-float summary of the audio — but they are an
experiment to justify by measurement, not part of the default path.

`e5` is the previous contract, multilingual-e5-large at 1024 dim, text only. It is
kept because it is *known* strong on Russian, which EmbeddingGemma 2 is not yet
proven to be — no Russian-specific benchmark is published for it. If Russian
retrieval regresses, roll back with:

    AEROSEARCH_EMBED_BACKEND=e5

That is a different vector space, so a rollback means re-embedding everything. The
fingerprint changing is how you can tell.

## Changing a pinned value

Re-embed everything. Either drop the table and let the store recreate it (the
pgvector column width is derived from `EMBED_DIM`, so 768 and 1024 are not
interchangeable), or re-run the worker over a cleared index.

Captioner and transcriber are deliberately *not* pinned: they change the `document`
text a row carries, so a re-index produces a better document, but they do not change
the geometry, and old and new rows stay comparable.
"""
from __future__ import annotations

import hashlib
import os

#: Which contract is live. "gemma2" (multimodal, 768) or "e5" (text-only, 1024).
EMBED_BACKEND = (os.getenv("AEROSEARCH_EMBED_BACKEND") or "gemma2").strip().lower()

#: Revisions are commit SHAs, not branches. Both repos have been reuploaded
#: upstream — embeddinggemma-2's own lastModified was the same day it was pinned
#: here (2026-10-28) — and an unpinned re-download would move the vector space
#: under an existing index with nothing to notice.
_CONTRACTS: dict[str, dict] = {
    "gemma2": {
        "model": "google/embeddinggemma-2",
        "revision": "f9972ac22e696a105c482202646979e33e4b5a1e",
        "dim": 768,
        # EmbeddingGemma 2 is asymmetric via task prefixes. sentence-transformers
        # exposes these as prompt_name="SearchQuery" / "Document"; the literal
        # strings are recorded so they are part of the fingerprint.
        "query_prefix": "task: search result | query: ",
        "doc_prefix": "title: none | text: ",
        # One model for every modality: images, video and audio go straight to
        # the embedder, no prefix, no captioner and no ASR in the retrieval path.
        # See embed_gemma2.py.
        "media_mode": "av-direct-v1",
    },
    "e5": {
        "model": "intfloat/multilingual-e5-large",
        "revision": "3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3",
        "dim": 1024,
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "media_mode": "none",  # text only; media needs a caption/transcript to be findable
    },
}

if EMBED_BACKEND not in _CONTRACTS:
    raise ValueError(
        f"AEROSEARCH_EMBED_BACKEND={EMBED_BACKEND!r} is not a known contract "
        f"(expected one of {sorted(_CONTRACTS)})"
    )

_ACTIVE = _CONTRACTS[EMBED_BACKEND]

EMBED_MODEL: str = _ACTIVE["model"]
EMBED_REVISION: str | None = os.getenv("AEROSEARCH_EMBED_REVISION") or _ACTIVE["revision"]
EMBED_DIM: int = _ACTIVE["dim"]
QUERY_PREFIX: str = _ACTIVE["query_prefix"]
PASSAGE_PREFIX: str = _ACTIVE["doc_prefix"]
MEDIA_MODE: str = _ACTIVE["media_mode"]

#: True when the active contract embeds raw media (pixels, audio) straight into the
#: same space as the query. That is what lets a photo be found without a caption and
#: a spoken clip without a transcript.
EMBEDS_MEDIA: bool = MEDIA_MODE != "none"

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
        MEDIA_MODE,
        FUSION_VERSION,
    ]
    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()[:12]


def describe() -> dict[str, str]:
    """The contract as plain data, for printing or storing beside an index."""
    return {
        "backend": EMBED_BACKEND,
        "embed_model": EMBED_MODEL,
        "embed_revision": EMBED_REVISION or "(unpinned)",
        "embed_dim": str(EMBED_DIM),
        "passage_prefix": PASSAGE_PREFIX,
        "query_prefix": QUERY_PREFIX,
        "media_mode": MEDIA_MODE,
        "embeds_media": str(EMBEDS_MEDIA),
        "fusion_version": FUSION_VERSION,
        "fingerprint": fingerprint(),
    }
