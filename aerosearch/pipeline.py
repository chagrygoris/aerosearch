"""The indexing pipeline: media file -> understood, embedded, stored document."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterable, Optional

from .config import Config
from .fusion import fuse
from .media import extract_media
from .store.base import VectorStore
from .types import MediaDocument
from .understanding.base import Captioner, Embedder, MultimodalEmbedder, Transcriber

logger = logging.getLogger(__name__)

MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp", ".gif"}


class Indexer:
    """Wires the four models + a store into a repeatable indexing step."""

    def __init__(
        self,
        cfg: Config,
        captioner: Captioner,
        transcriber: Transcriber,
        embedder: Embedder,
        store: VectorStore,
    ):
        self.cfg = cfg
        self.captioner = captioner
        self.transcriber = transcriber
        self.embedder = embedder
        self.store = store

    def understand(self, path: str, **provenance) -> MediaDocument:
        """Run the (blocking) model stack on one file -> a MediaDocument.

        `provenance` may carry chat_id/user_id/user_name/s3_url from Telegram.
        Embedding is filled in during `index_paths` (batched) — not here.
        """
        with tempfile.TemporaryDirectory(prefix="aerosearch_") as wd:
            extracted = extract_media(path, self.cfg.media, wd)

            caption = ""
            if extracted.frames:
                caption = self.captioner.caption(extracted.frames, extracted.media_type)

            transcript = ""
            if extracted.audio_path:
                transcript = self.transcriber.transcribe(extracted.audio_path)

            doc = fuse(extracted, caption, transcript)

            # A multimodal embedder puts the pixels in the same space as the query,
            # which is the only way a photo is findable without a caption. It has to
            # happen here, inside the `with`, because the frames die with `wd`.
            # Text-only rows are still embedded in a batch by the caller.
            if extracted.frames and isinstance(self.embedder, MultimodalEmbedder):
                doc.embedding = self.embedder.embed_media(doc.document, extracted.frames)

        for key in ("chat_id", "user_id", "user_name", "s3_url"):
            if key in provenance:
                setattr(doc, key, provenance[key])
        return doc

    async def index_paths(self, paths: Iterable[str]) -> list[MediaDocument]:
        """Understand + embed + store a batch of files. Returns the documents."""
        docs: list[MediaDocument] = []
        for path in paths:
            try:
                doc = self.understand(path)
                docs.append(doc)
                logger.info("understood %s -> %d chars", path, len(doc.document))
            except Exception:
                logger.exception("failed to index %s", path)

        if not docs:
            return []

        # Rows a multimodal embedder already handled (frames + text in one vector)
        # carry an embedding; the rest are batched here, which is cheaper per row.
        pending = [d for d in docs if d.embedding is None]
        if pending:
            embeddings = self.embedder.embed_documents([d.document for d in pending])
            for d, e in zip(pending, embeddings):
                d.embedding = e

        await self.store.init()
        await self.store.add(docs)
        return docs


def discover_media(root: str) -> list[str]:
    """List indexable media under a directory (or return [root] if it's a file)."""
    if os.path.isfile(root):
        return [root]
    found = []
    for name in sorted(os.listdir(root)):
        if os.path.splitext(name)[1].lower() in MEDIA_EXTS:
            found.append(os.path.join(root, name))
    return found


async def search(
    query: str,
    embedder: Embedder,
    store: VectorStore,
    k: int = 5,
    chat_id: Optional[int] = None,
):
    qv = embedder.embed_query(query)
    return await store.search(qv, k=k, chat_id=chat_id)
