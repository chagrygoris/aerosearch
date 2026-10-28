"""The indexing pipeline: media file -> understood, embedded, stored document."""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Iterable, Optional

from .config import Config
from .fusion import _filename_hint, fuse
from .media import extract_media
from .store.base import VectorStore
from .types import MediaDocument
from .understanding.base import Captioner, Embedder, MultimodalEmbedder, Transcriber

logger = logging.getLogger(__name__)

MEDIA_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".jpg", ".jpeg", ".png", ".webp", ".gif"}


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}


def media_type_of(path: str) -> str:
    """"photo" or "video", from the extension alone — no ffprobe needed."""
    return "photo" if os.path.splitext(path)[1].lower() in IMAGE_EXTS else "video"


class Indexer:
    """Wires the models + a store into a repeatable indexing step.

    Baseline (`enrich=False`): the embedder is the only model. A media file goes
    straight to it, so there is no frame sampling, no captioning and no ASR — see
    `aerosearch.pinned`.

    `enrich=True` restores the captioner + transcriber, which add *readable* text to
    a row and may retrieve specific phrasing better than a 768-float summary of the
    audio. That is a hypothesis to test, not the default.
    """

    def __init__(
        self,
        cfg: Config,
        captioner: Captioner,
        transcriber: Transcriber,
        embedder: Embedder,
        store: VectorStore,
        enrich: bool = False,
    ):
        self.cfg = cfg
        self.captioner = captioner
        self.transcriber = transcriber
        self.embedder = embedder
        self.store = store
        self.enrich = enrich

    def understand(self, path: str, **provenance) -> MediaDocument:
        """Run the model stack on one file -> a MediaDocument.

        `provenance` may carry chat_id/user_id/user_name/s3_url from Telegram.
        """
        if not self.enrich and isinstance(self.embedder, MultimodalEmbedder):
            doc = self._understand_direct(path)
        else:
            doc = self._understand_enriched(path)

        for key in ("chat_id", "user_id", "user_name", "s3_url"):
            if key in provenance:
                setattr(doc, key, provenance[key])
        return doc

    def _understand_direct(self, path: str) -> MediaDocument:
        """Hand the file to the embedder. No extraction, no captioner, no ASR.

        `document` keeps the filename hint only, so search results have something
        human-readable to show; retrieval no longer depends on it.
        """
        media_type = media_type_of(path)
        doc = MediaDocument(
            source=path,
            media_type=media_type,
            document=_filename_hint(path) or os.path.basename(path),
        )
        doc.embedding = self.embedder.embed_file(path, media_type, doc.document)
        return doc

    def _understand_enriched(self, path: str) -> MediaDocument:
        with tempfile.TemporaryDirectory(prefix="aerosearch_") as wd:
            extracted = extract_media(path, self.cfg.media, wd)

            caption = ""
            if extracted.frames:
                caption = self.captioner.caption(extracted.frames, extracted.media_type)

            transcript = ""
            if extracted.audio_path:
                transcript = self.transcriber.transcribe(extracted.audio_path)

            doc = fuse(extracted, caption, transcript)

            # Must happen inside the `with`: the frames die with `wd`.
            if extracted.frames and isinstance(self.embedder, MultimodalEmbedder):
                doc.embedding = self.embedder.embed_media(doc.document, extracted.frames)
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
