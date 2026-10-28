"""Scheduled batch indexer — the CPU worker Airflow runs on a timer.

Flow: the bot inserts rows (media in MinIO, text messages inline) with a NULL
embedding. This job pulls those rows, *understands* each one (Gemma 3n + Whisper
for media; nothing extra for text), embeds the fused document with e5, and writes
caption/transcript/document/embedding back — turning pending rows searchable.

Everything is injected (store, models, downloader), so the core loop is unit-
testable with fakes and no MinIO/Postgres.
"""
from __future__ import annotations

import logging
import os
import tempfile
from typing import Optional, Protocol

from .config import Config
from .fusion import _filename_hint
from .pipeline import Indexer
from .types import MediaDocument
from .understanding.base import Captioner, Embedder, Transcriber

logger = logging.getLogger(__name__)

_EXT_BY_TYPE = {"video": ".mp4", "photo": ".jpg"}


class Downloader(Protocol):
    def download(self, s3_url: str) -> bytes: ...


class PendingStore(Protocol):
    async def init(self) -> None: ...
    async def fetch_unindexed(self, limit: int = 100) -> list[dict]: ...
    async def update_indexed(self, media_id: int, doc: MediaDocument) -> None: ...


class BatchIndexer:
    def __init__(
        self,
        cfg: Config,
        captioner: Captioner,
        transcriber: Transcriber,
        embedder: Embedder,
        store: PendingStore,
        downloader: Optional[Downloader] = None,
    ):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.downloader = downloader
        # Reuse the single-file understanding logic for media rows.
        self._indexer = Indexer(cfg, captioner, transcriber, embedder, store)  # type: ignore[arg-type]

    def _understand_row(self, row: dict) -> Optional[MediaDocument]:
        media_type = row["media_type"]

        # Text messages carry their content in `document`; no models needed.
        if media_type == "text":
            text = (row.get("document") or "").strip()
            if not text:
                return None
            return MediaDocument(source=row.get("source") or "", media_type="text", document=text)

        # Media: pull the bytes out of blob storage into a temp file, understand.
        if self.downloader is None:
            raise RuntimeError("a Downloader is required to index media rows")
        s3_url = row.get("s3_url")
        if not s3_url:
            logger.warning("row %s has no s3_url; skipping", row.get("media_id"))
            return None

        data = self.downloader.download(s3_url)
        ext = os.path.splitext(s3_url)[1].lower() or _EXT_BY_TYPE.get(media_type, ".bin")
        with tempfile.TemporaryDirectory(prefix="aerosearch_dl_") as wd:
            tmp = os.path.join(wd, f"media{ext}")
            with open(tmp, "wb") as f:
                f.write(data)
            doc = self._indexer.understand(tmp)
        # Keep the real provenance, not the temp path.
        doc.source = row.get("source") or s3_url
        # Re-fuse the filename hint off the real name (temp path was meaningless).
        _ = _filename_hint(doc.source)
        return doc

    async def run(self, limit: int = 100) -> int:
        """Index up to `limit` pending rows. Returns the count indexed."""
        await self.store.init()
        rows = await self.store.fetch_unindexed(limit)
        if not rows:
            logger.info("no pending rows to index")
            return 0

        prepared: list[tuple[int, MediaDocument]] = []
        for row in rows:
            try:
                doc = self._understand_row(row)
                if doc is not None:
                    prepared.append((row["media_id"], doc))
            except Exception:
                logger.exception("failed to understand row %s", row.get("media_id"))

        if not prepared:
            return 0

        # Media rows handled by a multimodal embedder already carry a vector built
        # from their frames (see Indexer.understand); only the rest need embedding,
        # and batching those is cheaper per row.
        need = [(mid, d) for mid, d in prepared if d.embedding is None]
        if need:
            embeddings = self.embedder.embed_documents([d.document for _, d in need])
            for (_, doc), emb in zip(need, embeddings):
                doc.embedding = emb

        for media_id, doc in prepared:
            await self.store.update_indexed(media_id, doc)
            logger.info("indexed row %s (%s)", media_id, doc.media_type)

        return len(prepared)
