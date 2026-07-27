"""Batch indexer core loop, exercised with fakes (no MinIO / Postgres / models)."""
import shutil

import pytest

from aerosearch.batch import BatchIndexer
from aerosearch.config import load_config
from aerosearch.understanding.fakes import FakeCaptioner, FakeEmbedder, FakeTranscriber

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")


class FakePendingStore:
    def __init__(self, rows):
        self.rows = rows
        self.updated = {}  # media_id -> MediaDocument

    async def init(self):
        return None

    async def fetch_unindexed(self, limit=100):
        return [r for r in self.rows if r["media_id"] not in self.updated][:limit]

    async def update_indexed(self, media_id, doc):
        self.updated[media_id] = doc


class FileDownloader:
    """Stands in for MinIO: 'downloads' a local benchmark file's bytes."""

    def __init__(self, path):
        self.path = path

    def download(self, s3_url):
        with open(self.path, "rb") as f:
            return f.read()


async def test_batch_indexes_text_and_media(sample_video):
    cfg = load_config()
    cfg.media.num_frames = 2
    rows = [
        {"media_id": 1, "chat_id": 7, "media_type": "text",
         "s3_url": None, "source": "msg-1", "document": "привет как дела"},
        {"media_id": 2, "chat_id": 7, "media_type": "video",
         "s3_url": "minio:9000/bucket/macarena.mp4", "source": "macarena.mp4", "document": None},
    ]
    store = FakePendingStore(rows)
    batch = BatchIndexer(
        cfg,
        captioner=FakeCaptioner(),
        transcriber=FakeTranscriber(),
        embedder=FakeEmbedder(dim=128),
        store=store,
        downloader=FileDownloader(sample_video),
    )

    n = await batch.run()
    assert n == 2
    assert set(store.updated) == {1, 2}

    text_doc = store.updated[1]
    assert text_doc.media_type == "text"
    assert "привет" in text_doc.document
    assert text_doc.embedding is not None and len(text_doc.embedding) == 128

    media_doc = store.updated[2]
    assert media_doc.source == "macarena.mp4"  # real provenance, not the temp path
    assert media_doc.embedding is not None

    # A second run has nothing left to do.
    assert await batch.run() == 0


async def test_batch_media_row_requires_downloader():
    cfg = load_config()
    rows = [{"media_id": 1, "media_type": "video", "s3_url": "x/y.mp4",
             "source": "y.mp4", "document": None}]
    store = FakePendingStore(rows)
    batch = BatchIndexer(
        cfg, FakeCaptioner(), FakeTranscriber(), FakeEmbedder(dim=64),
        store=store, downloader=None,
    )
    # The row fails to understand (no downloader) and is skipped, not crashed.
    assert await batch.run() == 0
    assert store.updated == {}
