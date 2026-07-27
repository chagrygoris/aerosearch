"""End-to-end system test over the src-videos benchmark with fake models.

Exercises the real extraction + fusion + store + search wiring (only the model
inference is faked), so it catches integration breakage without any downloads.
"""
import shutil

import pytest

from aerosearch.config import load_config
from aerosearch.factory import build_captioner, build_embedder, build_transcriber
from aerosearch.pipeline import Indexer, discover_media, search
from aerosearch.store import MemoryVectorStore

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")


async def test_index_and_search_src_videos(src_videos_dir):
    cfg = load_config()
    cfg.media.num_frames = 2  # keep the test fast
    store = MemoryVectorStore()
    indexer = Indexer(
        cfg,
        captioner=build_captioner(cfg, "fake"),
        transcriber=build_transcriber(cfg, "fake"),
        embedder=build_embedder(cfg, "fake"),
        store=store,
    )

    paths = discover_media(src_videos_dir)
    assert len(paths) > 10
    docs = await indexer.index_paths(paths)
    assert len(docs) == len(paths)
    assert all(d.embedding is not None for d in docs)

    # Filename signal should let the fake lexical embedder find the right clip.
    embedder = build_embedder(cfg, "fake")
    hits = await search("macarena", embedder, store, k=3)
    assert hits
    assert "macarena" in hits[0].source.lower()
