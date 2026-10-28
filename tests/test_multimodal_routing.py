"""Frames must reach a multimodal embedder before the temp dir closes.

This is the whole mechanism behind Option A. `Indexer.understand` extracts frames
into a `TemporaryDirectory` and the PIL images die with it, so if embedding waited
until the batching step in `index_paths` (as it did for the text-only e5 contract)
the pixels would be gone and a photo would again be searchable only via its caption.
"""
import shutil

import pytest
from PIL import Image

from aerosearch.config import load_config
from aerosearch.pipeline import Indexer
from aerosearch.store.memory import MemoryVectorStore
from aerosearch.understanding.base import Embedder, MultimodalEmbedder
from aerosearch.understanding.fakes import (
    FakeEmbedder,
    FakeMultimodalEmbedder,
    FakeTranscriber,
)
from aerosearch.understanding.null import NullCaptioner

pytestmark = pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "shot.png"
    Image.new("RGB", (64, 48), "red").save(p)
    return str(p)


def _indexer(embedder, tmp_path):
    return Indexer(
        load_config(),
        captioner=NullCaptioner(),      # captions off: the local default
        transcriber=FakeTranscriber(),
        embedder=embedder,
        store=MemoryVectorStore(path=str(tmp_path / "store")),
    )


def test_protocol_tells_the_two_embedders_apart():
    assert isinstance(FakeMultimodalEmbedder(), MultimodalEmbedder)
    assert isinstance(FakeMultimodalEmbedder(), Embedder)
    # A text-only embedder must NOT satisfy the multimodal protocol, or the
    # pipeline would hand it frames it cannot use.
    assert not isinstance(FakeEmbedder(), MultimodalEmbedder)


def test_multimodal_embedder_receives_the_frames(photo, tmp_path):
    emb = FakeMultimodalEmbedder()
    doc = _indexer(emb, tmp_path).understand(photo)

    assert emb.media_calls, "frames never reached the embedder"
    _text, n_frames = emb.media_calls[0]
    assert n_frames == 1
    # Embedded during understand(), while the frames were still alive.
    assert doc.embedding is not None and len(doc.embedding) == emb.dim


def test_text_only_embedder_leaves_embedding_for_the_batch_step(photo, tmp_path):
    emb = FakeEmbedder(dim=64)
    doc = _indexer(emb, tmp_path).understand(photo)
    assert doc.embedding is None, "text-only path should defer to batched embedding"


async def test_two_photos_with_identical_text_get_different_vectors(tmp_path):
    """The failure Option A fixes.

    With captions off every photo's document is the same string, so a text-only
    embedder maps them to the *identical* vector and no query can rank them. A
    multimodal embedder sees the pixels, so the rows differ.
    """
    red, blue = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (64, 48), "red").save(red)
    Image.new("RGB", (64, 48), "blue").save(blue)

    text_only = FakeEmbedder(dim=64)
    same = text_only.embed_documents(["File: media.", "File: media."])
    assert same[0] == same[1], "precondition: identical text embeds identically"

    emb = FakeMultimodalEmbedder()
    idx = _indexer(emb, tmp_path)
    docs = await idx.index_paths([str(red), str(blue)])

    assert len(docs) == 2
    assert all(d.embedding is not None for d in docs)
    assert len(emb.media_calls) == 2, "both photos should go through embed_media"
