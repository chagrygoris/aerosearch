"""Fusion + fake embedder + memory store — pure, fast, no models."""
import numpy as np
import pytest

from aerosearch.fusion import fuse
from aerosearch.store import MemoryVectorStore
from aerosearch.types import ExtractedMedia, MediaDocument
from aerosearch.understanding.fakes import FakeEmbedder


def test_fuse_combines_signals():
    em = ExtractedMedia(source="/x/xi_jinping.mp4", media_type="video")
    doc = fuse(em, caption="A man gives a speech", transcript="привет мир")
    assert "xi jinping" in doc.document.lower()
    assert "A man gives a speech" in doc.document
    assert "привет мир" in doc.document
    assert doc.caption and doc.transcript


def test_fuse_empty_falls_back_to_filename():
    em = ExtractedMedia(source="/x/macarena.mp4", media_type="video")
    doc = fuse(em, caption="", transcript="")
    assert "macarena" in doc.document.lower()


def test_fake_embedder_is_unit_normalized():
    emb = FakeEmbedder(dim=128)
    v = np.asarray(emb.embed_query("hello world"))
    assert v.shape == (128,)
    assert np.isclose(np.linalg.norm(v), 1.0, atol=1e-5)


def test_fake_embedder_lexical_similarity():
    emb = FakeEmbedder(dim=512)
    a = np.asarray(emb.embed_query("macarena dance party"))
    close = np.asarray(emb.embed_query("macarena dance"))
    far = np.asarray(emb.embed_query("chess tournament"))
    assert float(a @ close) > float(a @ far)


async def test_memory_store_roundtrip_and_search():
    emb = FakeEmbedder(dim=256)
    store = MemoryVectorStore()
    docs = []
    for text, src in [("macarena dance", "macarena.mp4"), ("chess game", "chess.mp4")]:
        d = MediaDocument(source=src, media_type="video", document=text)
        d.embedding = emb.embed_documents([text])[0]
        docs.append(d)
    await store.add(docs)

    hits = await store.search(emb.embed_query("macarena"), k=2)
    assert hits[0].source == "macarena.mp4"
    assert hits[0].score >= hits[1].score


async def test_memory_store_persistence(tmp_path):
    base = str(tmp_path / "store")
    emb = FakeEmbedder(dim=64)
    d = MediaDocument(source="a.mp4", media_type="video", document="hello")
    d.embedding = emb.embed_documents(["hello"])[0]

    s1 = MemoryVectorStore(path=base)
    await s1.add([d])

    s2 = MemoryVectorStore(path=base)  # reload from disk
    hits = await s2.search(emb.embed_query("hello"), k=1)
    assert len(hits) == 1 and hits[0].source == "a.mp4"
