"""One model, every modality: the baseline hands the file to the embedder.

With `enrich=False` (the default) `Indexer.understand` does no frame sampling, no
captioning and no ASR — it calls `embed_file(path, media_type)` and stores the
result. `enrich=True` restores the old path, where frames are extracted and a
caption/transcript is fused into `document` before embedding.

The distinction is the whole architecture, so it is asserted here rather than
assumed.
"""
import shutil

import pytest
from PIL import Image

from aerosearch.config import load_config
from aerosearch.pipeline import Indexer, media_type_of
from aerosearch.store.memory import MemoryVectorStore
from aerosearch.understanding.base import Embedder, MultimodalEmbedder
from aerosearch.understanding.fakes import (
    FakeCaptioner,
    FakeEmbedder,
    FakeMultimodalEmbedder,
    FakeTranscriber,
)
from aerosearch.understanding.null import NullCaptioner, NullTranscriber


@pytest.fixture
def photo(tmp_path):
    p = tmp_path / "shot.png"
    Image.new("RGB", (64, 48), "red").save(p)
    return str(p)


def _indexer(embedder, tmp_path, enrich=False, captioner=None, transcriber=None):
    return Indexer(
        load_config(),
        captioner=captioner or NullCaptioner(),
        transcriber=transcriber or NullTranscriber(),
        embedder=embedder,
        store=MemoryVectorStore(path=str(tmp_path / "store")),
        enrich=enrich,
    )


def test_media_type_from_extension_needs_no_ffprobe():
    assert media_type_of("a/b/c.JPG") == "photo"
    assert media_type_of("x.png") == "photo"
    assert media_type_of("clip.mp4") == "video"
    assert media_type_of("clip.webm") == "video"


def test_protocol_tells_the_two_embedders_apart():
    assert isinstance(FakeMultimodalEmbedder(), MultimodalEmbedder)
    assert isinstance(FakeMultimodalEmbedder(), Embedder)
    # A text-only embedder must NOT satisfy the multimodal protocol, or the
    # pipeline would hand it media it cannot use.
    assert not isinstance(FakeEmbedder(), MultimodalEmbedder)


def test_baseline_hands_the_file_straight_to_the_embedder(photo, tmp_path):
    emb = FakeMultimodalEmbedder()
    doc = _indexer(emb, tmp_path).understand(photo)

    assert emb.file_calls == [(photo, "photo")], "baseline should call embed_file"
    assert not emb.media_calls, "baseline must not extract frames"
    assert doc.embedding is not None and len(doc.embedding) == emb.dim
    # document is only for display now; retrieval does not depend on it.
    assert doc.document == "shot"
    assert doc.media_type == "photo"


def test_baseline_runs_no_captioner_and_no_asr(photo, tmp_path):
    """A captioner/transcriber handed in are ignored unless enrich=True."""
    cap, tr = FakeCaptioner(), FakeTranscriber()
    doc = _indexer(FakeMultimodalEmbedder(), tmp_path, captioner=cap, transcriber=tr).understand(photo)
    # FakeCaptioner would have injected "[fake-caption ...]" into the document.
    assert "fake-caption" not in doc.document
    assert "Visual:" not in doc.document


@pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffmpeg not installed")
def test_enrich_restores_frame_extraction_and_fusion(photo, tmp_path):
    emb = FakeMultimodalEmbedder()
    doc = _indexer(emb, tmp_path, enrich=True, captioner=FakeCaptioner()).understand(photo)

    assert emb.media_calls, "enrich should extract frames and call embed_media"
    assert not emb.file_calls
    assert "fake-caption" in doc.document, "caption should be fused into the document"


def test_text_only_embedder_still_uses_the_enriched_path(photo, tmp_path):
    """e5 rollback: no embed_file, so it must fall back to extract + fuse."""
    emb = FakeEmbedder(dim=64)
    idx = _indexer(emb, tmp_path)
    if shutil.which("ffprobe") is None:
        pytest.skip("ffmpeg not installed")
    doc = idx.understand(photo)
    # Embedding is deferred to the batched step for text-only embedders.
    assert doc.embedding is None


async def test_two_photos_with_identical_display_text_get_different_vectors(tmp_path):
    """The failure this architecture fixes.

    Under the text-only contract every photo's document was the same string, so the
    embedder mapped them to the *identical* vector and no query could rank them.
    Embedding the file itself makes the rows differ.
    """
    red, blue = tmp_path / "a.png", tmp_path / "b.png"
    Image.new("RGB", (64, 48), "red").save(red)
    Image.new("RGB", (64, 48), "blue").save(blue)

    text_only = FakeEmbedder(dim=64)
    same = text_only.embed_documents(["File: media.", "File: media."])
    assert same[0] == same[1], "precondition: identical text embeds identically"

    emb = FakeMultimodalEmbedder()
    docs = await _indexer(emb, tmp_path).index_paths([str(red), str(blue)])

    assert len(docs) == 2
    assert all(d.embedding is not None for d in docs)
    assert len(emb.file_calls) == 2
    assert docs[0].embedding != docs[1].embedding
