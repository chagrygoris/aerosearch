"""Deterministic, dependency-light stand-ins for the real models.

These make the whole pipeline runnable offline in milliseconds. The fake
embedder is a genuine feature-hashing bag-of-words with L2 normalization, so
cosine similarity is *real* lexical similarity — a local ``search "macarena"``
actually ranks the macarena document first. That makes it useful for smoke
tests and demos, not just a null object.
"""
from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image

_TOKEN = re.compile(r"\w+", re.UNICODE)


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class FakeCaptioner:
    """Emits a stable, content-free caption. Plumbing only."""

    def caption(self, frames: list["Image"], media_type: str) -> str:
        n = len(frames)
        return f"[fake-caption {media_type} across {n} frame(s)]"


class FakeTranscriber:
    """No real ASR; echoes a marker so the fused doc is non-empty."""

    def transcribe(self, audio_path: str) -> str:
        return ""


class FakeEmbedder:
    """Feature-hashing bag-of-words -> L2-normalized vector of size `dim`."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _embed_one(self, text: str) -> list[float]:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in _tokens(text):
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)


class FakeMultimodalEmbedder(FakeEmbedder):
    """`FakeEmbedder` that also claims to embed pixels.

    Exists so the multimodal routing in `Indexer.understand` is testable without
    downloading EmbeddingGemma 2. Records each `embed_media` call so a test can
    assert the frames actually reached the embedder rather than being dropped when
    the temp dir closed.
    """

    def __init__(self, dim: int = 64):
        super().__init__(dim=dim)
        self.media_calls: list[tuple[str, int]] = []

    def embed_media(self, text: str, images: list) -> list[float]:
        self.media_calls.append((text, len(images)))
        # Fold the frame count into the text so two different frame sets differ.
        return self.embed_documents([f"{text} [frames={len(images)}]"])[0]
