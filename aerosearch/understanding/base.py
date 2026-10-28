"""Protocols every backend implements. Keep these tiny and stable."""
from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from PIL.Image import Image


@runtime_checkable
class Captioner(Protocol):
    """Describe what is happening across a set of video frames / an image."""

    def caption(self, frames: list["Image"], media_type: str) -> str: ...


@runtime_checkable
class Transcriber(Protocol):
    """Turn a 16k mono wav into text. Returns '' when there is no speech."""

    def transcribe(self, audio_path: str) -> str: ...


@runtime_checkable
class Embedder(Protocol):
    """Map texts to unit-normalized vectors of dimension `dim`."""

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class MultimodalEmbedder(Protocol):
    """An `Embedder` that puts raw media in the same space as the text.

    The distinction matters: with a text-only embedder a photo is findable only
    through a caption and a spoken clip only through a transcript, so without those
    models there is nothing to match on. A multimodal one embeds the pixels and the
    audio themselves.

    `embed_file` is the baseline: hand it the file, get the vector. Nothing is
    extracted, captioned or transcribed first, so this is the whole retrieval path
    for media.

    `embed_media` is the enriched variant, for when frames have already been sampled
    and a caption or transcript exists. It returns ONE vector, because a row holds a
    single vector — implementations interleave text and frames rather than averaging
    two separate embeddings.
    """

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_file(self, path: str, media_type: str, text: str = "") -> list[float]: ...

    def embed_media(self, text: str, images: list["Image"]) -> list[float]: ...
