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
    """An `Embedder` that also puts *pixels* in the same space as the text.

    The distinction matters: with a text-only embedder a photo is findable only
    through a caption, so no captioner means nothing to match on. A multimodal one
    embeds the frames directly, so a text query reaches the image itself.

    `embed_media` returns ONE vector for a row that has both frames and text (a clip
    with speech, say), because a row holds a single vector. Implementations
    interleave the two rather than averaging two separate embeddings.
    """

    dim: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def embed_media(self, text: str, images: list["Image"]) -> list[float]: ...
