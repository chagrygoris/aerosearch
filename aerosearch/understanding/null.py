"""Captioner that returns nothing — the vision path switched off.

Not a test double. `FakeCaptioner` emits a marker string ("[fake-caption …]") which
is useful in tests but would be embedded as if it were real content, polluting the
index. This returns "" so `fusion.fuse` simply omits the "Visual:" line and the
document falls back to transcript plus filename hint.

Why it exists: a local Mac run has no CUDA, and Gemma-class captioning on CPU/MPS
takes tens of seconds per clip (measured: 9-16 s/clip on a T4, several times that on
CPU). For a local deployment the useful trade is to ingest and search immediately on
speech plus filename, and let the captions arrive from a GPU batch later.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


class NullCaptioner:
    """Satisfies the `Captioner` protocol and describes nothing."""

    def caption(self, frames: list["Image"], media_type: str) -> str:
        return ""


class NullTranscriber:
    """Transcriber that returns nothing — ASR switched off.

    The baseline embeds speech audio straight into the retrieval space, so no
    transcript is needed to find a spoken clip. Returning "" means `fusion.fuse`
    omits the "Speech:" line rather than inserting a placeholder that would be
    embedded as if it were content.
    """

    def transcribe(self, audio_path: str) -> str:
        return ""
