"""Fuse the signals from one piece of media into a single embeddable document.

Search quality lives or dies here: the vision caption (what's on screen),
the speech transcript (what's said) and a light filename hint are combined into
one text blob. e5 wants a short instruction prefix on documents, but we keep the
raw fused text here and let the embedder add its own prefix, so the stored
`document` stays human-readable.
"""
from __future__ import annotations

import os

from .types import ExtractedMedia, MediaDocument


def _filename_hint(source: str) -> str:
    """Filenames often carry signal ('irina_allegrova_sigma_boy'). Use lightly."""
    base = os.path.splitext(os.path.basename(source))[0]
    return base.replace("_", " ").replace("-", " ").strip()


def fuse(extracted: ExtractedMedia, caption: str, transcript: str) -> MediaDocument:
    caption = (caption or "").strip()
    transcript = (transcript or "").strip()
    hint = _filename_hint(extracted.source)

    parts: list[str] = []
    if hint:
        parts.append(f"File: {hint}.")
    if caption:
        parts.append(f"Visual: {caption}")
    if transcript:
        parts.append(f"Speech: {transcript}")

    document = "\n".join(parts) if parts else hint

    return MediaDocument(
        source=extracted.source,
        media_type=extracted.media_type,
        caption=caption,
        transcript=transcript,
        document=document,
    )
