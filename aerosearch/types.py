"""Shared data types that flow through the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid importing PIL at module import time
    from PIL.Image import Image


@dataclass
class ExtractedMedia:
    """The result of decoding a raw file into model-ready inputs."""

    source: str  # path or URL the media came from
    media_type: str  # "video" | "photo"
    frames: list["Image"] = field(default_factory=list)  # sampled RGB frames
    audio_path: Optional[str] = None  # 16k mono wav, or None when silent
    duration_sec: float = 0.0

    @property
    def has_audio(self) -> bool:
        return self.audio_path is not None


@dataclass
class MediaDocument:
    """A fully-understood piece of media, ready to embed and store."""

    source: str
    media_type: str
    caption: str = ""  # what the vision model saw
    transcript: str = ""  # what Whisper heard
    document: str = ""  # fused text that actually gets embedded
    embedding: Optional[list[float]] = None
    # Telegram provenance (optional; set when indexing from the bot).
    chat_id: Optional[int] = None
    user_id: Optional[int] = None
    user_name: Optional[str] = None
    s3_url: Optional[str] = None


@dataclass
class SearchHit:
    source: str
    media_type: str
    score: float  # cosine similarity in [-1, 1], higher = closer
    document: str
    s3_url: Optional[str] = None
    user_name: Optional[str] = None
