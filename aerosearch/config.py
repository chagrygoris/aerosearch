"""Central configuration, populated from environment variables with sane defaults.

Everything the pipeline needs to pick models, sampling rates and storage lives
here so the CLI, the bot and the Kaggle notebook all read the same knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

from .pinned import EMBED_DIM, EMBED_MODEL

load_dotenv()


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw else default


def _str(name: str, default: str) -> str:
    return os.getenv(name) or default


# NOTE: env is read via default_factory so a fresh load_config() (or MediaConfig())
# always reflects the *current* environment — not whatever was set at import time.


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw not in ("0", "false", "False", "")


@dataclass
class MediaConfig:
    """How raw video/images are turned into frames + audio."""

    num_frames: int = field(default_factory=lambda: _int("AEROSEARCH_NUM_FRAMES", 6))
    max_frame_edge: int = field(default_factory=lambda: _int("AEROSEARCH_MAX_FRAME_EDGE", 768))
    audio_sample_rate: int = 16_000  # what Whisper expects
    frame_format: str = "png"


@dataclass
class ModelConfig:
    """Which models power captioning / transcription / embedding."""

    # Smallest *multimodal* Gemma 3 (1B is text-only). Fits a T4 at 4-bit.
    # Set AEROSEARCH_CAPTION_MODEL=google/gemma-3n-E2B-it for the CPU batch path.
    caption_model: str = field(
        default_factory=lambda: _str("AEROSEARCH_CAPTION_MODEL", "google/gemma-3-4b-it")
    )
    caption_max_new_tokens: int = field(
        default_factory=lambda: _int("AEROSEARCH_CAPTION_MAX_TOKENS", 256)
    )
    caption_load_4bit: bool = field(default_factory=lambda: _bool("AEROSEARCH_CAPTION_4BIT", True))

    # faster-whisper size: "small"/"base" for CPU dev, "large-v3" on the T4.
    transcribe_model: str = field(default_factory=lambda: _str("AEROSEARCH_WHISPER_MODEL", "small"))

    # Embedding model and width are PINNED in aerosearch/pinned.py, because they
    # define the vector space every stored row lives in. The env vars remain as a
    # deliberate escape hatch for experiments; overriding either one makes new
    # vectors incomparable with everything already indexed, so re-embed if you do.
    embed_model: str = field(default_factory=lambda: _str("AEROSEARCH_EMBED_MODEL", EMBED_MODEL))
    embed_dim: int = field(default_factory=lambda: _int("AEROSEARCH_EMBED_DIM", EMBED_DIM))


def _database_url() -> str:
    """Container runs use DATABASE_URL (host `postgres`); host/venv runs prefer
    LOCAL_DATABASE_URL (host `localhost`), mirroring the MinIO endpoint switch."""
    in_docker = os.getenv("RUNNING_IN_DOCKER") is not None
    order = ["DATABASE_URL", "LOCAL_DATABASE_URL"] if in_docker else ["LOCAL_DATABASE_URL", "DATABASE_URL"]
    for name in order:
        val = os.getenv(name)
        if val:
            return val
    return "postgresql://postgres:postgres@localhost:5432/postgres"


@dataclass
class StoreConfig:
    """Where documents + vectors are persisted."""

    database_url: str = field(default_factory=_database_url)
    table: str = field(default_factory=lambda: _str("AEROSEARCH_TABLE", "media"))


@dataclass
class Config:
    media: MediaConfig = field(default_factory=MediaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    store: StoreConfig = field(default_factory=StoreConfig)


def load_config() -> Config:
    return Config()
