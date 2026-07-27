"""Build backends by name so the pipeline stays model-agnostic.

backend="fake" -> instant offline stand-ins (tests, smoke runs)
backend="real" -> e5 + faster-whisper + Gemma 3 (CPU on the M2, CUDA/4-bit on T4)
"""
from __future__ import annotations

from .config import Config
from .understanding.base import Captioner, Embedder, Transcriber


def build_embedder(cfg: Config, backend: str = "real") -> Embedder:
    if backend == "fake":
        from .understanding.fakes import FakeEmbedder

        return FakeEmbedder(dim=256)
    from .understanding.embed_e5 import E5Embedder

    return E5Embedder(cfg.model.embed_model)


def build_captioner(cfg: Config, backend: str = "real") -> Captioner:
    if backend == "fake":
        from .understanding.fakes import FakeCaptioner

        return FakeCaptioner()

    name = cfg.model.caption_model
    # Gemma 3n (E2B/E4B) — small, multimodal, CPU-friendly for the Airflow batch.
    if "3n" in name.lower():
        from .understanding.caption_gemma3n import Gemma3nCaptioner

        return Gemma3nCaptioner(name, max_new_tokens=cfg.model.caption_max_new_tokens)

    # Gemma 3 4B — CUDA 4-bit path for a T4 batch run.
    from .understanding.caption_gemma import GemmaCaptioner

    return GemmaCaptioner(
        name,
        max_new_tokens=cfg.model.caption_max_new_tokens,
        load_4bit=cfg.model.caption_load_4bit,
    )


def build_transcriber(cfg: Config, backend: str = "real") -> Transcriber:
    if backend == "fake":
        from .understanding.fakes import FakeTranscriber

        return FakeTranscriber()
    from .understanding.transcribe_whisper import WhisperTranscriber

    return WhisperTranscriber(cfg.model.transcribe_model)
