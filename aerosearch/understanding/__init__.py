"""Model-facing interfaces plus swappable backends.

The pipeline depends only on the Protocols in `base`, so the same code runs
with fake backends (fast, offline unit tests), CPU backends (M2 dev), or the
quantized Gemma/Whisper stack on a T4 — chosen by `factory.build_*`.
"""
from .base import Captioner, Transcriber, Embedder

__all__ = ["Captioner", "Transcriber", "Embedder"]
