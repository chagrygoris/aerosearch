"""Speech-to-text via faster-whisper (CTranslate2 — much faster/lighter than
openai-whisper, and int8 on CPU makes the M2 dev loop usable).

Model size is configurable: "small"/"base" for CPU dev, "large-v3" on the T4.
Returns '' when there's no detectable speech, which keeps silent clips (dances,
memes) out of the transcript field so the vision caption carries them.
"""
from __future__ import annotations

import logging
from functools import cached_property

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_size: str = "small", device: str | None = None, compute_type: str | None = None):
        self.model_size = model_size
        self._device = device
        self._compute_type = compute_type

    @cached_property
    def _model(self):
        from faster_whisper import WhisperModel

        device = self._device
        compute_type = self._compute_type
        if device is None:
            # Auto: CUDA float16 when available, else CPU int8.
            try:
                import torch

                if torch.cuda.is_available():
                    device, compute_type = "cuda", compute_type or "float16"
                else:
                    device, compute_type = "cpu", compute_type or "int8"
            except Exception:
                device, compute_type = "cpu", compute_type or "int8"
        logger.info("loading faster-whisper %s on %s (%s)", self.model_size, device, compute_type)
        return WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio_path: str) -> str:
        segments, info = self._model.transcribe(audio_path, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text
