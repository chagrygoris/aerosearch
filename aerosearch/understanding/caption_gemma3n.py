"""Vision captioner on **Gemma 3n E2B** — small, multimodal, built for CPU/edge.

Why this exists:
- It's the CPU-friendly captioner for the Airflow batch indexer (no always-on
  GPU). Gemma 3n E2B has ~2B *effective* params (MatFormer), so it runs on CPU
  for a scheduled batch job, unlike the 4B model at bf16.
- It's the A/B alternative to the Whisper+Gemma-3-4B combo. Gemma 3n also has an
  audio encoder, so `caption_av` can fold speech in with a single model — handy
  for benchmarking a one-model pipeline on src-videos later.

Same `Captioner` protocol (`caption(frames, media_type)`), so it's a drop-in via
the factory. Set AEROSEARCH_CAPTION_MODEL=google/gemma-3n-E2B-it to select it.
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Optional

from .caption_gemma import IMAGE_PROMPT, SYSTEM_PROMPT, VIDEO_PROMPT

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

# When folding audio in with the same model (single-model experiment).
AV_PROMPT = (
    VIDEO_PROMPT
    + " You are also given the clip's audio; use what is said to inform the "
    "description and include a short paraphrase of any speech."
)


class Gemma3nCaptioner:
    def __init__(
        self,
        model_name: str = "google/gemma-3n-E2B-it",
        max_new_tokens: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._device = device

    @cached_property
    def _resolved_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @cached_property
    def _bundle(self):
        import torch
        from transformers import AutoProcessor, Gemma3nForConditionalGeneration

        device = self._resolved_device
        # CPU has no usable bf16 matmul path; float32 is the safe choice there.
        dtype = torch.float32 if device == "cpu" else torch.bfloat16
        logger.info("loading %s on %s (%s)", self.model_name, device, dtype)

        processor = AutoProcessor.from_pretrained(self.model_name)
        model = (
            Gemma3nForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=dtype
            )
            .to(device)
            .eval()
        )
        return processor, model

    def _generate(self, content: list[dict]) -> str:
        import torch

        processor, model = self._bundle
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content},
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        new_tokens = generation[0][input_len:]
        return processor.decode(new_tokens, skip_special_tokens=True).strip()

    def caption(self, frames: list["Image"], media_type: str) -> str:
        if not frames:
            return ""
        prompt = IMAGE_PROMPT if media_type == "photo" else VIDEO_PROMPT
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
        return self._generate(content)

    def caption_av(self, frames: list["Image"], audio_path: str, media_type: str) -> str:
        """Single-model vision+audio caption (Gemma 3n hears too).

        Used by the one-model A/B experiment to skip Whisper entirely.
        """
        content: list[dict] = [{"type": "image", "image": img} for img in frames]
        if audio_path:
            content.append({"type": "audio", "audio": audio_path})
        content.append({"type": "text", "text": AV_PROMPT})
        return self._generate(content)
