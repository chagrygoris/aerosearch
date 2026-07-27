"""Vision captioner built on Gemma 3 4B-IT (the smallest *multimodal* Gemma 3).

Given the frames sampled from a clip, it writes a compact, search-oriented
description: who/what is on screen, the action, setting, mood, and any visible
on-screen text transcribed verbatim (memes often live in the text). This is what
makes silent clips searchable — Whisper only hears speech, Gemma sees.

Runtime:
- CUDA (Kaggle T4): 4-bit NF4 via bitsandbytes -> ~3-4 GB, fits 16 GB with room.
- CPU/MPS (M2 dev): falls back to bf16/fp16, slow but functional for spot checks.
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an assistant that describes short video clips and images so they "
    "can be found later by semantic search. Be concise, concrete and factual."
)

VIDEO_PROMPT = (
    "These frames are sampled in order from one short video clip. In 2-4 "
    "sentences describe what happens: the people or characters, their actions, "
    "the setting, notable objects, and the overall mood or genre (e.g. meme, "
    "dance, speech, sport). If any text, captions or logos are visible, quote "
    "them verbatim. Do not mention that these are frames."
)

IMAGE_PROMPT = (
    "Describe this image in 1-3 sentences for search: the subject, action, "
    "setting and mood. Quote any visible text verbatim."
)


class GemmaCaptioner:
    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",
        max_new_tokens: int = 256,
        load_4bit: bool = True,
        device_map: str = "auto",
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.load_4bit = load_4bit
        self.device_map = device_map

    @cached_property
    def _cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            return False

    @cached_property
    def _bundle(self):
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        kwargs: dict = {}
        if self.load_4bit and self._cuda:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            kwargs["device_map"] = self.device_map
        else:
            if self.load_4bit and not self._cuda:
                logger.warning("4-bit requested but no CUDA — loading in bf16 instead")
            kwargs["torch_dtype"] = torch.bfloat16
            kwargs["device_map"] = self.device_map

        logger.info("loading %s (4bit=%s, cuda=%s)", self.model_name, self.load_4bit and self._cuda, self._cuda)
        processor = AutoProcessor.from_pretrained(self.model_name)
        model = Gemma3ForConditionalGeneration.from_pretrained(self.model_name, **kwargs).eval()
        return processor, model

    def caption(self, frames: list["Image"], media_type: str) -> str:
        if not frames:
            return ""
        import torch

        processor, model = self._bundle
        prompt = IMAGE_PROMPT if media_type == "photo" else VIDEO_PROMPT

        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": prompt})
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
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        new_tokens = generation[0][input_len:]
        text = processor.decode(new_tokens, skip_special_tokens=True)
        return text.strip()
