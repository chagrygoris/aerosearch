"""EmbeddingGemma 2 embedder — text *and* images in one 768-dim space.

Why this replaces the e5 embedder: e5 is text-only, so a photo was findable only
through a caption, and captioning needs a GPU to be practical (9-16 s/clip on a T4,
several times that on CPU). With captions off, a photo's document was literally
`File: media.` for every photo — identical, so nothing could ever match. Here the
pixels go into the same space as the query text, so a photo is retrievable with no
captioner in the loop at all.

Conventions, all pinned in `aerosearch.pinned`:

* queries get ``task: search result | query: ``,
* documents get ``title: none | text: ``,
* **images get no prefix at all**, and ride alongside the text with one
  ``<|image|>`` placeholder each.

Prefixes are applied here, by hand, rather than through sentence-transformers'
``prompt_name``, so that the strings recorded in the fingerprint are exactly the
strings that reach the model. Any default prompt baked into the model config is
cleared on load for the same reason — otherwise it would be prepended on top of
ours and silently shift the space.

The 300M audio encoder is not loaded: aerosearch has no audio-embedding path (speech
goes through Whisper to text), so text+vision at 440M is all that is needed.
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

from ..pinned import EMBED_DIM, EMBED_MODEL, EMBED_REVISION, PASSAGE_PREFIX, QUERY_PREFIX

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

#: One placeholder per image, as the model card specifies.
IMAGE_TOKEN = "<|image|>"

#: Frames per media row. A row holds one vector, so more frames means more of the
#: clip in that vector but a longer encode; 4 keeps a video's start/middle/end
#: without making each row several times more expensive than a photo.
DEFAULT_MAX_IMAGES = 4


class EmbeddingGemma2Embedder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str | None = None,
        revision: str | None = EMBED_REVISION,
        max_images: int = DEFAULT_MAX_IMAGES,
    ):
        self.model_name = model_name
        self._device = device
        self._revision = revision
        self.max_images = max_images
        self.dim = EMBED_DIM  # confirmed against the loaded model below

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        kwargs: dict[str, Any] = {
            "device": self._device,
            # Text + vision only (440M). Dropping the audio encoder saves 300M
            # params we would never call.
            "config_kwargs": {"audio_config": None},
        }
        if self._revision:
            kwargs["revision"] = self._revision
        model = SentenceTransformer(self.model_name, **kwargs)

        # We prefix by hand; a config default would stack on top of ours.
        if getattr(model, "default_prompt_name", None):
            logger.debug("clearing model default prompt %r", model.default_prompt_name)
            model.default_prompt_name = None

        get_dim = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        actual = get_dim()
        if actual != EMBED_DIM:
            # Writing these would mix two geometries in one column, which degrades
            # ranking without raising anything. Refuse instead.
            raise ValueError(
                f"{self.model_name} emits {actual}-dim vectors but the pinned contract "
                f"is {EMBED_DIM} (aerosearch/pinned.py). Re-embed the index before "
                f"changing the embedding model."
            )
        self.dim = actual
        return model

    def _encode(self, payload, ) -> Any:
        return self._model.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._encode([f"{PASSAGE_PREFIX}{t}" for t in texts])
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"{QUERY_PREFIX}{text}").tolist()

    def embed_media(self, text: str, images: list["Image"]) -> list[float]:
        """One vector for a row that has frames, and optionally speech/caption text.

        Falls back to text-only when there are no usable frames, so a silent clip
        whose frames failed to extract still gets indexed rather than skipped.
        """
        frames = [im for im in (images or []) if im is not None][: self.max_images]
        if not frames:
            return self.embed_documents([text])[0]

        body = (text or "").strip()
        placeholders = " ".join([IMAGE_TOKEN] * len(frames))
        payload = {
            "text": f"{PASSAGE_PREFIX}{body} {placeholders}".strip(),
            "image": frames,
        }
        return self._encode(payload).tolist()

    def resolved_revision(self) -> str | None:
        """The revision in use, so `aerosearch pipeline` can report it."""
        if self._revision:
            return self._revision
        try:
            from huggingface_hub import HfApi

            return HfApi().model_info(self.model_name).sha
        except Exception:
            return None
