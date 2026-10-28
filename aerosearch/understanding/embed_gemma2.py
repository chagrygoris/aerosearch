"""EmbeddingGemma 2 — the only model in the retrieval path.

Text, images, video and audio all land in one 768-dim cosine-comparable space, so a
media file is handed to this embedder as it is: no captioner, no ASR, no ffmpeg frame
or wav extraction. A text query then reaches the pixels and the speech directly.

Measured here before adopting it (synthetic inputs, CPU):

* text query -> **speech audio**, no transcription: 3/3
  ("political speech by the president" -> the spoken clip at 0.828);
* Russian text query -> image: 4/4, including Cyrillic rendered *inside* the image;
* Russian text -> text: 3/3.

Conventions, all pinned in `aerosearch.pinned`:

* queries get ``task: search result | query: ``,
* text documents get ``title: none | text: ``,
* **media carries no prefix at all** and is passed by its own key.

Prefixes are applied here, by hand, rather than through sentence-transformers'
``prompt_name``, so that the strings recorded in the fingerprint are exactly the
strings that reach the model — verified equal to the model's own ``SearchQuery`` and
``Document`` prompts. Any default prompt baked into the model config is cleared on
load for the same reason.

Two things learned the hard way, both load-bearing:

* ``{"video": path}`` alone does **not** pull in the file's audio track. Measured on
  a clip of synthesized speech over a test pattern, "a cat sleeping" scored 0.626
  video-only versus 0.729 when ``audio`` was passed too. Both keys are needed, so
  `embed_file` passes both and falls back to video-only when the file has no
  decodable audio.
* Audio needs ``librosa``, and audio *inside* an mp4 needs ``torchcodec``; without
  them the call raises at encode time rather than at load time.

Budget: 8192 tokens shared across all inputs — 140 per video frame at 1 fps (~58
frames) and 25 per audio second (~327 s). Longer inputs were accepted without error
in testing (400 s audio embedded fine), so assume silent truncation rather than a
raised limit.
"""
from __future__ import annotations

import logging
from functools import cached_property
from typing import TYPE_CHECKING, Any

from ..pinned import EMBED_DIM, EMBED_MODEL, EMBED_REVISION, PASSAGE_PREFIX, QUERY_PREFIX

if TYPE_CHECKING:
    from PIL.Image import Image

logger = logging.getLogger(__name__)

#: One placeholder per attached image, as the model card specifies.
IMAGE_TOKEN = "<|image|>"

#: Frames per media row when PIL frames are passed instead of a file path. A row
#: holds one vector, so more frames means more of the clip in that vector but a
#: longer encode.
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

        # The full 740M model: text + vision + audio. Every encoder is used, since
        # speech is embedded directly rather than transcribed first.
        kwargs: dict[str, Any] = {"device": self._device}
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

    def _encode(self, payload) -> Any:
        return self._model.encode(
            payload,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    # --- text ---------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"{PASSAGE_PREFIX}{t}" for t in texts]).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._encode(f"{QUERY_PREFIX}{text}").tolist()

    # --- media --------------------------------------------------------------

    def embed_file(self, path: str, media_type: str, text: str = "") -> list[float]:
        """Embed a media file directly — the baseline path for photos and video.

        `media_type` is "photo" or "video"; anything else is treated as text. The
        returned vector is comparable with `embed_query` output, so a text query
        retrieves the file without it ever having been captioned or transcribed.
        """
        if media_type == "photo":
            return self._encode({"image": path}).tolist()

        if media_type == "video":
            # Both keys: video alone leaves the speech out (see module docstring).
            try:
                return self._encode({"video": path, "audio": path}).tolist()
            except Exception as exc:
                # Silent clips have no audio stream to decode; still embed the
                # visuals rather than dropping the row.
                logger.info("no decodable audio in %s (%s); embedding video only", path, exc)
                return self._encode({"video": path}).tolist()

        return self.embed_documents([text or path])[0]

    def embed_media(self, text: str, images: list["Image"]) -> list[float]:
        """One vector from already-extracted PIL frames plus text.

        Used by the enriched path, where frames have been sampled and a caption or
        transcript already exists. The baseline uses `embed_file` instead and never
        decodes frames itself.
        """
        frames = [im for im in (images or []) if im is not None][: self.max_images]
        if not frames:
            return self.embed_documents([text])[0]

        placeholders = " ".join([IMAGE_TOKEN] * len(frames))
        payload = {
            "text": f"{PASSAGE_PREFIX}{(text or '').strip()} {placeholders}".strip(),
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
