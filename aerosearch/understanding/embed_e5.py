"""multilingual-e5-large embedder via sentence-transformers.

e5 is asymmetric: documents get a "passage: " prefix and queries a "query: "
prefix — skipping this measurably hurts retrieval. Vectors are L2-normalized so
the stores can treat dot product as cosine similarity. Runs on CPU (M2 dev) or
CUDA (T4); ~560M params, comfortably inside 16 GB.

Model name, revision, prefixes and dimension all come from `aerosearch.pinned`,
which explains why they are frozen: they define the vector space every stored row
lives in.
"""
from __future__ import annotations

from functools import cached_property

from ..pinned import EMBED_DIM, EMBED_MODEL, EMBED_REVISION, PASSAGE_PREFIX, QUERY_PREFIX


class E5Embedder:
    def __init__(
        self,
        model_name: str = EMBED_MODEL,
        device: str | None = None,
        revision: str | None = EMBED_REVISION,
    ):
        self.model_name = model_name
        self._device = device
        self._revision = revision
        self.dim = EMBED_DIM  # confirmed against the loaded model below

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": self._device}
        if self._revision:
            kwargs["revision"] = self._revision
        model = SentenceTransformer(self.model_name, **kwargs)

        # Renamed in newer sentence-transformers; keep working on both.
        get_dim = getattr(model, "get_embedding_dimension", None) or (
            model.get_sentence_embedding_dimension
        )
        actual = get_dim()
        if actual != EMBED_DIM:
            # Writing these vectors would mix two geometries in one table, which
            # degrades ranking without raising anything. Refuse instead.
            raise ValueError(
                f"{self.model_name} emits {actual}-dim vectors but the pinned contract "
                f"is {EMBED_DIM} (aerosearch/pinned.py). Re-embed the index before "
                f"changing the embedding model."
            )
        self.dim = actual
        return model

    def _encode(self, texts: list[str], prefix: str) -> list[list[float]]:
        prefixed = [f"{prefix}{t}" for t in texts]
        vecs = self._model.encode(
            prefixed,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts, PASSAGE_PREFIX)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], QUERY_PREFIX)[0]

    def resolved_revision(self) -> str | None:
        """The revision actually on disk, so `aerosearch pipeline` can report it.

        Returns None when the model has not been downloaded yet or the local copy
        carries no revision metadata.
        """
        if self._revision:
            return self._revision
        try:
            from huggingface_hub import HfApi

            return HfApi().model_info(self.model_name).sha
        except Exception:
            return None
