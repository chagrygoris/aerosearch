"""multilingual-e5-large embedder via sentence-transformers.

e5 is asymmetric: documents get a "passage: " prefix and queries a "query: "
prefix — skipping this measurably hurts retrieval. Vectors are L2-normalized so
the stores can treat dot product as cosine similarity. Runs on CPU (M2 dev) or
CUDA (T4); ~560M params, comfortably inside 16 GB.
"""
from __future__ import annotations

from functools import cached_property


class E5Embedder:
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large", device: str | None = None):
        self.model_name = model_name
        self._device = device
        self.dim = 1024  # e5-large; overwritten from the model once loaded

    @cached_property
    def _model(self):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(self.model_name, device=self._device)
        self.dim = model.get_sentence_embedding_dimension()
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
        return self._encode(texts, "passage: ")

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text], "query: ")[0]
