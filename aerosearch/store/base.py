from __future__ import annotations

from typing import Optional, Protocol

from ..types import MediaDocument, SearchHit


class VectorStore(Protocol):
    async def init(self) -> None:
        """Create schema / indexes if needed."""

    async def add(self, docs: list[MediaDocument]) -> int:
        """Persist documents (each must have `.embedding` set). Returns count."""

    async def search(
        self, query_embedding: list[float], k: int = 5, chat_id: Optional[int] = None
    ) -> list[SearchHit]:
        """Return the top-k most similar documents by cosine similarity."""

    async def close(self) -> None:
        ...
