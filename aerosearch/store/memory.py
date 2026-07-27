"""In-memory numpy store — for tests, quick demos and the no-Postgres dev loop.

Optionally persists to a .npz + .json sidecar so `aerosearch index` and
`aerosearch search` can be separate CLI invocations without a database.
"""
from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from ..types import MediaDocument, SearchHit


class MemoryVectorStore:
    def __init__(self, path: Optional[str] = None):
        # `path` is a base path; state lives in `{path}.json` + `{path}.npz`.
        self.path = path
        self._docs: list[MediaDocument] = []
        self._matrix: Optional[np.ndarray] = None  # (N, dim), rows L2-normalized
        if path and os.path.exists(path + ".json"):
            self._load()

    async def init(self) -> None:  # nothing to create
        return None

    async def add(self, docs: list[MediaDocument]) -> int:
        vecs = []
        for d in docs:
            if d.embedding is None:
                raise ValueError(f"document {d.source!r} has no embedding")
            self._docs.append(d)
            vecs.append(np.asarray(d.embedding, dtype=np.float32))
        if vecs:
            block = np.vstack(vecs)
            self._matrix = block if self._matrix is None else np.vstack([self._matrix, block])
        if self.path:
            self._save()
        return len(docs)

    async def search(
        self, query_embedding: list[float], k: int = 5, chat_id: Optional[int] = None
    ) -> list[SearchHit]:
        if self._matrix is None or not self._docs:
            return []
        q = np.asarray(query_embedding, dtype=np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        scores = self._matrix @ q  # rows are already L2-normalized -> cosine

        order = np.argsort(-scores)
        hits: list[SearchHit] = []
        for i in order:
            doc = self._docs[i]
            if chat_id is not None and doc.chat_id != chat_id:
                continue
            hits.append(
                SearchHit(
                    source=doc.source,
                    media_type=doc.media_type,
                    score=float(scores[i]),
                    document=doc.document,
                    s3_url=doc.s3_url,
                    user_name=doc.user_name,
                )
            )
            if len(hits) >= k:
                break
        return hits

    async def close(self) -> None:
        return None

    # --- persistence ---
    def _save(self) -> None:
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        meta = [
            {
                "source": d.source,
                "media_type": d.media_type,
                "caption": d.caption,
                "transcript": d.transcript,
                "document": d.document,
                "chat_id": d.chat_id,
                "user_id": d.user_id,
                "user_name": d.user_name,
                "s3_url": d.s3_url,
            }
            for d in self._docs
        ]
        with open(self.path + ".json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self._matrix is not None:
            np.savez_compressed(self.path, matrix=self._matrix)

    def _load(self) -> None:
        with open(self.path + ".json", encoding="utf-8") as f:
            meta = json.load(f)
        npz_path = self.path + ".npz"
        matrix = np.load(npz_path)["matrix"] if os.path.exists(npz_path) else None
        self._docs = [
            MediaDocument(
                source=m["source"],
                media_type=m["media_type"],
                caption=m.get("caption", ""),
                transcript=m.get("transcript", ""),
                document=m.get("document", ""),
                embedding=(matrix[i].tolist() if matrix is not None else None),
                chat_id=m.get("chat_id"),
                user_id=m.get("user_id"),
                user_name=m.get("user_name"),
                s3_url=m.get("s3_url"),
            )
            for i, m in enumerate(meta)
        ]
        self._matrix = matrix
