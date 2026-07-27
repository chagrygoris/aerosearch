"""Postgres + pgvector store, built on asyncpg to match the existing bot/db code.

Extends the original `media` table with the text columns the semantic index
needs (source/caption/transcript/document). Cosine distance via the `<=>`
operator; similarity is reported as ``1 - distance``.
"""
from __future__ import annotations

from typing import Optional

import asyncpg

from ..types import MediaDocument, SearchHit


def _vec_literal(embedding: list[float]) -> str:
    """pgvector accepts a bracketed text literal cast to ::vector."""
    return "[" + ",".join(f"{x:.7g}" for x in embedding) + "]"


class PgVectorStore:
    def __init__(self, dsn: str, table: str = "media", dim: int = 1024):
        self.dsn = dsn
        self.table = table
        self.dim = dim
        self._pool: Optional[asyncpg.Pool] = None

    async def _get_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(self.dsn)
        return self._pool

    async def init(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            await conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    media_id SERIAL PRIMARY KEY,
                    chat_id BIGINT,
                    user_id BIGINT,
                    user_name TEXT,
                    media_type VARCHAR(50) NOT NULL,
                    s3_url TEXT,
                    source TEXT,
                    caption TEXT,
                    transcript TEXT,
                    document TEXT,
                    embedding vector({self.dim}),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            # Make the store additive over the legacy table (empty embeddings).
            for col, decl in [
                ("source", "TEXT"),
                ("caption", "TEXT"),
                ("transcript", "TEXT"),
                ("document", "TEXT"),
            ]:
                await conn.execute(
                    f"ALTER TABLE {self.table} ADD COLUMN IF NOT EXISTS {col} {decl};"
                )
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_chat_id "
                f"ON {self.table} (chat_id);"
            )
            # Cosine HNSW index for fast ANN search.
            await conn.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{self.table}_embedding "
                f"ON {self.table} USING hnsw (embedding vector_cosine_ops);"
            )

    async def add(self, docs: list[MediaDocument]) -> int:
        pool = await self._get_pool()
        rows = []
        for d in docs:
            if d.embedding is None:
                raise ValueError(f"document {d.source!r} has no embedding")
            rows.append(
                (
                    d.chat_id,
                    d.user_id,
                    d.user_name,
                    d.media_type,
                    d.s3_url,
                    d.source,
                    d.caption,
                    d.transcript,
                    d.document,
                    _vec_literal(d.embedding),
                )
            )
        async with pool.acquire() as conn:
            await conn.executemany(
                f"""
                INSERT INTO {self.table}
                    (chat_id, user_id, user_name, media_type, s3_url,
                     source, caption, transcript, document, embedding)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10::vector)
                """,
                rows,
            )
        return len(rows)

    async def add_pending(self, doc: MediaDocument) -> int:
        """Insert a row with NO embedding yet — the bot's ingest path.

        The scheduled BatchIndexer fills in caption/transcript/embedding later.
        `document` may already hold text (for text messages). Returns media_id.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            media_id = await conn.fetchval(
                f"""
                INSERT INTO {self.table}
                    (chat_id, user_id, user_name, media_type, s3_url, source, document)
                VALUES ($1,$2,$3,$4,$5,$6,$7)
                RETURNING media_id
                """,
                doc.chat_id, doc.user_id, doc.user_name, doc.media_type,
                doc.s3_url, doc.source, doc.document or None,
            )
        return int(media_id)

    async def fetch_unindexed(self, limit: int = 100) -> list[dict]:
        """Rows the bot inserted but the batch job hasn't embedded yet.

        Covers both media (needs Gemma/Whisper) and plain text messages (just
        need embedding). Identified by a NULL embedding.
        """
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            records = await conn.fetch(
                f"""
                SELECT media_id, chat_id, media_type, s3_url, source, document
                FROM {self.table}
                WHERE embedding IS NULL
                ORDER BY media_id
                LIMIT {int(limit)}
                """
            )
        return [dict(r) for r in records]

    async def update_indexed(self, media_id: int, doc: MediaDocument) -> None:
        """Write the understood text + embedding back onto an existing row."""
        if doc.embedding is None:
            raise ValueError("cannot mark a row indexed without an embedding")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                f"""
                UPDATE {self.table}
                SET caption = $2, transcript = $3, document = $4, embedding = $5::vector
                WHERE media_id = $1
                """,
                media_id,
                doc.caption,
                doc.transcript,
                doc.document,
                _vec_literal(doc.embedding),
            )

    async def search(
        self, query_embedding: list[float], k: int = 5, chat_id: Optional[int] = None
    ) -> list[SearchHit]:
        pool = await self._get_pool()
        qv = _vec_literal(query_embedding)
        where = "WHERE embedding IS NOT NULL"
        params: list = [qv]
        if chat_id is not None:
            where += " AND chat_id = $2"
            params.append(chat_id)
        # $1 is the query vector; k is interpolated (safe int) into LIMIT.
        sql = f"""
            SELECT source, media_type, document, s3_url, user_name,
                   1 - (embedding <=> $1::vector) AS score
            FROM {self.table}
            {where}
            ORDER BY embedding <=> $1::vector
            LIMIT {int(k)}
        """
        async with pool.acquire() as conn:
            records = await conn.fetch(sql, *params)
        return [
            SearchHit(
                source=r["source"] or "",
                media_type=r["media_type"],
                score=float(r["score"]),
                document=r["document"] or "",
                s3_url=r["s3_url"],
                user_name=r["user_name"],
            )
            for r in records
        ]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
