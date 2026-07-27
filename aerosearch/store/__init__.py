"""Vector stores. Async interface to match the aiogram/asyncpg codebase."""
from .base import VectorStore
from .memory import MemoryVectorStore

__all__ = ["VectorStore", "MemoryVectorStore", "get_pgvector_store"]


def get_pgvector_store(*args, **kwargs):
    """Lazy import so the CLI/tests don't require asyncpg unless used."""
    from .pgvector import PgVectorStore

    return PgVectorStore(*args, **kwargs)
