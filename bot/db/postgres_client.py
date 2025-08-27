import asyncpg
import dotenv 
import os 
import asyncio
import logging

dotenv.load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LOCAL_DATABASE_URL = os.getenv("LOCAL_DATABASE_URL")

logger = logging.getLogger(__name__)

async def get_db_pool():
    """Attempt to connect to PostgreSQL using DATABASE_URL, fall back to LOCAL_DATABASE_URL."""
    for db_url in [DATABASE_URL, LOCAL_DATABASE_URL]:
        try:
            pool = await asyncpg.create_pool(db_url)
            logger.info(f"Connected to PostgreSQL using {db_url}")
            return pool
        except Exception as e:
            logger.warning(f"Failed to connect to {db_url}: {e}")
    raise Exception("Could not connect to any PostgreSQL database")

async def init_db():
    """Initialize the PostgreSQL database and create media table with embedding column."""
    async with await get_db_pool() as pool:
        async with pool.acquire() as conn:
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS vector;
                CREATE TABLE IF NOT EXISTS media (
                    media_id SERIAL PRIMARY KEY,
                    chat_id BIGINT NOT NULL,
                    user_id BIGINT NOT NULL,
                    user_name TEXT NOT NULL,
                    media_type VARCHAR(50) NOT NULL,
                    s3_url TEXT NOT NULL,
                    embedding vector(1536),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_media_chat_id ON media (chat_id);
            """)

async def save_media(chat_id: int, user_id: int, user_name: str, media_type: str, s3_url: str, embedding: list = None):
    """Save media metadata and optional embedding to the database."""
    async with await get_db_pool() as pool:
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO media (chat_id, user_id, user_name, media_type, s3_url, embedding)
                VALUES ($1, $2, $3, $4, $5, $6)
                """,
                chat_id, user_id, user_name, media_type, s3_url, embedding
            )
    return True