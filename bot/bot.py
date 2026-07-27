"""Telegram frontend.

Two jobs, both lightweight — the heavy understanding happens later in the
Airflow batch worker (see airflow/dags/index_pending.py):

  1. Ingest: capture text messages + photos/videos from the chat, push media to
     MinIO, and insert a *pending* row (NULL embedding) for each.
  2. Search: `/search <query>` embeds the query with e5 (CPU) and runs a cosine
     search over already-indexed rows in pgvector.
"""
import asyncio
import logging
import os

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv

from aerosearch.config import load_config
from aerosearch.store import get_pgvector_store
from aerosearch.types import MediaDocument
from db import upload_to_minio

load_dotenv(dotenv_path="../.env")

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

BOT_TOKEN = os.getenv("API_KEY")
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

cfg = load_config()
store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim)

# e5 query embedder is loaded lazily on the first /search so bot startup stays fast.
_embedder = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from aerosearch.understanding.embed_e5 import E5Embedder

        _embedder = E5Embedder(cfg.model.embed_model)
    return _embedder


def _display_name(user: types.User) -> str:
    return user.username or f"{user.first_name} {user.last_name or ''}".strip()


@dp.message(Command("start", "help"))
async def handle_start(message: types.Message) -> None:
    await message.answer(
        "👋 I index this chat's photos, videos and messages for semantic search.\n"
        "Use /search <what you remember> to find media by meaning."
    )


@dp.message(Command("search"))
async def handle_search(message: types.Message) -> None:
    query = message.text.split(maxsplit=1)
    if len(query) < 2:
        await message.answer("Usage: /search <query>")
        return
    query = query[1]

    embedder = get_embedder()
    qv = await asyncio.to_thread(embedder.embed_query, query)
    hits = await store.search(qv, k=5, chat_id=message.chat.id)

    if not hits:
        await message.answer("Nothing indexed matches that yet. (Indexing runs periodically.)")
        return

    lines = [f"🔎 Top matches for <b>{query}</b>:"]
    for i, h in enumerate(hits, 1):
        snippet = (h.document or "").replace("\n", " ")[:120]
        lines.append(f"{i}. <i>{h.media_type}</i> ({h.score:.2f}) — {snippet}")
        if h.s3_url:
            lines.append(f"    {h.s3_url}")
    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message()
async def handle_message(message: types.Message) -> None:
    """Ingest anything that isn't a command as a pending row."""
    chat_id = message.chat.id
    user = message.from_user
    user_name = _display_name(user)

    if message.photo:
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_name = f"photo_{message.message_id}_{photo.file_unique_id}.jpg"
        file_bytes = (await bot.download_file(file.file_path)).read()
        s3_url = await upload_to_minio(file_bytes, file_name)
        await store.add_pending(
            MediaDocument(source=file_name, media_type="photo", s3_url=s3_url,
                          chat_id=chat_id, user_id=user.id, user_name=user_name)
        )
        logger.info("queued photo %s", file_name)

    elif message.video:
        video = message.video
        file = await bot.get_file(video.file_id)
        file_name = f"video_{message.message_id}_{video.file_unique_id}.mp4"
        file_bytes = await file.download_as_bytes()
        s3_url = await upload_to_minio(file_bytes, file_name)
        await store.add_pending(
            MediaDocument(source=file_name, media_type="video", s3_url=s3_url,
                          chat_id=chat_id, user_id=user.id, user_name=user_name)
        )
        logger.info("queued video %s", file_name)

    elif message.text:
        await store.add_pending(
            MediaDocument(source=f"msg_{message.message_id}", media_type="text",
                          document=message.text, chat_id=chat_id,
                          user_id=user.id, user_name=user_name)
        )
        logger.info("queued text message from %s", user_name)


async def main() -> None:
    logger.info("Initializing store")
    await store.init()
    logger.info("Starting bot…")
    await dp.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    asyncio.run(main())
