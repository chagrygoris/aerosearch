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

import httpx

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
if not BOT_TOKEN:
    # Fail with the fix rather than aiogram's opaque validation error.
    raise SystemExit(
        "API_KEY is not set — that's your Telegram bot token from @BotFather.\n"
        "  docker: echo 'API_KEY=123456:your-token' > .env && docker compose up -d\n"
        "  local : export API_KEY=123456:your-token"
    )
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

cfg = load_config()
store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim)

# The bot holds NO model. Query embedding and search live in the retrieval
# server, so this process stays small and starts instantly; the server owns the
# text-only embedder, and media embedding happens on the GPU pool.
SERVER = os.getenv("AEROSEARCH_SERVER", "http://server:8000").rstrip("/")


async def _server_get(path: str, **params):
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{SERVER}{path}", params=params)
        r.raise_for_status()
        return r.json()


async def _server_post(path: str, **data):
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{SERVER}{path}", data=data)
        r.raise_for_status()
        return r.json()


async def _pending_count(chat_id: int) -> int:
    """Unindexed messages still waiting on the scheduled GPU run."""
    try:
        return int((await _server_get("/stats", chat_id=chat_id)).get("pending", 0))
    except Exception:
        logger.warning("could not read /stats from %s", SERVER, exc_info=True)
        return -1


def _display_name(user: types.User) -> str:
    return user.username or f"{user.first_name} {user.last_name or ''}".strip()


@dp.message(Command("start", "help"))
async def handle_start(message: types.Message) -> None:
    await message.answer(
        "👋 I index this chat's photos, videos and messages for semantic search.\n\n"
        "/search <what you remember> — find media by meaning\n"
        "/stats — how much is indexed vs still unindexed\n\n"
        "Bulk-import a chat export at the upload page; indexing runs on the GPU "
        "pool, scheduled by Airflow."
    )


@dp.message(Command("stats"))
async def handle_stats(message: types.Message) -> None:
    """Indexed vs still-unindexed counts for this chat."""
    try:
        c = await _server_get("/stats", chat_id=message.chat.id)
    except Exception:
        await message.answer("Can't reach the retrieval server right now.")
        return

    if not c.get("indexed") and not c.get("pending"):
        await message.answer("Nothing from this chat yet. Send a photo, video or message.")
        return

    lines = [f"📊 <b>{c.get('indexed', 0)}</b> searchable, "
             f"<b>{c.get('pending', 0)}</b> not indexed yet."]
    for kind in ("photo", "video", "text"):
        got, wait = c.get(f"{kind}_indexed", 0), c.get(f"{kind}_pending", 0)
        if got or wait:
            lines.append(f"  {kind}: {got} indexed, {wait} pending")
    if c.get("pending"):
        lines.append("\nIndexing runs on the GPU pool — trigger it from the Airflow UI.")
    await message.answer("\n".join(lines), parse_mode="HTML")


@dp.message(Command("search"))
async def handle_search(message: types.Message) -> None:
    query = message.text.split(maxsplit=1)
    if len(query) < 2:
        await message.answer("Usage: /search <query>")
        return
    query = query[1]

    try:
        result = await _server_post("/search", query=query, k=5, chat_id=message.chat.id)
    except Exception:
        logger.exception("search failed")
        await message.answer("Can't reach the retrieval server right now.")
        return

    hits = result.get("hits", [])
    pending = await _pending_count(message.chat.id)

    if not hits:
        note = f" {pending} item(s) are still waiting to be indexed." if pending > 0 else ""
        await message.answer(f"Nothing indexed matches that yet.{note}")
        return

    lines = [f"🔎 Top matches for <b>{query}</b>:"]
    for i, h in enumerate(hits, 1):
        snippet = (h.get("document") or "").replace("\n", " ")[:120]
        lines.append(f"{i}. <i>{h['media_type']}</i> ({h['score']:.2f}) — {snippet}")
        if h.get("s3_url"):
            lines.append(f"    {h['s3_url']}")
    if pending > 0:
        lines.append(f"\n<i>{pending} item(s) not indexed yet.</i>")
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
        # aiogram 3's File has no download_* methods — go through the bot, same as
        # the photo branch above. (Was File.download_as_bytes(), which never existed
        # on aiogram 3 and crashed every video ingest.)
        file_bytes = (await bot.download_file(file.file_path)).read()
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
