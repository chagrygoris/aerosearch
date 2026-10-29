"""Parse a Telegram Desktop chat export into pending rows.

Telegram Desktop's **Export chat history → JSON** produces a folder containing
``result.json`` plus media subfolders (``photos/``, ``video_files/``,
``voice_messages/``, ...). The upload site accepts either the zipped folder or a
bare ``result.json`` (text-only, since a lone JSON carries no media).

Each message becomes one row with a NULL embedding, the same shape the live bot
inserts, so the scheduled GPU run picks both up without knowing the difference.
Media bytes are pushed to object storage; nothing is embedded here.

The export is untrusted input: paths inside the archive are attacker-controlled,
so members are extracted only after checking they stay inside the target
directory (a ``../`` entry would otherwise overwrite files outside it).
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile
from typing import Any, Optional

from .types import MediaDocument

logger = logging.getLogger(__name__)

#: Export keys that point at a media file, mapped to our media_type.
_MEDIA_KEYS = {
    "photo": "photo",
    "file": None,  # resolved from mime_type / media_type below
}

_VIDEO_HINTS = ("video", ".mp4", ".mov", ".webm", ".mkv", ".gif")
_IMAGE_HINTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _safe_extract(zf: zipfile.ZipFile, dest: str) -> None:
    """Extract `zf` into `dest`, refusing any member that escapes it."""
    dest_abs = os.path.realpath(dest)
    for member in zf.infolist():
        target = os.path.realpath(os.path.join(dest, member.filename))
        if not (target == dest_abs or target.startswith(dest_abs + os.sep)):
            raise ValueError(f"refusing archive member outside the target dir: {member.filename!r}")
    zf.extractall(dest)


def _text_of(message: dict[str, Any]) -> str:
    """Flatten Telegram's `text` field, which is a string or a list of runs."""
    raw = message.get("text")
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, list):
        parts = []
        for run in raw:
            if isinstance(run, str):
                parts.append(run)
            elif isinstance(run, dict):
                parts.append(str(run.get("text", "")))
        return "".join(parts).strip()
    return ""


def _media_rel_path(message: dict[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """(relative path, media_type) for a message's media, or (None, None)."""
    for key in ("photo", "file", "thumbnail"):
        rel = message.get(key)
        if not isinstance(rel, str) or not rel or rel == "(File not included. Change data exporting settings to download.)":
            continue
        lowered = rel.lower()
        mime = str(message.get("mime_type", "")).lower()
        kind = str(message.get("media_type", "")).lower()
        if key == "photo" or lowered.endswith(_IMAGE_HINTS):
            return rel, "photo"
        if any(h in lowered for h in _VIDEO_HINTS) or "video" in mime or "video" in kind:
            return rel, "video"
        if mime.startswith("audio") or kind in ("voice_message", "audio_file"):
            # The embedder takes audio directly, so treat it as a clip.
            return rel, "video"
        return None, None  # documents (pdf, zip, ...) are not indexable media
    return None, None


def _find_result_json(root: str) -> str:
    direct = os.path.join(root, "result.json")
    if os.path.isfile(direct):
        return direct
    for dirpath, _dirs, files in os.walk(root):
        if "result.json" in files:
            return os.path.join(dirpath, "result.json")
    raise ValueError("no result.json in the export (use Telegram Desktop → Export chat history → JSON)")


async def ingest_export(path: str, store, chat_id: Optional[int] = None) -> dict[str, int]:
    """Queue every message + media file in the export at `path`.

    `store` needs `add_pending`. Returns counts for the upload page.
    """
    lowered = path.lower()
    with tempfile.TemporaryDirectory(prefix="aerosearch_export_") as work:
        if lowered.endswith(".zip"):
            try:
                with zipfile.ZipFile(path) as zf:
                    _safe_extract(zf, work)
            except zipfile.BadZipFile as exc:
                raise ValueError("not a valid ZIP archive") from exc
            result_path = _find_result_json(work)
        elif lowered.endswith(".json"):
            result_path = path
        else:
            raise ValueError("upload a .zip export or a result.json")

        with open(result_path, encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError as exc:
                raise ValueError(f"result.json is not valid JSON: {exc}") from exc

        export_root = os.path.dirname(result_path)
        export_chat_id = chat_id if chat_id is not None else _export_chat_id(data)
        messages = data.get("messages")
        if not isinstance(messages, list):
            raise ValueError("result.json has no 'messages' list — is this a Telegram export?")

        uploader = _make_uploader()
        queued = skipped = 0
        for message in messages:
            if not isinstance(message, dict) or message.get("type") != "message":
                skipped += 1  # service messages: joins, pins, calls
                continue
            author = str(message.get("from") or "unknown")
            mid = message.get("id")
            rel, media_type = _media_rel_path(message)

            if rel and media_type:
                abs_path = os.path.join(export_root, rel)
                if not os.path.isfile(abs_path):
                    # Export made without media, or a stripped ZIP.
                    skipped += 1
                    continue
                name = f"tg_{mid}_{os.path.basename(rel)}"
                try:
                    s3_url = await uploader(abs_path, name)
                except Exception:
                    logger.exception("upload failed for %s", name)
                    skipped += 1
                    continue
                await store.add_pending(MediaDocument(
                    source=name, media_type=media_type, s3_url=s3_url,
                    chat_id=export_chat_id, user_name=author,
                ))
                queued += 1
                continue

            text = _text_of(message)
            if text:
                await store.add_pending(MediaDocument(
                    source=f"tg_msg_{mid}", media_type="text", document=text,
                    chat_id=export_chat_id, user_name=author,
                ))
                queued += 1
            else:
                skipped += 1

        return {"messages": len(messages), "queued": queued, "skipped": skipped,
                "chat_id": export_chat_id or 0}


def _export_chat_id(data: dict[str, Any]) -> Optional[int]:
    raw = data.get("id")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _make_uploader():
    """Return ``async (local_path, object_name) -> s3_url`` for the configured store."""
    from minio import Minio

    endpoint = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    bucket = os.getenv("MINIO_BUCKET", "media")
    client = Minio(
        endpoint,
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=False,
    )
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

    async def upload(local_path: str, name: str) -> str:
        import asyncio

        await asyncio.to_thread(client.fput_object, bucket, name, local_path)
        return f"{endpoint}/{bucket}/{name}"

    return upload
