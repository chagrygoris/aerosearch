"""Parsing a Telegram Desktop export into pending rows.

The export is user-supplied and its paths are attacker-controlled, so the Zip
traversal guard is tested as carefully as the happy path.
"""
import json
import os
import zipfile

import pytest

from aerosearch.telegram_export import (
    _media_rel_path,
    _safe_extract,
    _text_of,
    ingest_export,
)


class FakeStore:
    def __init__(self):
        self.rows = []

    async def add_pending(self, doc):
        self.rows.append(doc)
        return len(self.rows)


@pytest.fixture
def uploader(monkeypatch):
    """Stand in for object storage; records what would have been uploaded."""
    uploaded = []

    def _make():
        async def upload(local_path, name):
            uploaded.append(name)
            return f"minio:8000/media/{name}"

        return upload

    monkeypatch.setattr("aerosearch.telegram_export._make_uploader", _make)
    return uploaded


def _export(tmp_path, messages, with_media=None):
    """Build a Telegram-shaped export dir; returns the zip path."""
    root = tmp_path / "ChatExport"
    (root / "photos").mkdir(parents=True)
    for rel in with_media or []:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 32)
    (root / "result.json").write_text(
        json.dumps({"name": "chat", "id": 4242, "messages": messages}), encoding="utf-8"
    )
    zpath = tmp_path / "export.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for dirpath, _d, files in os.walk(root):
            for f in files:
                full = os.path.join(dirpath, f)
                zf.write(full, os.path.relpath(full, tmp_path))
    return str(zpath)


def test_text_field_is_a_string_or_a_list_of_runs():
    assert _text_of({"text": "  hi  "}) == "hi"
    # Telegram splits formatted messages into runs; links become dicts.
    assert _text_of({"text": ["look at ", {"type": "link", "text": "this"}]}) == "look at this"
    assert _text_of({"text": []}) == ""
    assert _text_of({}) == ""


def test_media_type_detection():
    assert _media_rel_path({"photo": "photos/x.jpg"}) == ("photos/x.jpg", "photo")
    assert _media_rel_path({"file": "video_files/x.mp4"})[1] == "video"
    # Voice notes are embedded as audio clips, not transcribed.
    assert _media_rel_path({"file": "voice_messages/x.ogg",
                            "media_type": "voice_message"})[1] == "video"
    # Non-media documents are not indexable.
    assert _media_rel_path({"file": "files/report.pdf",
                            "mime_type": "application/pdf"}) == (None, None)
    # Telegram's placeholder for an export made without media.
    assert _media_rel_path({
        "photo": "(File not included. Change data exporting settings to download.)"
    }) == (None, None)


def test_zip_traversal_is_refused(tmp_path):
    """A `../` member must not be allowed to write outside the extract dir."""
    zpath = tmp_path / "evil.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        zf.writestr("../escaped.txt", "pwned")
    dest = tmp_path / "dest"
    dest.mkdir()
    with zipfile.ZipFile(zpath) as zf:
        with pytest.raises(ValueError, match="outside the target dir"):
            _safe_extract(zf, str(dest))
    assert not (tmp_path / "escaped.txt").exists()


async def test_ingest_queues_text_and_media(tmp_path, uploader):
    messages = [
        {"id": 1, "type": "message", "from": "grig", "text": "привет как дела"},
        {"id": 2, "type": "message", "from": "grig", "photo": "photos/cat.jpg", "text": ""},
        {"id": 3, "type": "service", "action": "pin_message"},
        {"id": 4, "type": "message", "from": "grig", "text": ""},
    ]
    zpath = _export(tmp_path, messages, with_media=["photos/cat.jpg"])
    store = FakeStore()

    out = await ingest_export(zpath, store, chat_id=None)

    assert out["messages"] == 4
    assert out["queued"] == 2          # one text, one photo
    assert out["skipped"] == 2         # service message + empty text
    assert out["chat_id"] == 4242      # read from the export

    kinds = sorted(r.media_type for r in store.rows)
    assert kinds == ["photo", "text"]
    assert all(r.embedding is None for r in store.rows), "ingest must not embed"
    photo = next(r for r in store.rows if r.media_type == "photo")
    assert photo.s3_url and photo.s3_url.endswith(".jpg")
    assert uploader == [photo.source]


async def test_media_missing_from_the_archive_is_skipped(tmp_path, uploader):
    """An export made without media references files that are not there."""
    messages = [{"id": 9, "type": "message", "from": "g", "photo": "photos/gone.jpg"}]
    zpath = _export(tmp_path, messages)  # no media written
    store = FakeStore()
    out = await ingest_export(zpath, store)
    assert out["queued"] == 0 and out["skipped"] == 1
    assert store.rows == []


async def test_bare_result_json_is_accepted(tmp_path, uploader):
    p = tmp_path / "result.json"
    p.write_text(json.dumps({"id": 7, "messages": [
        {"id": 1, "type": "message", "from": "g", "text": "hello"}]}), encoding="utf-8")
    store = FakeStore()
    out = await ingest_export(str(p), store)
    assert out["queued"] == 1 and store.rows[0].media_type == "text"


async def test_rejects_things_that_are_not_exports(tmp_path, uploader):
    bad = tmp_path / "notes.txt"
    bad.write_text("nope")
    with pytest.raises(ValueError, match="zip export or a result.json"):
        await ingest_export(str(bad), FakeStore())

    nojson = tmp_path / "empty.zip"
    with zipfile.ZipFile(nojson, "w") as zf:
        zf.writestr("readme.txt", "hi")
    with pytest.raises(ValueError, match="no result.json"):
        await ingest_export(str(nojson), FakeStore())
