"""Retrieval server: the only thing that holds a model locally.

Runs on your machine and owns three jobs:

* ``POST /search`` -- embed a query with the **text-only** EmbeddingGemma 2 tower
  (~270M of 740M; a search box never embeds pixels) and run the pgvector cosine
  search. The bot calls this over HTTP rather than loading a model itself.
* ``GET /stats`` -- indexed vs pending counts, which the bot shows as the
  unindexed counter and the upload page shows as a badge.
* ``GET /`` + ``POST /upload`` -- the upload site: hand it a Telegram Desktop
  export and it queues every message and media file for indexing.

What it deliberately does NOT do is embed media. That needs the vision and audio
encoders and real compute, so it happens on a Kaggle T4, dispatched by the local
Airflow DAG (``airflow/dags/index_on_kaggle.py``).
"""
from __future__ import annotations

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import load_config
from .pinned import describe, fingerprint
from .store import get_pgvector_store

logger = logging.getLogger(__name__)

_state: dict = {}

#: Where the bearer token for /internal/* is kept. Generated on first boot and
#: shared with the Airflow container through a named volume, so nothing has to be
#: configured by hand and no default token is ever published.
TOKEN_PATH = os.getenv("AEROSEARCH_TOKEN_PATH", "/secrets/internal_token")


def internal_token() -> str:
    """The shared secret for /internal/*, created once and reused.

    These endpoints are the only ones reachable through the public tunnel, and
    they serve private chat media, so the token is generated rather than defaulted
    -- a published default would be an open door.
    """
    if "token" in _state:
        return _state["token"]
    env = os.getenv("AEROSEARCH_INTERNAL_TOKEN")
    if env:
        _state["token"] = env.strip()
        return _state["token"]
    try:
        with open(TOKEN_PATH) as fh:
            token = fh.read().strip()
        if token:
            _state["token"] = token
            return token
    except FileNotFoundError:
        pass
    import secrets

    token = secrets.token_urlsafe(32)
    try:
        os.makedirs(os.path.dirname(TOKEN_PATH), exist_ok=True)
        with open(TOKEN_PATH, "w") as fh:
            fh.write(token)
        os.chmod(TOKEN_PATH, 0o600)
        logger.info("generated a new internal token at %s", TOKEN_PATH)
    except OSError:
        logger.warning("could not persist the internal token; it will not survive a restart")
    _state["token"] = token
    return token


def require_token(authorization: str = Header(default="")) -> None:
    """Bearer auth for the tunnel-facing endpoints."""
    import hmac

    expected = f"Bearer {internal_token()}"
    if not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="bad or missing bearer token")


def _embedder():
    """Query-side embedder, loaded once on first use.

    text_only=True keeps the local footprint to the text tower; this process never
    embeds media, so the vision and audio encoders would just be resident memory.
    """
    if "embedder" not in _state:
        from .understanding.embed_gemma2 import EmbeddingGemma2Embedder

        cfg = _state["cfg"]
        logger.info("loading query embedder (text-only) %s", cfg.model.embed_model)
        _state["embedder"] = EmbeddingGemma2Embedder(cfg.model.embed_model, text_only=True)
    return _state["embedder"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    _state["cfg"] = cfg
    _state["store"] = get_pgvector_store(
        cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim
    )
    await _state["store"].init()
    logger.info("store ready, contract %s", fingerprint())
    yield
    await _state["store"].close()


app = FastAPI(title="aerosearch retrieval", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"ok": True, "fingerprint": fingerprint()}


@app.get("/pipeline")
async def pipeline():
    """The pinned embedding contract, so the bot and the DAG can assert they agree."""
    return describe()


@app.get("/stats")
async def stats(chat_id: Optional[int] = None):
    counts = await _state["store"].counts(chat_id=chat_id)
    return counts


@app.post("/search")
async def search(query: str = Form(...), k: int = Form(5), chat_id: Optional[int] = Form(None)):
    if not query.strip():
        raise HTTPException(status_code=400, detail="empty query")
    import asyncio

    qv = await asyncio.to_thread(_embedder().embed_query, query)
    hits = await _state["store"].search(qv, k=k, chat_id=chat_id)
    return JSONResponse(
        {
            "query": query,
            "hits": [
                {
                    "source": h.source,
                    "media_type": h.media_type,
                    "score": round(float(h.score), 4),
                    "document": h.document,
                    "s3_url": h.s3_url,
                    "user_name": h.user_name,
                }
                for h in hits
            ],
        }
    )


@app.post("/upload")
async def upload(file: UploadFile = File(...), chat_id: int = Form(0)):
    """Ingest a Telegram Desktop export: a ZIP, or a bare result.json.

    Media is pushed to object storage and one pending row is written per message,
    exactly as the bot does for live messages, so the same Airflow run picks them
    up. Nothing is embedded here.
    """
    from .telegram_export import ingest_export

    suffix = os.path.splitext(file.filename or "")[1].lower() or ".zip"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        while chunk := await file.read(1 << 20):
            tmp.write(chunk)
        tmp_path = tmp.name
    try:
        result = await ingest_export(tmp_path, _state["store"], chat_id=chat_id or None)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        os.unlink(tmp_path)
    return result


# --------------------------------------------------------------------------
# /internal/* -- the only surface exposed through the tunnel.
#
# The GPU notebook pulls work and pushes vectors through these three endpoints.
# Deliberately NOT a Postgres connection: this keeps the database off the public
# internet, keeps its password out of a notebook, and needs one HTTP tunnel
# instead of two TCP ones.
# --------------------------------------------------------------------------


@app.get("/internal/pending", dependencies=[Depends(require_token)])
async def internal_pending(limit: int = 200):
    """Rows that still need a vector. Media bytes are fetched separately."""
    rows = await _state["store"].fetch_unindexed(limit)
    return {
        "fingerprint": fingerprint(),
        "contract": describe(),
        "rows": [
            {"media_id": r["media_id"], "media_type": r["media_type"],
             "source": r.get("source"), "document": r.get("document")}
            for r in rows
        ],
    }


@app.get("/internal/media/{media_id}", dependencies=[Depends(require_token)])
async def internal_media(media_id: int):
    """Stream one row's media straight from object storage."""
    import asyncio

    rows = await _state["store"].fetch_unindexed(10_000)
    row = next((r for r in rows if r["media_id"] == media_id), None)
    if row is None or not row.get("s3_url"):
        raise HTTPException(status_code=404, detail=f"no pending media for id {media_id}")

    from .store.minio_source import MinioDownloader

    downloader = _state.setdefault("downloader", MinioDownloader())
    try:
        data = await asyncio.to_thread(downloader.download, row["s3_url"])
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"storage read failed: {exc}") from exc
    ext = os.path.splitext(row["s3_url"])[1].lower()
    media = "image/jpeg" if row["media_type"] == "photo" else "video/mp4"
    return Response(content=data, media_type=media,
                    headers={"X-Aerosearch-Ext": ext or ""})


@app.post("/internal/embeddings", dependencies=[Depends(require_token)])
async def internal_embeddings(payload: dict):
    """Write vectors back.

    Body: ``{"fingerprint": "...", "items": [{"media_id": 1, "embedding": [...],
    "document": "..."}]}``. The fingerprint is checked because a mismatch means
    the GPU side used a different contract, and those vectors would land in a
    different space with nothing raising.
    """
    got = payload.get("fingerprint")
    if got and got != fingerprint():
        raise HTTPException(
            status_code=409,
            detail=f"contract mismatch: worker sent {got!r}, server pins {fingerprint()!r}. "
                   f"Refusing to mix vector spaces.",
        )
    items = payload.get("items") or []
    dim = _state["cfg"].model.embed_dim
    written, rejected = 0, []
    for item in items:
        vec = item.get("embedding")
        mid = item.get("media_id")
        if not isinstance(vec, list) or len(vec) != dim:
            rejected.append({"media_id": mid, "why": f"expected {dim} floats, got {len(vec) if isinstance(vec, list) else type(vec).__name__}"})
            continue
        from .types import MediaDocument

        doc = MediaDocument(
            source=item.get("source") or "",
            media_type=item.get("media_type") or "video",
            document=item.get("document") or "",
            embedding=vec,
        )
        await _state["store"].update_indexed(mid, doc)
        written += 1
    logger.info("wrote %s vector(s) from the GPU worker, rejected %s", written, len(rejected))
    return {"written": written, "rejected": rejected}


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>aerosearch upload</title>
<style>
  :root { --bg:#faf9f7; --fg:#1a1a1a; --muted:#6b6b6b; --line:#e3e0db; --accent:#2f6f4f; }
  @media (prefers-color-scheme: dark) { :root:not([data-theme=light]) {
    --bg:#15171a; --fg:#ececec; --muted:#9a9a9a; --line:#2c2f34; --accent:#7fc4a0; } }
  * { box-sizing:border-box }
  body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.55 ui-sans-serif,system-ui,-apple-system,sans-serif; }
  .wrap { max-width:46rem; margin:0 auto; padding:3rem 16px 4rem }
  h1 { font-size:1.6rem; margin:0 0 .3rem }
  p.sub { color:var(--muted); margin:0 0 2rem }
  .card { border:1px solid var(--line); border-radius:12px; padding:1.4rem; margin-bottom:1.2rem; }
  .counts { display:flex; gap:2rem; flex-wrap:wrap }
  .n { font-size:2rem; font-weight:650; font-variant-numeric:tabular-nums }
  .lbl { color:var(--muted); font-size:.85rem; text-transform:uppercase; letter-spacing:.04em }
  input[type=file] { width:100%; padding:.8rem; border:1px dashed var(--line); border-radius:8px; background:transparent; color:var(--fg) }
  button { margin-top:1rem; background:var(--accent); color:#fff; border:0; border-radius:8px;
           padding:.7rem 1.3rem; font-size:1rem; cursor:pointer }
  button[disabled] { opacity:.55; cursor:default }
  code { background:color-mix(in srgb, var(--fg) 8%, transparent); padding:.1rem .35rem; border-radius:4px; font-size:.9em }
  #out { white-space:pre-wrap; margin-top:1rem; color:var(--muted); font-size:.92rem }
  a { color:var(--accent) }
</style></head><body><div class="wrap">
<h1>aerosearch</h1>
<p class="sub">Upload a Telegram chat export to make it searchable.</p>

<div class="card">
  <div class="counts">
    <div><div class="n" id="indexed">–</div><div class="lbl">searchable</div></div>
    <div><div class="n" id="pending">–</div><div class="lbl">waiting to index</div></div>
  </div>
  <p class="sub" style="margin:1rem 0 0">Indexing runs on the GPU pool, scheduled by Airflow.
  Trigger it now from the <a href="http://localhost:8080">Airflow UI</a>.</p>
</div>

<div class="card">
  <form id="f">
    <label class="lbl" for="file">Telegram Desktop export</label>
    <input id="file" name="file" type="file" accept=".zip,.json" required>
    <p class="sub" style="margin:.7rem 0 0">In Telegram Desktop: <b>⋮ → Export chat history</b>,
    format <b>JSON</b>. Upload the resulting ZIP, or just <code>result.json</code> for
    text-only messages.</p>
    <button id="go" type="submit">Upload</button>
  </form>
  <div id="out"></div>
</div>
</div><script>
async function refresh() {
  try {
    const r = await fetch('/stats'); const d = await r.json();
    document.getElementById('indexed').textContent = d.indexed ?? 0;
    document.getElementById('pending').textContent = d.pending ?? 0;
  } catch (e) { /* server starting */ }
}
refresh(); setInterval(refresh, 5000);

document.getElementById('f').addEventListener('submit', async (e) => {
  e.preventDefault();
  const f = document.getElementById('file').files[0];
  if (!f) return;
  const btn = document.getElementById('go'), out = document.getElementById('out');
  btn.disabled = true; out.textContent = `Uploading ${f.name} (${(f.size/1048576).toFixed(1)} MB)…`;
  try {
    const body = new FormData(); body.append('file', f);
    const r = await fetch('/upload', { method:'POST', body });
    const d = await r.json();
    out.textContent = r.ok
      ? `Queued ${d.queued} item(s) from ${d.messages} message(s)` +
        (d.skipped ? `, skipped ${d.skipped}` : '') + '. Trigger indexing in Airflow.'
      : `Failed: ${d.detail || r.status}`;
  } catch (err) { out.textContent = 'Failed: ' + err; }
  btn.disabled = false; refresh();
});
</script></body></html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return PAGE
