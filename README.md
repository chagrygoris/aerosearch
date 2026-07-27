# aerosearch

Multimodal semantic search over Telegram chat media. The bot captures photos and
videos from a chat; an indexing pipeline *understands* each clip — what's on
screen **and** what's said — embeds it, and stores vectors in Postgres/pgvector
so you can search the chat's media by meaning.

## Why the rebuild

The original (summer 2025) prototype understood **audio only** (Whisper →
`multilingual-e5` → cosine). A silent Macarena dance or a Xi Jinping clip with no
transcribable speech was invisible. The rebuild adds a small **multimodal** model
— **Gemma 3 4B** (the smallest multimodal Gemma 3; 1B is text-only) — that looks
at sampled frames and describes the visuals + any on-screen text. It's sized to
run on a free Kaggle **T4** at 4-bit.

## Pipeline

```
media file ──▶ ffmpeg ──▶ frames ─────▶ Gemma (3n E2B / 3 4B) ─┐
                     └──▶ 16k wav ──▶ faster-whisper ───────────┼─▶ fuse ─▶ e5 ─▶ pgvector
                                                                ┘                    │
query ─────────────────────────────────────────────────────────── e5 embed ─▶ cosine ┘
```

## Deployment (no always-on GPU)

The bot is thin; understanding runs on a **CPU batch schedule**:

```
Telegram ─▶ bot ─▶ MinIO (media) + Postgres row w/ NULL embedding  (ingest, cheap)
                                    │
              Airflow every 15m ─▶ BatchIndexer: Gemma 3n E2B (CPU) + Whisper + e5
                                    │            ─▶ writes embedding back
Telegram ◀─ bot ◀─ /search ◀─ e5 query embed ◀─ pgvector cosine search  (bot only needs e5)
```

- **Ingest** (`bot/bot.py`): captures photos, videos **and text messages**, uploads
  media to MinIO, inserts a *pending* row via `store.add_pending`.
- **Batch worker** (`aerosearch/batch.py`, DAG `airflow/dags/index_pending.py`):
  picks up NULL-embedding rows, understands + embeds them (Gemma 3n E2B on CPU),
  writes vectors back. Runs every 15 min.
- **Search** (`/search <query>`): the only model the bot loads is e5 for the query.

Captioner is chosen by model name: `AEROSEARCH_CAPTION_MODEL=google/gemma-3n-E2B-it`
(CPU batch) or `google/gemma-3-4b-it` (T4 4-bit). Gemma 3n also hears audio, so
`Gemma3nCaptioner.caption_av` enables a single-model A/B vs Whisper+Gemma3.

- **Extraction** (`aerosearch/media`): ffmpeg samples N frames/clip + pulls mono
  16k audio. No moviepy.
- **Understanding** (`aerosearch/understanding`): pluggable `Captioner` /
  `Transcriber` / `Embedder` protocols with `fake` (offline), CPU, and CUDA/4-bit
  backends chosen by `factory.py`.
- **Fusion** (`aerosearch/fusion.py`): caption + transcript + filename hint → one
  embeddable document.
- **Store** (`aerosearch/store`): in-memory numpy store (dev/tests, persists to
  `.npz`+`.json`) and async pgvector store (prod, reuses the bot's Postgres).

## Quickstart (offline, no GPU, no models)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[dev]"

.venv/bin/aerosearch index src-videos --backend fake
.venv/bin/aerosearch search "macarena dance" --backend fake
```

The `fake` embedder is a real feature-hashing bag-of-words, so search actually
ranks sensibly — good for smoke tests and CI.

## Real models

```bash
uv pip install --python .venv -e ".[cpu]"     # e5 + faster-whisper (M2 dev)
.venv/bin/aerosearch index src-videos --backend real   # captions are faked-off on CPU-only

# Full vision path on a T4:
pip install -e ".[gpu]"                        # + transformers/bitsandbytes/timm
python scripts/kaggle_index.py --media src-videos --out index/store --whisper large-v3
```

Gemma 3 is gated on Hugging Face — set `HF_TOKEN` before downloading.

## Running locally

First, create your env file:

```bash
cp .env.example .env          # then edit: put your Telegram token in API_KEY
```

`config.py` is host/container aware: containers use `DATABASE_URL` (host
`postgres`) and `MINIO_ENDPOINT` (`minio:9000`); host/venv runs automatically
switch to `LOCAL_DATABASE_URL` and `localhost:9000`. The same `.env` works for both.

### Path A — lightweight (recommended to try first)

Run only the infra in Docker; run the bot from your venv. No Airflow, no gated
Gemma download — you index manually and watch data land.

```bash
# 0) Install the bot deps (aiogram + e5 + db/blob clients) into the venv
uv pip install --python .venv -e ".[bot]"

# 1) Start MinIO + Postgres only
docker compose up -d minio postgres

# 2) Run the bot on the host (uses LOCAL_DATABASE_URL + localhost MinIO)
cd bot && ../.venv/bin/python bot.py        # logs stream to this terminal
```

Now DM your bot (or add it to a group) and send a photo/video/text. Each lands in
MinIO + a *pending* Postgres row. Then index those rows yourself — no GPU needed
for a smoke test with the fake captioner, or real Whisper/e5 for actual quality:

```bash
# understand + embed the pending rows straight into Postgres
.venv/bin/python -c "
import asyncio, os
os.environ['AEROSEARCH_CAPTION_MODEL']='google/gemma-3-4b-it'  # captions skipped on CPU-only
from aerosearch.batch import BatchIndexer
from aerosearch.config import load_config
from aerosearch.factory import build_captioner, build_embedder, build_transcriber
from aerosearch.store import get_pgvector_store
from aerosearch.store.minio_source import MinioDownloader
cfg=load_config()
store=get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim)
b=BatchIndexer(cfg, build_captioner(cfg,'real'), build_transcriber(cfg,'real'),
               build_embedder(cfg,'real'), store, MinioDownloader(endpoint='localhost:9000'))
print('indexed', asyncio.run(b.run()))
"
```

Then search from Telegram with `/search <query>`.

### Path B — full stack (bot + Airflow + batch worker)

```bash
docker compose up -d --build      # minio, postgres, airflow (web+scheduler), bot
docker compose logs -f bot        # follow bot logs
```

⚠️ The Airflow worker downloads **Gemma 3n** (gated) on its first run — set
`HF_TOKEN` in `.env` and expect a large first-run download. The
`index_pending_media` DAG runs every 15 min; trigger it manually from the UI to
index immediately.

### Where to observe things

| What | Where | Notes |
|---|---|---|
| **Bot logs** | terminal (Path A) or `docker compose logs -f bot` | shows "queued photo/video/text" per ingest |
| **Media files** | MinIO console → http://localhost:9001 | log in with `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`, open the `media` bucket |
| **Saved messages/rows** | Postgres `media` table | `docker compose exec postgres psql -U postgres -d aerosearch -c "select media_id, media_type, user_name, left(document,60), (embedding is not null) as indexed from media order by media_id desc limit 20;"` |
| **Airflow UI + task logs** | http://localhost:8080 | log in `admin` / `AIRFLOW_ADMIN_PASSWORD`; DAG `index_pending_media`; click a run → task → Logs |
| **Airflow log files** | `airflow/logs/` on disk | mounted from the scheduler/worker |

Stop everything: `docker compose down` (add `-v` to also wipe MinIO/Postgres data).

> **Schema note:** the legacy `media` table was `vector(1536)`; e5-large is
> **1024**. On a fresh DB the store creates the right schema. If you hit a
> dimension error on an existing volume, drop the table (or `docker compose down
> -v`) and let it recreate.

## Config

Everything is env-driven (`aerosearch/config.py`): `AEROSEARCH_NUM_FRAMES`,
`AEROSEARCH_CAPTION_MODEL`, `AEROSEARCH_WHISPER_MODEL`, `AEROSEARCH_EMBED_MODEL`,
`DATABASE_URL` / `LOCAL_DATABASE_URL`, …

## Benchmark

`src-videos/` is the mini-benchmark corpus (Russian-language memes/clips) used by
the system tests — overkill for unit tests, but the real integration signal.

```bash
.venv/bin/python -m pytest -q          # unit + fake system tests (needs ffmpeg)
```
