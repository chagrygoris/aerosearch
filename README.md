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
MinIO + a *pending* Postgres row. A third terminal drains those rows:

```bash
# 3) Understand + embed pending rows, on a 60s poll loop
.venv/bin/aerosearch worker                 # Ctrl-C to stop
.venv/bin/aerosearch worker --once          # single pass, then exit
```

`worker` is the local stand-in for the Airflow DAG — same `BatchIndexer`, driven by
a poll loop instead of a scheduler, so nothing but Postgres and MinIO needs to run.

**Captions are off by default** (`--no-captions`). Gemma-class captioning needs a
GPU to be practical — measured 9–16 s/clip on a T4 and several times that on CPU —
so a local run indexes on speech plus filename hint, which is immediate, and leaves
the vision pass to a GPU batch. `--captions` turns it on if you want to wait.

⚠️ **Captions off means silent clips stay nearly invisible** — the exact gap the
rebuild exists to close. Measured with the pinned embedder on three documents: the
queries `политик про Винни-Пуха` (0.829) and `милые животные` (0.809) retrieved the
right clip, but `танец без музыки` ranked a *silent* `macarena.mp4` last (0.738),
because with no caption and no speech its document is only `File: macarena.` and
there is nothing to match on. Run the GPU batch (or `--captions`) before trusting
visual queries.

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

## Accelerator pool

CPU captioning is too slow to be the plan (Gemma-class models decode at single-digit tokens/sec on
CPU). Instead, GPU work is routed to whichever **Kaggle** account in our pool has spare weekly
quota, using [`gpu_pool/`](gpu_pool/) — vendored from
[`optimist1938/GA-Research`](https://github.com/optimist1938/GA-Research/tree/main/gpu_pool).
Each account gets 30 GPU-hours/week; `router.py` reports who is free, `launcher.py` pushes a
notebook there.

```bash
uv pip install --python .venv -e ".[pool]"     # the official Kaggle CLI; needs Python 3.11+
export KAGGLE_TOKENS_DIR=~/pool_tokens

python gpu_pool/router.py --no-kernels                    # quota table, ~1s per account
python gpu_pool/router.py --best --idle-only --min-gpu-hours 6
```

**Credentials live outside the repo** — one directory per account holding a `token` file, pointed
at by `KAGGLE_TOKENS_DIR` (conventionally `~/pool_tokens`). Never commit them; `.gitignore` blocks
`*/token`, `kaggle*.json` and `*_tokens/` as a backstop. Full layout in
[`gpu_pool/README.md`](gpu_pool/README.md).

Launching, pinned to one account:

```bash
python gpu_pool/launcher.py --title aerosearch-gemma4-smoke \
    --notebook notebooks/kaggle_gpu_batch.ipynb \
    --accelerator t4 --min-hours 2 --account account8_PepeSchneyne --wait 3600
```

Outside a competition Kaggle offers only **T4** and **P100**; L4 and RTX Pro 6000 are granted by a
specific competition and the router tracks which accounts can reach them
(`python gpu_pool/router.py --competitions`). `--dry-run` prints the chosen account and the exact
push body without touching Kaggle.

### Two caveats for this repo

- **aerosearch is private**, so `--repo https://github.com/chagrygoris/aerosearch.git` will fail to
  clone on Kaggle. There is no secrets API, so a GitHub PAT has to be added **by hand** to each
  account under *Add-ons → Secrets* and passed with `--github-token-secret`. For self-contained
  experiments (e.g. benchmarking a captioner) prefer `--notebook`, which needs no clone, and ship
  the benchmark clips as a Kaggle Dataset via `--dataset` since `src-videos/` is gitignored.
- **`gpu_pool/rtx_monitor.py` does nothing here.** It replays offline W&B runs that print
  `WANDB_SYNC` lines, a convention from GA-Research's `pose3d` trainer. Kept only so the folder
  stays a faithful copy of upstream.

Runs cannot be cancelled once pushed — neither the CLI nor the API exposes a cancel — so check the
ETA before launching.

## Pinned embeddings

The embedding side is frozen in [`aerosearch/pinned.py`](aerosearch/pinned.py), because
the embedding model, its revision, the e5 prefixes and the fusion layout together
define the vector space every stored row lives in. Change one and new vectors land in
a *different* space from the old ones: nothing raises, but similarity between an old
row and a new query stops meaning anything, so it surfaces as mediocre ranking rather
than as a bug.

```bash
.venv/bin/aerosearch pipeline     # print the contract + its fingerprint
```

| Setting | Value |
|---|---|
| `EMBED_MODEL` | `intfloat/multilingual-e5-large` |
| `EMBED_REVISION` | `3d7cfbdacd47fdda877c5cd8a79fbcc4f2a574f3` |
| `EMBED_DIM` | 1024 |
| prefixes | `passage: ` / `query: ` |
| `FUSION_VERSION` | `fuse-v1` |
| fingerprint | `43adcb174828` |

The revision is pinned to a commit SHA, not a branch: this repo has been reuploaded
upstream before (that commit's own `lastModified` is 2026-04-02), and an unpinned
re-download would move the space under an existing index silently. `E5Embedder`
also refuses to run if the loaded model's width disagrees with `EMBED_DIM`, so a
model swap fails loudly instead of writing mixed-geometry vectors.

Two indexes are comparable only if their fingerprints match. **Rows embedded before
the pin went in may have used another revision**, so treat the current index as
fingerprint-unknown and re-embed if ranking looks off.

Captioner and transcriber are deliberately *not* pinned: they change the `document`
text a row carries, so a re-index yields a better document, but they do not change
the geometry, and old and new rows stay comparable.

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
