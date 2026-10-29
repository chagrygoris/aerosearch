# aerosearch

Multimodal semantic search over Telegram chat media. The bot captures photos and
videos from a chat; an indexing pipeline *understands* each clip — what's on
screen **and** what's said — embeds it, and stores vectors in Postgres/pgvector
so you can search the chat's media by meaning.

## Why the rebuild

The original (summer 2025) prototype understood **audio only** (Whisper →
`multilingual-e5` → cosine). A silent Macarena dance or a Xi Jinping clip with no
transcribable speech was invisible, because nothing in the pipeline ever looked at
the pixels.

The fix turned out not to be more models but one better one. **EmbeddingGemma 2**
puts text, images, video and audio in a single 768-dim space, so a text query
reaches the pixels and the speech directly and the whole understand-then-embed
stage disappears. Measured end to end on CPU with no captioner and no ASR: 4/4 on a
mixed set — a spoken clip found from "speech about the government", an image found
from the Russian text rendered inside it.

## Pipeline

```
media file (photo / video+audio) ──▶ EmbeddingGemma 2 ──▶ 768-d vector ──▶ pgvector
                                             ▲                                  │
query text ──────────────────────────────────┘────────────────────── cosine ─────┘
```

One model, every modality. A media file is handed to the embedder as it is — no
frame sampling, no captioning, no ASR, no ffmpeg in the retrieval path. A captioner
(Gemma) and a transcriber (Whisper) remain available behind `--enrich` for when you
want the words as *readable text*, but they are an experiment to justify by
measurement rather than part of the default path.

## Deployment (no always-on GPU)

The bot is thin; understanding runs on a **CPU batch schedule**:

```
Telegram ─▶ bot ─▶ MinIO (media) + Postgres row w/ NULL embedding  (ingest, cheap)
                                    │
              Airflow every 15m ─▶ BatchIndexer: Whisper + EmbeddingGemma 2 (CPU)
                                    │            ─▶ writes embedding back
Telegram ◀─ bot ◀─ /search ◀─ eg2 query embed ◀─ pgvector cosine search  (query-side only)
```

- **Ingest** (`bot/bot.py`): captures photos, videos **and text messages**, uploads
  media to MinIO, inserts a *pending* row via `store.add_pending`.
- **Batch worker** (`aerosearch/batch.py`, DAG `airflow/dags/index_pending.py`):
  picks up NULL-embedding rows, understands + embeds them (Gemma 3n E2B on CPU),
  writes vectors back. Runs every 15 min.
- **Search** (`/search <query>`): the only model the bot loads is EmbeddingGemma 2,
  to embed the query.

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
uv pip install --python .venv -e ".[cpu]"     # EmbeddingGemma 2 + faster-whisper
.venv/bin/aerosearch index src-videos --backend real   # captions are faked-off on CPU-only

# Full vision path on a T4:
pip install -e ".[gpu]"                        # + transformers/bitsandbytes/timm
python scripts/kaggle_index.py --media src-videos --out index/store --whisper large-v3
```

Gemma 3 is gated on Hugging Face — set `HF_TOKEN` before downloading.

## Running it

One command, on any machine with Docker. The only thing nobody can default for
you is your Telegram bot token from [@BotFather](https://t.me/botfather):

```bash
echo 'API_KEY=123456:your-token-from-BotFather' > .env
docker compose up -d
```

That starts four services — `postgres`, `minio` (object storage), `worker` and
`bot` — and nothing else needs configuring: every other value has a working
default in `docker-compose.yml`, so **no `.env` beyond the token is required**.
First start downloads EmbeddingGemma 2 (~1.5 GB) into a named volume, so it
happens once rather than per restart.

```bash
docker compose logs -f bot worker     # follow both
docker compose down                   # stop; add -v to wipe data too
```

Then DM the bot a photo, video or message and `/search` for it. `/stats` shows
how much is indexed versus still queued — ingest and indexing are separate
processes, so a just-sent clip becomes searchable on the worker's next poll
(30 s by default, `AEROSEARCH_POLL_SECONDS`).

> **In a group chat, Telegram delivers nothing to the bot by default.** Privacy
> mode means it only sees slash commands and replies to itself. Either DM it, or
> run `/setprivacy` → *Disable* in @BotFather and re-add it to the group.

### What runs where

| Service | Image | Notes |
|---|---|---|
| `postgres` | `pgvector/pgvector:pg17` | vectors + rows; volume `pgdata-pg17` |
| `minio` | `zenko/cloudserver` | S3 API on `localhost:9000`, no web console |
| `worker` | built from `Dockerfile` | indexes pending rows on a poll loop |
| `bot` | same image | ingest, `/search`, `/stats` |

`worker` and `bot` share one image (`aerosearch-app:local`) because both load the
same embedder — building separately would download torch twice.

**Airflow is not started.** The `worker` poll loop does the same indexing without
a scheduler, a metadata DB or a Fernet key. If you want its UI anyway:

```bash
docker compose --profile airflow up -d     # needs AIRFLOW_FERNET_KEY set
```

### Two infrastructure notes

- **Postgres volumes carry their major version** (`pgdata-pg17`). A data directory
  cannot be read by a different major Postgres, so pinning the version into the
  volume name means a future bump creates a fresh volume instead of crashing with
  *"database files are incompatible with server"*. Nothing to remember.
- **The torch stack comes entirely from PyPI, in one `pip install`.** Splitting it
  across indexes (a `+cpu` torch from download.pytorch.org, torchvision from PyPI)
  produces wheels that import and then fail with *"operator torchvision::nms does
  not exist"*, which surfaces far away as transformers failing to import
  `PreTrainedModel`. The CPU index is not an option because it does not carry
  `torchcodec`. On arm64, including Apple Silicon, the PyPI wheels are CPU-only
  anyway; on amd64 this costs ~2 GB of unused CUDA libraries.

### Running the bot outside Docker

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[app]"
docker compose up -d postgres minio        # infra only
export API_KEY=123456:your-token
cd bot && ../.venv/bin/python bot.py       # terminal 2
.venv/bin/aerosearch worker                # terminal 3
```

`config.py` is host/container aware: containers use `DATABASE_URL` and reach S3 at
`minio:8000`, while host runs switch to `LOCAL_DATABASE_URL` and `localhost:9000`.

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
the embedding model, its revision, the prefixes, how images are fed in, and the fusion
layout together define the vector space every stored row lives in. Change one and new
vectors land in a *different* space from the old ones: nothing raises, but similarity
between an old row and a new query stops meaning anything, so it surfaces as mediocre
ranking rather than as a bug.

```bash
.venv/bin/aerosearch pipeline     # print the contract + its fingerprint
```

| Setting | Value |
|---|---|
| `EMBED_BACKEND` | `gemma2` |
| `EMBED_MODEL` | `google/embeddinggemma-2` |
| `EMBED_REVISION` | `f9972ac22e696a105c482202646979e33e4b5a1e` |
| `EMBED_DIM` | 768 |
| query prefix | `task: search result \| query: ` |
| document prefix | `title: none \| text: ` |
| `MEDIA_MODE` | `av-direct-v1` |
| `FUSION_VERSION` | `fuse-v1` |
| fingerprint | `5fe9500239c2` |

**All modalities share the text space**, which is the point: a photo is retrievable
without being captioned and a spoken clip without being transcribed. Media carries
*no* prefix and is passed by its own key. The full 740M model is loaded — text,
vision *and* audio — because speech is embedded directly rather than transcribed.

One API detail that is load-bearing: `{"video": path}` alone does **not** pull in the
file's audio track. On a clip of synthesized speech over a test pattern, "a cat
sleeping" scored 0.626 video-only versus **0.729** with `audio` passed as well, so
`embed_file` passes both keys and falls back to video-only when a file has no
decodable audio. Audio wants mono 16 kHz; `librosa` is required for wav and
`torchcodec` for audio inside an mp4.

Budget is 8192 tokens shared across all inputs — 140 per video frame at 1 fps (~58
frames), 25 per audio second (~327 s). Longer inputs were accepted without error in
testing (400 s of audio embedded fine), so assume silent truncation rather than a
raised limit.

Revisions are commit SHAs, not branches. `embeddinggemma-2`'s own `lastModified` was
the same day it was pinned here, so it is actively moving; an unpinned re-download
would shift the space under an existing index silently. The embedder also refuses to
run if the loaded model's width disagrees with `EMBED_DIM`, and `PgVectorStore.init()`
refuses if an existing table's column width disagrees — a contract change is a
re-embed, not a migration.

### Rolling back to e5

`multilingual-e5-large` is kept as a second contract because it is *known* strong on
Russian, which EmbeddingGemma 2 is not yet independently proven to be:

```bash
AEROSEARCH_EMBED_BACKEND=e5      # 1024-dim, text only
```

That is a different space (and text-only, so photos go back to needing captions), so
a rollback means re-embedding everything. The fingerprint changing is how you tell.

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
