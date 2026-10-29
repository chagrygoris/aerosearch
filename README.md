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

## Architecture

Nothing heavy runs on your machine. The only local model is the **text-only**
query encoder (~270M of EmbeddingGemma 2's 740M) — a search box never embeds
pixels. Media embedding, which does need the vision and audio encoders, happens
on a Kaggle **T4**.

```
YOUR MACHINE (docker compose)                      KAGGLE (T4, gpu_pool)
─────────────────────────────                      ─────────────────────
  bot          Telegram ingest, /search, /stats
   │                      │
   ▼                      ▼
  server       query embedder (text-only) ─▶ pgvector cosine
   │           upload site for chat exports
   ▼
  postgres + minio  ◀───────── tunnel ──────────▶  embeds media on GPU,
   ▲                                               writes vectors back
   │
  airflow      schedules + manual trigger ──push notebook──▶
```

Media never lands on Kaggle's storage: the notebook streams it over your tunnel
and keeps it on the kernel's ephemeral disk.

| Service | Role | Port |
|---|---|---|
| `bot` | Telegram ingest, `/search`, `/stats`. Holds **no** model. | — |
| `server` | query embedder, `/search` API, upload site, `/stats` | 8000 |
| `postgres` | rows + vectors (pgvector) | 5432 |
| `minio` | media (CloudServer, S3 API) | 9000 |
| `airflow-webserver` | UI: schedule + **manual trigger** | 8080 |
| `airflow-scheduler` | runs `index_on_kaggle` every 15 min | — |

### The indexing run

[`airflow/dags/index_on_kaggle.py`](airflow/dags/index_on_kaggle.py):

```
tunnel_reachable → count_pending → launch_on_pool → wait_for_kernel → verify
```

- **`tunnel_reachable`** fails first and loudly if the tunnel is down, rather than
  spending a GPU session to discover it.
- **`count_pending`** *skips the run* when nothing is pending, so a 15-minute
  schedule costs no quota on an idle chat.
- **`launch_on_pool`** generates a notebook with your tunnel endpoints baked in and
  pushes it to whichever pool account has spare T4 quota (`gpu_pool/`).
- **`verify`** checks the pending count actually dropped — a kernel can finish
  "complete" having embedded nothing.

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

```bash
echo 'API_KEY=123456:your-token-from-BotFather' > .env
docker compose up -d
```

Six services, every other value defaulted. Then:

| What | Where |
|---|---|
| Upload a chat export, see counts | <http://localhost:8000> |
| Airflow UI — trigger indexing | <http://localhost:8080> |
| Telegram | DM the bot: `/search`, `/stats` |

### Bulk-importing chat history

In Telegram Desktop: **⋮ → Export chat history**, format **JSON**. Upload the
resulting ZIP at <http://localhost:8000>. Every message becomes a pending row —
text inline, media pushed to storage — in the same shape the live bot inserts, so
one indexing run picks up both. A bare `result.json` works too, text only.

### Indexing needs a tunnel

The Kaggle notebook has to reach *this* machine, and it cannot reach `localhost`:

```bash
ngrok tcp 5432        # -> AEROSEARCH_TUNNEL_PG=0.tcp.ngrok.io:17284
ngrok tcp 9000        # -> AEROSEARCH_TUNNEL_S3=0.tcp.ngrok.io:19112
```

Put both in `.env` and restart Airflow. With Tailscale, use the tailnet IP and the
normal ports. Pool credentials come from `KAGGLE_TOKENS_DIR` (default
`~/pool_tokens`), mounted read-only into the Airflow containers.

Then in the Airflow UI press ▶ on `index_on_kaggle`; the unindexed count in
Telegram (`/stats`) and on the upload page drops when it lands. The DAG arrives
**active**, not paused, so the button works immediately — and so the `*/15`
schedule starts firing. Without a tunnel each run fails its first task with the
commands above in the log; set `AEROSEARCH_SCHEDULE=` (empty) in `.env` for
manual-only until your tunnel is permanent.

The webserver takes **~50 s** to answer on first boot: it waits for
`airflow-init` to migrate the metadata tables and create the `admin` user
(password from `AIRFLOW_ADMIN_PASSWORD`, default `admin`).

> **One honest caveat on credentials.** The generated notebook is pushed *private*,
> but it carries the Postgres password and S3 keys in its source, because Kaggle has
> no secrets API to provision (see [`gpu_pool/README.md`](gpu_pool/README.md)). For a
> personal project that is a fair trade. If it stops being one: give the tunnel a
> restricted Postgres role, and rotate the tunnel URL between runs rather than
> treating the notebook as a secret.

### Running pieces outside Docker

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv -e ".[app]"
docker compose up -d postgres minio
.venv/bin/uvicorn aerosearch.server:app --port 8000     # retrieval + site
cd bot && API_KEY=... AEROSEARCH_SERVER=http://localhost:8000 ../.venv/bin/python bot.py
```

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
