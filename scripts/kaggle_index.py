"""Index a folder of media on a Kaggle/Colab T4 with the real Gemma 3 + Whisper.

Designed to be pasted into a Kaggle notebook cell (or run as a script). It builds
a portable MemoryVectorStore (.json + .npz) you can download and later load into
Postgres, or push straight to pgvector if DATABASE_URL points at a reachable DB.

Kaggle setup cell (run once):
    !pip install -q "git+https://github.com/…/aerosearch"  # or upload the repo
    !pip install -q transformers accelerate bitsandbytes timm \
        sentence-transformers faster-whisper
    # HF token needed to download gemma-3 (gated):
    import os; os.environ["HF_TOKEN"] = "hf_..."

Then:
    !python scripts/kaggle_index.py --media /kaggle/input/my-clips --out /kaggle/working/index
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from aerosearch.config import load_config
from aerosearch.factory import build_captioner, build_embedder, build_transcriber
from aerosearch.pipeline import Indexer, discover_media
from aerosearch.store import MemoryVectorStore, get_pgvector_store


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", required=True, help="Folder (or file) of media to index")
    ap.add_argument("--out", default="/kaggle/working/index/aerosearch_store",
                    help="Memory-store base path (writes .json + .npz)")
    ap.add_argument("--store", choices=["memory", "pg"], default="memory")
    ap.add_argument("--whisper", default=None, help="Override Whisper size (e.g. large-v3)")
    ap.add_argument("--frames", type=int, default=None, help="Frames sampled per video")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    if args.whisper:
        cfg.model.transcribe_model = args.whisper
    if args.frames:
        cfg.media.num_frames = args.frames

    if args.store == "pg":
        store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim)
    else:
        store = MemoryVectorStore(path=args.out)

    indexer = Indexer(
        cfg,
        captioner=build_captioner(cfg, "real"),
        transcriber=build_transcriber(cfg, "real"),
        embedder=build_embedder(cfg, "real"),
        store=store,
    )

    paths = discover_media(args.media)
    print(f"Indexing {len(paths)} file(s) → {args.store}")

    async def _run():
        docs = await indexer.index_paths(paths)
        await store.close()
        return docs

    docs = asyncio.run(_run())
    for d in docs:
        print(f"\n=== {d.source} ===")
        print(f"  caption:    {d.caption[:200]}")
        print(f"  transcript: {d.transcript[:200]}")
    print(f"\nDone: {len(docs)} documents indexed.")


if __name__ == "__main__":
    main()
