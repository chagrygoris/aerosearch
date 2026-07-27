"""Airflow DAG: periodically embed the chat's pending media + messages.

This is the CPU worker in the architecture — no always-on GPU. On each run it
pulls rows the bot inserted with a NULL embedding, understands them (Gemma 3n
E2B on CPU for visuals + faster-whisper for speech), embeds with e5, and writes
the vectors back so they become searchable.

Requires `aerosearch` installed in the Airflow image (see airflow/Dockerfile) and
the usual MINIO_* / DATABASE_URL / AEROSEARCH_* env vars.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_batch_index():
    import asyncio
    import logging
    import os

    from aerosearch.batch import BatchIndexer
    from aerosearch.config import load_config
    from aerosearch.factory import build_captioner, build_embedder, build_transcriber
    from aerosearch.store import get_pgvector_store
    from aerosearch.store.minio_source import MinioDownloader

    logging.basicConfig(level=logging.INFO)

    # CPU-friendly defaults for the scheduled worker.
    os.environ.setdefault("AEROSEARCH_CAPTION_MODEL", "google/gemma-3n-E2B-it")

    cfg = load_config()
    store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim)
    batch = BatchIndexer(
        cfg,
        captioner=build_captioner(cfg, "real"),
        transcriber=build_transcriber(cfg, "real"),
        embedder=build_embedder(cfg, "real"),
        store=store,
        downloader=MinioDownloader(),
    )

    async def _run():
        n = await batch.run(limit=int(os.getenv("AEROSEARCH_BATCH_LIMIT", "50")))
        await store.close()
        return n

    indexed = asyncio.run(_run())
    logging.info("batch indexed %s row(s)", indexed)


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "index_pending_media",
    default_args=default_args,
    description="Embed pending chat media + messages for semantic search",
    schedule="*/15 * * * *",  # every 15 minutes
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    PythonOperator(task_id="batch_index", python_callable=run_batch_index)
