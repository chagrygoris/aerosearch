"""Scheduled indexing: local Airflow drives, a Kaggle T4 does the embedding.

Nothing heavy runs on this machine. The DAG counts what is pending, pushes a
notebook to whichever pool account has spare GPU quota, and that notebook reaches
*back* through a tunnel to this machine's Postgres and object storage, embeds the
media on the T4, and writes the vectors into the same rows. The DAG then confirms
the count moved.

Media never leaves your machine's storage for Kaggle's: the notebook streams it
over the tunnel and keeps it in the kernel's ephemeral disk.

    tunnel_reachable -> count_pending -> launch_on_pool -> wait -> verify

`tunnel_reachable` fails first and loudly, because every later task would
otherwise fail slowly and confusingly. The run is skipped when nothing is
pending, so a 15-minute schedule costs nothing on an idle chat.

## What you have to provide

* **A tunnel** exposing this machine's Postgres (5432) and S3 (9000) to the
  internet, and `AEROSEARCH_TUNNEL_PG` / `AEROSEARCH_TUNNEL_S3` set to the
  public `host:port` for each (e.g. `0.tcp.ngrok.io:17284`). With Tailscale,
  the tailnet IP and the normal ports.
* **Pool tokens** at `KAGGLE_TOKENS_DIR` (default `/opt/pool_tokens`, mounted
  into the Airflow containers read-only by docker-compose).

## A note on credentials

The notebook is pushed **private**, but it carries the Postgres password and S3
keys in its source, because Kaggle has no secrets API to provision (see
`gpu_pool/README.md`). For a personal project that is a reasonable trade; if it
ever stops being one, give the tunnel a read/write-restricted Postgres role and
rotate the tunnel URL between runs rather than treating the notebook as secret.
"""
from __future__ import annotations

import logging
import os
import socket
import tempfile
from datetime import datetime, timedelta

from airflow import DAG
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

POOL_DIR = os.getenv("KAGGLE_TOKENS_DIR", "/opt/pool_tokens")
ACCELERATOR = os.getenv("AEROSEARCH_KAGGLE_ACCELERATOR", "t4")
#: Which pool account to prefer. The launcher falls back to another account when
#: this one is at its 2-session concurrency cap, so this is a preference.
ACCOUNT = os.getenv("AEROSEARCH_KAGGLE_ACCOUNT") or None
KERNEL_TIMEOUT = int(os.getenv("AEROSEARCH_KAGGLE_TIMEOUT", "3600"))
BATCH_LIMIT = int(os.getenv("AEROSEARCH_BATCH_LIMIT", "200"))


def _tunnel_endpoints() -> tuple[str, str]:
    pg = os.getenv("AEROSEARCH_TUNNEL_PG", "")
    s3 = os.getenv("AEROSEARCH_TUNNEL_S3", "")
    if not pg or not s3:
        raise AirflowFailException(
            "AEROSEARCH_TUNNEL_PG and AEROSEARCH_TUNNEL_S3 must point at this "
            "machine's Postgres and S3 as reachable from the internet, e.g.\n"
            "  ngrok tcp 5432   -> AEROSEARCH_TUNNEL_PG=0.tcp.ngrok.io:17284\n"
            "  ngrok tcp 9000   -> AEROSEARCH_TUNNEL_S3=0.tcp.ngrok.io:19112\n"
            "Set them in .env; the Kaggle notebook cannot reach localhost."
        )
    return pg, s3


def tunnel_reachable(**_):
    """Fail fast if the tunnel is down, instead of burning a GPU session on it."""
    for label, hostport in zip(("postgres", "s3"), _tunnel_endpoints()):
        host, _, port = hostport.rpartition(":")
        try:
            with socket.create_connection((host, int(port)), timeout=10):
                logger.info("tunnel OK: %s at %s", label, hostport)
        except Exception as exc:
            raise AirflowFailException(
                f"tunnel for {label} ({hostport}) is not reachable from here: {exc}. "
                f"Start the tunnel, then trigger the DAG again."
            ) from exc


def count_pending(**context):
    """Skip the whole run when there is nothing to embed."""
    import asyncio

    from aerosearch.config import load_config
    from aerosearch.store import get_pgvector_store

    cfg = load_config()
    store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table,
                               dim=cfg.model.embed_dim)

    async def _count():
        await store.init()
        try:
            return (await store.counts())["pending"]
        finally:
            await store.close()

    pending = asyncio.run(_count())
    logger.info("%s row(s) pending", pending)
    if not pending:
        raise AirflowSkipException("nothing pending; not spending a GPU session")
    context["ti"].xcom_push(key="pending_before", value=pending)
    return pending


def _notebook_source(pg: str, s3: str) -> str:
    """The cell the T4 runs: pull pending rows over the tunnel, embed, write back."""
    pg_host, _, pg_port = pg.rpartition(":")
    return f'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers[image]>=6.1.0", "librosa", "torchcodec",
                "asyncpg", "minio", "pillow"], check=True)

import asyncio, os, tempfile
import asyncpg
from minio import Minio
from sentence_transformers import SentenceTransformer

DSN = "postgresql://{os.getenv('POSTGRES_USER', 'postgres')}:{os.getenv('POSTGRES_PASSWORD', 'postgres')}@{pg_host}:{pg_port}/{os.getenv('POSTGRES_DB', 'aerosearch')}"
S3_ENDPOINT = "{s3}"
BUCKET = "{os.getenv('MINIO_BUCKET', 'media')}"
LIMIT = {BATCH_LIMIT}

# Must match aerosearch/pinned.py exactly, or the vectors land in another space.
MODEL = "{os.getenv('AEROSEARCH_EMBED_MODEL', 'google/embeddinggemma-2')}"
REVISION = "{os.getenv('AEROSEARCH_EMBED_REVISION', 'f9972ac22e696a105c482202646979e33e4b5a1e')}"
DOC_PREFIX = "title: none | text: "

import torch
print("device:", "cuda" if torch.cuda.is_available() else "CPU (no GPU!)",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
model = SentenceTransformer(MODEL, revision=REVISION,
                            device="cuda" if torch.cuda.is_available() else "cpu")
if getattr(model, "default_prompt_name", None):
    model.default_prompt_name = None
print("dim:", model.get_embedding_dimension() if hasattr(model, "get_embedding_dimension")
      else model.get_sentence_embedding_dimension())

s3 = Minio(S3_ENDPOINT, access_key="{os.getenv('MINIO_ACCESS_KEY', 'minioadmin')}",
           secret_key="{os.getenv('MINIO_SECRET_KEY', 'minioadmin123')}", secure=False)

def embed_media(path, media_type):
    if media_type == "photo":
        payload = {{"image": path}}
    else:
        try:
            v = model.encode({{"video": path, "audio": path}}, normalize_embeddings=True)
            return v.tolist()
        except Exception as exc:
            print("  no decodable audio, video only:", exc)
            payload = {{"video": path}}
    return model.encode(payload, normalize_embeddings=True).tolist()

async def main():
    conn = await asyncpg.connect(DSN)
    rows = await conn.fetch(
        "SELECT media_id, media_type, s3_url, source, document FROM media "
        "WHERE embedding IS NULL ORDER BY media_id LIMIT $1", LIMIT)
    print(f"{{len(rows)}} pending row(s)")
    done = 0
    for r in rows:
        try:
            if r["media_type"] == "text":
                text = (r["document"] or "").strip()
                if not text:
                    continue
                vec = model.encode(DOC_PREFIX + text, normalize_embeddings=True).tolist()
                doc = text
            else:
                obj = (r["s3_url"] or "").split("/" + BUCKET + "/", 1)[-1]
                ext = os.path.splitext(obj)[1] or (".jpg" if r["media_type"] == "photo" else ".mp4")
                with tempfile.TemporaryDirectory() as wd:
                    local = os.path.join(wd, "media" + ext)
                    s3.fget_object(BUCKET, obj, local)
                    vec = embed_media(local, r["media_type"])
                doc = (r["source"] or "").rsplit(".", 1)[0].replace("_", " ")
            lit = "[" + ",".join(f"{{x:.7f}}" for x in vec) + "]"
            await conn.execute(
                "UPDATE media SET embedding = $1::vector, document = COALESCE(document, $2) "
                "WHERE media_id = $3", lit, doc, r["media_id"])
            done += 1
            print(f"  [{{done}}/{{len(rows)}}] {{r['media_id']}} {{r['media_type']}}")
        except Exception as exc:
            print(f"  FAILED {{r['media_id']}}: {{type(exc).__name__}}: {{exc}}")
    await conn.close()
    print("EMBEDDED", done)

asyncio.run(main())
'''


def launch_on_pool(**context):
    import json
    import sys

    sys.path.insert(0, "/opt/aerosearch")  # gpu_pool lives beside the package
    from gpu_pool import PoolLauncher, PoolRouter
    from gpu_pool.launcher import NotebookSpec

    pg, s3 = _tunnel_endpoints()
    router = PoolRouter.from_dir(POOL_DIR)
    router.probe_all()
    launcher = PoolLauncher(router)

    with tempfile.TemporaryDirectory(prefix="aerosearch_nb_") as wd:
        nb_path = os.path.join(wd, "index.ipynb")
        with open(nb_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": _notebook_source(pg, s3).splitlines(True)}],
                    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                                "name": "python3"},
                                 "language_info": {"name": "python"}},
                    "nbformat": 4, "nbformat_minor": 5,
                },
                fh,
            )
        spec = NotebookSpec(
            title=f"aerosearch-index-{context['ds_nodash']}-{context['ti'].try_number}",
            notebook_path=nb_path,
            accelerator=ACCELERATOR,
        )
        handle = launcher.launch(spec, account=ACCOUNT, min_hours=1)

    logger.info("launched %s on %s", handle.url, handle.account.name)
    context["ti"].xcom_push(key="kernel_ref", value=handle.ref)
    context["ti"].xcom_push(key="kernel_account", value=handle.account.name)
    context["ti"].xcom_push(key="kernel_url", value=handle.url)
    return handle.url


def wait_for_kernel(**context):
    import sys

    sys.path.insert(0, "/opt/aerosearch")
    from gpu_pool import PoolRouter

    ti = context["ti"]
    ref = ti.xcom_pull(task_ids="launch_on_pool", key="kernel_ref")
    account_name = ti.xcom_pull(task_ids="launch_on_pool", key="kernel_account")
    router = PoolRouter.from_dir(POOL_DIR)
    account = next(a for a in router.accounts if a.name == account_name)

    import time

    deadline = time.time() + KERNEL_TIMEOUT
    status = "unknown"
    while time.time() < deadline:
        status = router.kernel_status(account, ref)
        logger.info("kernel %s: %s", ref, status)
        if status in ("complete", "error", "cancelAcknowledged"):
            break
        time.sleep(20)

    if status != "complete":
        log = ""
        try:
            log = router.kernel_logs(account, ref)[-4000:]
        except Exception:
            pass
        raise AirflowFailException(
            f"kernel {ref} ended as {status!r} (expected 'complete').\n"
            f"Notebook: {ti.xcom_pull(task_ids='launch_on_pool', key='kernel_url')}\n{log}"
        )
    return status


def verify(**context):
    """Confirm vectors actually landed; a 'complete' kernel can still have failed."""
    import asyncio

    from aerosearch.config import load_config
    from aerosearch.store import get_pgvector_store

    cfg = load_config()
    store = get_pgvector_store(cfg.store.database_url, table=cfg.store.table,
                               dim=cfg.model.embed_dim)

    async def _count():
        await store.init()
        try:
            return (await store.counts())["pending"]
        finally:
            await store.close()

    before = context["ti"].xcom_pull(task_ids="count_pending", key="pending_before") or 0
    after = asyncio.run(_count())
    logger.info("pending %s -> %s (embedded %s)", before, after, before - after)
    if after >= before:
        raise AirflowFailException(
            f"the kernel finished but pending did not drop ({before} -> {after}). "
            f"Check the notebook output: "
            f"{context['ti'].xcom_pull(task_ids='launch_on_pool', key='kernel_url')}"
        )
    return {"before": before, "after": after, "embedded": before - after}


with DAG(
    "index_on_kaggle",
    description="Embed pending chat media on a Kaggle T4, over a tunnel to this machine",
    default_args={"owner": "aerosearch", "retries": 0,
                  "retry_delay": timedelta(minutes=5)},
    schedule=os.getenv("AEROSEARCH_SCHEDULE", "*/15 * * * *"),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,  # one GPU session at a time
    tags=["aerosearch", "gpu"],
) as dag:
    t_tunnel = PythonOperator(task_id="tunnel_reachable", python_callable=tunnel_reachable)
    t_count = PythonOperator(task_id="count_pending", python_callable=count_pending)
    t_launch = PythonOperator(task_id="launch_on_pool", python_callable=launch_on_pool)
    t_wait = PythonOperator(task_id="wait_for_kernel", python_callable=wait_for_kernel)
    t_verify = PythonOperator(task_id="verify", python_callable=verify)

    t_tunnel >> t_count >> t_launch >> t_wait >> t_verify
