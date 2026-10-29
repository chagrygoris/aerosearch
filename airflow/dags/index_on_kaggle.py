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


NGROK_API = os.getenv("NGROK_API", "http://ngrok:4040/api/tunnels")
SERVER_URL = os.getenv("AEROSEARCH_SERVER", "http://server:8000")
TOKEN_PATH = os.getenv("AEROSEARCH_TOKEN_PATH", "/secrets/internal_token")


def _internal_token() -> str:
    """The shared secret the retrieval server generated on first boot."""
    env = os.getenv("AEROSEARCH_INTERNAL_TOKEN")
    if env:
        return env.strip()
    try:
        with open(TOKEN_PATH) as fh:
            token = fh.read().strip()
    except FileNotFoundError as exc:
        raise AirflowFailException(
            f"no internal token at {TOKEN_PATH}. The retrieval server writes it on "
            f"first boot; is the `server` service up, and does it share the "
            f"`secrets` volume with Airflow?"
        ) from exc
    if not token:
        raise AirflowFailException(f"{TOKEN_PATH} is empty")
    return token


def _public_url() -> str:
    """Ask the local ngrok agent for its public URL.

    Discovered, never configured: ngrok's free URLs change on restart, and having
    to paste one into .env after every restart is exactly the friction this
    avoids. AEROSEARCH_PUBLIC_URL overrides it for a stable domain or a different
    tunnel entirely.
    """
    override = os.getenv("AEROSEARCH_PUBLIC_URL", "").strip()
    if override:
        return override.rstrip("/")

    import json
    import urllib.request

    try:
        with urllib.request.urlopen(NGROK_API, timeout=10) as resp:
            data = json.load(resp)
    except Exception as exc:
        raise AirflowFailException(
            f"could not reach the ngrok agent at {NGROK_API}: {exc}\n"
            f"Start it with `docker compose up -d ngrok` (needs NGROK_AUTHTOKEN in "
            f".env -- a free ngrok account), or set AEROSEARCH_PUBLIC_URL to your "
            f"own https tunnel pointing at the `server` service."
        ) from exc

    for tunnel in data.get("tunnels", []):
        url = tunnel.get("public_url", "")
        if url.startswith("https://"):
            return url.rstrip("/")
    raise AirflowFailException(
        f"the ngrok agent at {NGROK_API} has no https tunnel yet. "
        f"Check `docker compose logs ngrok` -- an invalid NGROK_AUTHTOKEN or the "
        f"3-endpoint free-plan cap are the usual causes."
    )


def tunnel_reachable(**context):
    """Resolve the public URL and prove the GPU side can actually use it.

    Fails here rather than after a GPU session has been spent discovering it.
    """
    import json
    import urllib.request

    url = _public_url()
    token = _internal_token()
    req = urllib.request.Request(
        f"{url}/internal/pending?limit=1",
        headers={"Authorization": f"Bearer {token}",
                 "ngrok-skip-browser-warning": "1"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.load(resp)
    except Exception as exc:
        raise AirflowFailException(
            f"{url}/internal/pending is not usable from here: {exc}\n"
            f"The tunnel must point at the `server` service (port 8000)."
        ) from exc

    remote_fp = body.get("fingerprint")
    from aerosearch.pinned import fingerprint as local_fp

    if remote_fp != local_fp():
        raise AirflowFailException(
            f"contract mismatch through the tunnel: {remote_fp!r} vs {local_fp()!r}"
        )
    logger.info("tunnel OK: %s (contract %s)", url, remote_fp)
    context["ti"].xcom_push(key="public_url", value=url)
    return url


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


def _notebook_source(public_url: str, token: str) -> str:
    """The cell the T4 runs.

    It never touches Postgres or S3: it pulls work, media and pushes vectors over
    one authenticated HTTP endpoint, so the database stays off the public internet
    and its password stays out of this notebook. The only secret here is a bearer
    token scoped to /internal/*.
    """
    return f'''
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "sentence-transformers[image]>=6.1.0", "librosa", "torchcodec",
                "requests", "pillow"], check=True)

import os, tempfile, requests, torch
from sentence_transformers import SentenceTransformer

BASE = "{public_url}"
H = {{"Authorization": "Bearer {token}", "ngrok-skip-browser-warning": "1"}}
LIMIT = {BATCH_LIMIT}

print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU!)")

work = requests.get(f"{{BASE}}/internal/pending", params={{"limit": LIMIT}},
                    headers=H, timeout=60).json()
rows = work["rows"]
c = work["contract"]
print(f"{{len(rows)}} pending row(s); contract {{work['fingerprint']}}")

# Load exactly what the server pins, or the vectors land in another space.
model = SentenceTransformer(c["embed_model"], revision=c["embed_revision"],
                            device="cuda" if torch.cuda.is_available() else "cpu")
if getattr(model, "default_prompt_name", None):
    model.default_prompt_name = None
DOC_PREFIX = c["passage_prefix"]

def embed_media(path, media_type):
    if media_type == "photo":
        return model.encode({{"image": path}}, normalize_embeddings=True).tolist()
    try:
        # Both keys: video alone leaves the speech out.
        return model.encode({{"video": path, "audio": path}},
                            normalize_embeddings=True).tolist()
    except Exception as exc:
        print("   no decodable audio, video only:", exc)
        return model.encode({{"video": path}}, normalize_embeddings=True).tolist()

items, failed = [], 0
for i, r in enumerate(rows, 1):
    mid, mtype = r["media_id"], r["media_type"]
    try:
        if mtype == "text":
            text = (r.get("document") or "").strip()
            if not text:
                continue
            vec = model.encode(DOC_PREFIX + text, normalize_embeddings=True).tolist()
            doc = text
        else:
            resp = requests.get(f"{{BASE}}/internal/media/{{mid}}", headers=H, timeout=300)
            resp.raise_for_status()
            ext = resp.headers.get("X-Aerosearch-Ext") or (
                ".jpg" if mtype == "photo" else ".mp4")
            with tempfile.TemporaryDirectory() as wd:
                local = os.path.join(wd, "media" + ext)
                with open(local, "wb") as fh:
                    fh.write(resp.content)
                vec = embed_media(local, mtype)
            doc = (r.get("source") or "").rsplit(".", 1)[0].replace("_", " ")
        items.append({{"media_id": mid, "embedding": vec, "document": doc,
                      "media_type": mtype, "source": r.get("source") or ""}})
        print(f"  [{{i}}/{{len(rows)}}] {{mid}} {{mtype}}")
    except Exception as exc:
        failed += 1
        print(f"  FAILED {{mid}}: {{type(exc).__name__}}: {{exc}}")

    # Push in batches so a late crash does not throw away finished work.
    if len(items) >= 25:
        out = requests.post(f"{{BASE}}/internal/embeddings", headers=H, timeout=300,
                            json={{"fingerprint": work["fingerprint"], "items": items}}).json()
        print("   pushed:", out)
        items = []

if items:
    out = requests.post(f"{{BASE}}/internal/embeddings", headers=H, timeout=300,
                        json={{"fingerprint": work["fingerprint"], "items": items}}).json()
    print("   pushed:", out)
print("DONE failed=", failed)
'''


def launch_on_pool(**context):
    import json
    import sys

    sys.path.insert(0, "/opt/aerosearch")  # gpu_pool lives beside the package
    from gpu_pool import PoolLauncher, PoolRouter
    from gpu_pool.launcher import NotebookSpec

    url = context["ti"].xcom_pull(task_ids="tunnel_reachable", key="public_url")
    token = _internal_token()
    router = PoolRouter.from_dir(POOL_DIR)
    router.probe_all()
    launcher = PoolLauncher(router)

    with tempfile.TemporaryDirectory(prefix="aerosearch_nb_") as wd:
        nb_path = os.path.join(wd, "index.ipynb")
        with open(nb_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "cells": [{"cell_type": "code", "metadata": {}, "execution_count": None,
                               "outputs": [], "source": _notebook_source(url, token).splitlines(True)}],
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
    # Empty AEROSEARCH_SCHEDULE means manual-only (trigger from the UI).
    schedule=os.getenv("AEROSEARCH_SCHEDULE", "*/15 * * * *") or None,
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
