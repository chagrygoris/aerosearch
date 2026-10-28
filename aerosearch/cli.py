"""Command line: `aerosearch index <path>` and `aerosearch search "<query>"`.

Defaults to the offline fake backend + a persisted in-memory store so the whole
loop runs on the M2 with no models and no database. Switch with --backend real
and --store pg once the T4 / Postgres path is in play.
"""
from __future__ import annotations

import asyncio
import logging

import typer
from rich.console import Console
from rich.table import Table

from .config import load_config
from .factory import build_captioner, build_embedder, build_transcriber
from .pipeline import Indexer, discover_media, search as run_search
from .store import MemoryVectorStore, get_pgvector_store

app = typer.Typer(add_completion=False, help="Multimodal semantic search over chat media.")
console = Console()

DEFAULT_STORE_PATH = "index/aerosearch_store"


def _make_store(store: str, store_path: str, dim: int):
    if store == "memory":
        return MemoryVectorStore(path=store_path)
    if store == "pg":
        cfg = load_config()
        return get_pgvector_store(cfg.store.database_url, table=cfg.store.table, dim=dim)
    raise typer.BadParameter(f"unknown store {store!r} (use memory|pg)")


@app.command()
def index(
    path: str = typer.Argument(..., help="File or directory of media to index"),
    backend: str = typer.Option("fake", help="fake | real"),
    store: str = typer.Option("memory", help="memory | pg"),
    store_path: str = typer.Option(DEFAULT_STORE_PATH, help="memory store base path"),
    enrich: bool = typer.Option(
        False, "--enrich/--no-enrich",
        help="Also run the captioner + Whisper (needs a GPU to be practical).",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Understand + embed + store every media file under PATH."""
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    cfg = load_config()

    paths = discover_media(path)
    if not paths:
        console.print(f"[yellow]No media found under {path}[/]")
        raise typer.Exit(1)

    console.print(f"Indexing [bold]{len(paths)}[/] file(s) with backend=[cyan]{backend}[/]…")
    # backend="fake" keeps the offline smoke path; otherwise the baseline is the
    # embedder alone, with caption/ASR only under --enrich.
    aux = backend if backend == "fake" else ("real" if enrich else "none")
    indexer = Indexer(
        cfg,
        captioner=build_captioner(cfg, aux),
        transcriber=build_transcriber(cfg, aux),
        embedder=build_embedder(cfg, backend),
        store=_make_store(store, store_path, cfg.model.embed_dim),
        enrich=enrich or backend == "fake",
    )

    async def _run():
        docs = await indexer.index_paths(paths)
        await indexer.store.close()
        return docs

    docs = asyncio.run(_run())
    console.print(f"[green]Indexed {len(docs)} document(s).[/]")
    for d in docs:
        console.print(f"  • [bold]{d.source}[/] — {d.document[:80].replace(chr(10), ' ')}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language search query"),
    k: int = typer.Option(5, "-k", help="Number of results"),
    backend: str = typer.Option("fake", help="fake | real"),
    store: str = typer.Option("memory", help="memory | pg"),
    store_path: str = typer.Option(DEFAULT_STORE_PATH, help="memory store base path"),
):
    """Search the index for QUERY."""
    cfg = load_config()
    embedder = build_embedder(cfg, backend)
    store_obj = _make_store(store, store_path, cfg.model.embed_dim)

    async def _run():
        hits = await run_search(query, embedder, store_obj, k=k)
        await store_obj.close()
        return hits

    hits = asyncio.run(_run())
    if not hits:
        console.print("[yellow]No results (is the index built?)[/]")
        return

    table = Table(title=f"Results for: {query!r}")
    table.add_column("#", justify="right")
    table.add_column("score", justify="right")
    table.add_column("source")
    table.add_column("document")
    for i, h in enumerate(hits, 1):
        table.add_row(str(i), f"{h.score:.3f}", h.source, h.document[:70].replace("\n", " "))
    console.print(table)


@app.command()
def pipeline():
    """Print the pinned embedding contract and its fingerprint.

    Two indexes agree only if their fingerprints match; see aerosearch/pinned.py.
    """
    from .pinned import describe

    table = Table(title="Pinned embedding contract")
    table.add_column("setting")
    table.add_column("value")
    for key, value in describe().items():
        table.add_row(key, value)
    console.print(table)

    if describe()["embed_revision"] == "(unpinned)":
        from .understanding.embed_e5 import E5Embedder

        resolved = E5Embedder().resolved_revision()
        if resolved:
            console.print(
                f"\n[yellow]Revision is unpinned.[/] Upstream currently serves "
                f"[bold]{resolved}[/].\nFreeze it with "
                f"[cyan]AEROSEARCH_EMBED_REVISION={resolved}[/] (or edit pinned.py), "
                f"then re-run this command — the fingerprint will change, so do it "
                f"before indexing anything you intend to keep."
            )
        else:
            console.print(
                "\n[yellow]Revision is unpinned[/] and could not be resolved "
                "(offline, or the model is not downloaded yet)."
            )


@app.command()
def worker(
    once: bool = typer.Option(False, "--once", help="Drain pending rows once and exit"),
    interval: int = typer.Option(60, help="Seconds between polls when looping"),
    limit: int = typer.Option(50, help="Max rows per pass"),
    enrich: bool = typer.Option(
        False, "--enrich/--no-enrich",
        help="Also run the captioner + Whisper. Off by default: the embedder covers "
             "images, video and audio on its own, and these need a GPU to be practical.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", "-v"),
):
    """Index rows the bot left pending, against Postgres + MinIO.

    The local stand-in for the Airflow DAG (`airflow/dags/index_pending.py`): same
    BatchIndexer, driven by a plain poll loop instead of a scheduler, so a local
    deployment needs only Postgres and MinIO running.

    By default EmbeddingGemma 2 is the only model: a media file goes straight to it,
    so photos and spoken clips are retrievable with no captioner and no ASR.
    """
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)

    from .batch import BatchIndexer
    from .pinned import fingerprint
    from .store.minio_source import MinioDownloader

    cfg = load_config()
    store_obj = get_pgvector_store(
        cfg.store.database_url, table=cfg.store.table, dim=cfg.model.embed_dim
    )
    batch = BatchIndexer(
        cfg,
        captioner=build_captioner(cfg, "real" if enrich else "none"),
        transcriber=build_transcriber(cfg, "real" if enrich else "none"),
        embedder=build_embedder(cfg, "real"),
        store=store_obj,
        downloader=MinioDownloader(),
        enrich=enrich,
    )

    console.print(
        f"worker: embed=[cyan]{cfg.model.embed_model}[/] "
        f"enrich=[cyan]{'on (caption + whisper)' if enrich else 'off (embedder only)'}[/] "
        f"fingerprint=[cyan]{fingerprint()}[/]"
    )

    async def _run():
        try:
            while True:
                n = await batch.run(limit=limit)
                if n:
                    console.print(f"[green]indexed {n} row(s)[/]")
                if once:
                    return n
                await asyncio.sleep(interval)
        finally:
            await store_obj.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        console.print("\nstopped")


if __name__ == "__main__":
    app()
