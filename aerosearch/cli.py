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
    indexer = Indexer(
        cfg,
        captioner=build_captioner(cfg, backend),
        transcriber=build_transcriber(cfg, backend),
        embedder=build_embedder(cfg, backend),
        store=_make_store(store, store_path, cfg.model.embed_dim),
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


if __name__ == "__main__":
    app()
