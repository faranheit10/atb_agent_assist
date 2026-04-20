"""
ATB Agent Assist — KB Ingestion CLI
=====================================
Run this script once (or whenever KB files are updated) to:

  1. Use Gemini to structure all KB .txt files into atomic knowledge units
  2. Generate dense embeddings for atomic_text  (RETRIEVAL_DOCUMENT)
  3. Generate dense embeddings for summaries    (SEMANTIC_SIMILARITY)
  4. Store everything in ChromaDB + build BM25 index

Usage:
    python ingest.py                        # ingest all files in data/kb/
    python ingest.py --kb-dir /path/to/kb  # custom KB directory
    python ingest.py --force               # re-ingest even if store exists

Typical runtime: 2-5 minutes for 6 KB files (Gemini API calls + embeddings)
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import GEMINI_API_KEY, KB_DIR, EMBEDDING_DIM, EMBEDDING_MODEL, GENERATION_MODEL
from pipeline.ingestion import ingest_all_kb_files
from pipeline.embedder import embed_documents, embed_summaries
from pipeline.vector_store import KnowledgeStore

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    kb_dir: Path = typer.Option(KB_DIR, help="Directory containing .txt KB files"),
    force: bool = typer.Option(False, "--force", help="Re-ingest even if store is populated"),
) -> None:
    """Ingest ATB KB files into the vector store."""

    console.print(
        Panel.fit(
            "[bold cyan]ATB Agent Assist — KB Ingestion[/bold cyan]\n"
            f"[dim]Generation model : {GENERATION_MODEL}[/dim]\n"
            f"[dim]Embedding model  : {EMBEDDING_MODEL} ({EMBEDDING_DIM}-dim)[/dim]\n"
            f"[dim]KB directory     : {kb_dir}[/dim]",
            border_style="cyan",
        )
    )

    if not GEMINI_API_KEY:
        console.print("[red]✗ GEMINI_API_KEY is not set. Check your .env file.[/red]")
        raise typer.Exit(code=1)

    # Check if store already populated
    store = KnowledgeStore()
    if not store.is_empty() and not force:
        console.print(
            f"\n[yellow]⚠ Vector store already contains {store.count()} chunks.[/yellow]\n"
            "[dim]Use --force to re-ingest from scratch.[/dim]\n"
        )
        raise typer.Exit(code=0)

    # ── Step 1: LLM structuring ──────────────────────────────────────────
    console.rule("[bold]Step 1/3 — LLM Structuring[/bold]")
    units = ingest_all_kb_files(kb_dir)

    if not units:
        console.print("[red]✗ No knowledge units produced. Check KB files.[/red]")
        raise typer.Exit(code=1)

    # ── Step 2: Embedding generation ────────────────────────────────────
    console.rule("[bold]Step 2/3 — Generating Embeddings[/bold]")

    atomic_texts = [u.atomic_text for u in units]
    summaries = [u.summary for u in units]

    console.print(f"  Embedding {len(atomic_texts)} chunk texts …")
    chunk_embeddings = embed_documents(atomic_texts)

    console.print(f"  Embedding {len(summaries)} summaries …")
    summary_embeddings = embed_summaries(summaries)

    console.print(f"  [green]✓[/green] Generated {len(chunk_embeddings)} chunk embeddings")
    console.print(f"  [green]✓[/green] Generated {len(summary_embeddings)} summary embeddings")

    # ── Step 3: Store ────────────────────────────────────────────────────
    console.rule("[bold]Step 3/3 — Storing in Vector DB + BM25 Index[/bold]")
    store.upsert_units(units, chunk_embeddings, summary_embeddings)

    # ── Summary table ────────────────────────────────────────────────────
    console.print()
    table = Table(title="Ingestion Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", justify="right")

    from collections import Counter
    cat_counts = Counter(u.product_category for u in units)

    table.add_row("Total knowledge units", str(len(units)))
    table.add_row("Embedding dimensions", str(EMBEDDING_DIM))
    table.add_row("", "")
    for cat, count in sorted(cat_counts.items()):
        table.add_row(f"  {cat}", str(count))

    console.print(table)
    console.print(
        "\n[bold green]✓ Ingestion complete.[/bold green] "
        "Run [cyan]python api.py[/cyan] to start the API server.\n"
    )


if __name__ == "__main__":
    app()
