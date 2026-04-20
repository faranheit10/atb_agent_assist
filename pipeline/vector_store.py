"""
ATB Agent Assist — Vector Store & Sparse Index
===============================================
Two storage layers working together:

  1. ChromaDB (persistent)  —  dense vector search over two collections:
       • atb_kb_chunks    : embeddings of atomic_text  (fine-grained)
       • atb_kb_summaries : embeddings of summaries    (coarse / hierarchical)

  2. BM25 in-memory index  —  keyword / sparse search over atomic_text,
       serialised to JSON on disk so it survives process restarts.

The store is the single source of truth: ingest once, query many times.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from rich.console import Console

from config import (
    CHROMA_DIR,
    BM25_INDEX_PATH,
    COLLECTION_CHUNKS,
    COLLECTION_SUMMARIES,
    EMBEDDING_DIM,
    KnowledgeUnit,
)

console = Console()


# ---------------------------------------------------------------------------
# ChromaDB client (persistent, local)
# ---------------------------------------------------------------------------

def _get_chroma_client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_or_create_collection(
    client: chromadb.PersistentClient, name: str
) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


# ---------------------------------------------------------------------------
# BM25 sparse index (serialised as JSON)
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Lightweight BM25 index over KB chunks.
    Serialises tokenised corpus + metadata to JSON for persistence.
    """

    def __init__(self) -> None:
        self._corpus_tokens: list[list[str]] = []
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------
    def _tokenise(self, text: str) -> list[str]:
        """Lower-case, split on non-alpha, remove short tokens."""
        return [t for t in text.lower().split() if len(t) > 2]

    def add(self, unit_id: str, text: str) -> None:
        # If ID exists, remove it first to allow update
        if unit_id in self._ids:
            idx = self._ids.index(unit_id)
            self._ids.pop(idx)
            self._texts.pop(idx)
            self._corpus_tokens.pop(idx)
            
        self._ids.append(unit_id)
        self._texts.append(text)
        self._corpus_tokens.append(self._tokenise(text))
        self._bm25 = None  # invalidate

    def remove_by_ids(self, unit_ids: list[str]) -> None:
        """Remove multiple units from the index."""
        indices_to_remove = []
        for i, uid in enumerate(self._ids):
            if uid in unit_ids:
                indices_to_remove.append(i)
        
        # Remove in reverse order to keep indices valid
        for idx in sorted(indices_to_remove, reverse=True):
            self._ids.pop(idx)
            self._texts.pop(idx)
            self._corpus_tokens.pop(idx)
        
        self._bm25 = None # invalidate
        self.build()
        self.save()

    def build(self) -> None:
        """Build (or rebuild) the BM25 index after all documents are added."""
        if self._corpus_tokens:
            self._bm25 = BM25Okapi(self._corpus_tokens)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """Returns list of (unit_id, bm25_score) sorted desc."""
        if not self._bm25:
            self.build()
        if not self._bm25:
            return []
        tokens = self._tokenise(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._ids, scores), key=lambda x: x[1], reverse=True
        )
        return [(uid, float(score)) for uid, score in ranked[:top_k] if score > 0]

    # ------------------------------------------------------------------
    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        data = {
            "ids": self._ids,
            "texts": self._texts,
            "corpus_tokens": self._corpus_tokens,
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = BM25_INDEX_PATH) -> "BM25Index":
        idx = cls()
        if not path.exists():
            return idx
        data = json.loads(path.read_text(encoding="utf-8"))
        idx._ids = data["ids"]
        idx._texts = data["texts"]
        idx._corpus_tokens = data["corpus_tokens"]
        idx.build()
        return idx


# ---------------------------------------------------------------------------
# KnowledgeStore — unified write/read interface
# ---------------------------------------------------------------------------

class KnowledgeStore:
    """
    Unified interface over ChromaDB + BM25.

    Usage:
        store = KnowledgeStore()
        store.upsert_units(units, chunk_embeddings, summary_embeddings)
        # Later:
        dense_hits  = store.dense_search(query_embedding, top_k=20)
        sparse_hits = store.bm25_search(query_text, top_k=20)
        coarse_hits = store.coarse_search(query_embedding, top_k=3)
    """

    def __init__(self) -> None:
        self._chroma = _get_chroma_client()
        self._chunks_col = _get_or_create_collection(self._chroma, COLLECTION_CHUNKS)
        self._summaries_col = _get_or_create_collection(self._chroma, COLLECTION_SUMMARIES)
        self._bm25 = BM25Index.load()

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_units(
        self,
        units: list[KnowledgeUnit],
        chunk_embeddings: list[list[float]],
        summary_embeddings: list[list[float]],
        reset_bm25: bool = False,
    ) -> None:
        """
        Store all units in ChromaDB (chunks + summaries collections) and
        rebuild the BM25 index.  Existing entries with the same id are updated.
        """
        if not units:
            return

        ids = [u.id for u in units]

        # Build ChromaDB metadata dicts (values must be str/int/float/bool)
        chunk_metas = [
            {
                "title": u.title,
                "product_category": u.product_category,
                "product_name": u.product_name,
                "topic": u.topic,
                "customer_segment": u.customer_segment,
                "content_type": u.content_type,
                "hierarchy_path": u.hierarchy_path,
                "requires_advisor_verification": u.requires_advisor_verification,
                "source_file": u.source_file,
                "keywords_json": json.dumps(u.keywords),
                "summary": u.summary,
            }
            for u in units
        ]
        chunk_docs = [u.atomic_text for u in units]
        summary_docs = [u.summary for u in units]

        # Upsert to ChromaDB
        self._chunks_col.upsert(
            ids=ids,
            embeddings=chunk_embeddings,
            documents=chunk_docs,
            metadatas=chunk_metas,
        )
        self._summaries_col.upsert(
            ids=ids,
            embeddings=summary_embeddings,
            documents=summary_docs,
            metadatas=chunk_metas,
        )

        # Update/Rebuild BM25
        idx = BM25Index() if reset_bm25 else self._bm25
        for u in units:
            search_text = f"{u.title} {u.atomic_text} {' '.join(u.keywords)}"
            idx.add(u.id, search_text)
        idx.build()
        idx.save()
        self._bm25 = idx

        console.print(
            f"  [green]✓[/green] Stored {len(units)} units "
            f"(chunks col + summaries col + BM25 index)"
        )

    def delete_by_source(self, filename: str) -> int:
        """
        Delete all units associated with the given filename.
        Returns the number of deleted units.
        """
        # Find all units for this filename in ChromaDB first
        results = self._chunks_col.get(
            where={"source_file": filename},
            include=["metadatas"]
        )
        ids = results["ids"]
        if not ids:
            return 0

        # Delete from ChromaDB
        self._chunks_col.delete(ids=ids)
        self._summaries_col.delete(ids=ids)

        # Delete from BM25
        self._bm25.remove_by_ids(ids)
        
        console.print(f"  [red]✖[/red] Deleted {len(ids)} units associated with {filename}")
        return len(ids)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def coarse_search(
        self,
        query_embedding: list[float],
        top_k: int = 3,
        filters: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """
        Stage-1 hierarchical search over SUMMARY embeddings.
        Returns list of (id, distance, metadata).
        """
        where = filters or None
        results = self._summaries_col.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._summaries_col.count() or 1),
            where=where,
            include=["metadatas", "distances"],
        )
        hits = []
        for uid, dist, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["metadatas"][0],
        ):
            hits.append((uid, float(dist), meta))
        return hits

    def dense_search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> list[tuple[str, float, str, dict]]:
        """
        Stage-2 dense vector search over CHUNK embeddings.
        Returns list of (id, distance, document_text, metadata).
        Optionally filter by product_category or other metadata fields.
        """
        n = min(top_k, self._chunks_col.count() or 1)
        results = self._chunks_col.query(
            query_embeddings=[query_embedding],
            n_results=n,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for uid, dist, doc, meta in zip(
            results["ids"][0],
            results["distances"][0],
            results["documents"][0],
            results["metadatas"][0],
        ):
            hits.append((uid, float(dist), doc, meta))
        return hits

    def bm25_search(
        self, query_text: str, top_k: int = 20
    ) -> list[tuple[str, float]]:
        """BM25 keyword search. Returns (id, score) pairs."""
        return self._bm25.search(query_text, top_k=top_k)

    def get_by_ids(self, ids: list[str]) -> list[tuple[str, str, dict]]:
        """
        Fetch specific units by ID.
        Returns list of (id, document_text, metadata).
        """
        if not ids:
            return []
        results = self._chunks_col.get(
            ids=ids, include=["documents", "metadatas"]
        )
        return list(
            zip(results["ids"], results["documents"], results["metadatas"])
        )

    def count(self) -> int:
        return self._chunks_col.count()

    def is_empty(self) -> bool:
        return self.count() == 0
