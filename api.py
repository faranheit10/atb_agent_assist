"""
ATB Agent Assist — FastAPI Server
===================================
Exposes the full retrieval + generation pipeline as an HTTP API.

Endpoints:
  POST /suggest          — Main endpoint: takes conversation, returns AgentSuggestion
  POST /ingest           — Trigger a fresh ingestion run (admin)
  GET  /health           — Health check + store status
  GET  /search           — Debug: raw retrieval results for a query
  GET  /stats            — Vector store statistics

Start with:
    python api.py               # defaults: host=0.0.0.0, port=8000
    uvicorn api:app --reload    # dev mode with hot reload

The frontend (demo artifact) can call POST /suggest directly.
"""

from __future__ import annotations

import time
import os
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    BackgroundTasks,
    File,
    UploadFile,
    Form,
    Query,
)
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from rich.console import Console

from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    FINAL_TOP_K,
    AgentSuggestion,
    ConversationTurn,
    KnowledgeUnit,
)

# Global status tracking for KB files
# format: { filename: { "status": "indexed|processing|error", "progress": 0-100 } }
_kb_status: dict[str, dict] = {}
from pipeline.vector_store import KnowledgeStore
from pipeline.retrieval import async_retrieve
from pipeline.generation import generate_suggestions, generate_fallback_suggestion, generate_suggestions_stream

console = Console()
start_time = time.time()

# ---------------------------------------------------------------------------
# Global store — initialised once at startup
# ---------------------------------------------------------------------------

_store: Optional[KnowledgeStore] = None


def get_store() -> KnowledgeStore:
    global _store
    if _store is None:
        _store = KnowledgeStore()
    app.state.store = _store
    return _store


# ---------------------------------------------------------------------------
# Lifespan: warm up store on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _store
    console.print("[cyan]ATB Agent Assist API starting up...[/cyan]")
    if not GEMINI_API_KEY:
        console.print("[red][!] GEMINI_API_KEY not set - requests will fail[/red]")
    _store = KnowledgeStore()
    count = _store.count()
    if count == 0:
        console.print(
            "[yellow][!] Vector store is empty. Starting automatic background ingestion...[/yellow]"
        )
        asyncio.create_task(run_background_ingest(force=True))
    else:
        console.print(f"[green][OK] Vector store ready - {count} knowledge chunks loaded[/green]")
    yield
    console.print("[dim]Shutting down...[/dim]")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ATB Agent Assist API",
    description=(
        "Real-time AI coaching for ATB Financial customer service agents. "
        "Retrieves grounded suggestions from the ATB Knowledge Base using "
        "Gemini embeddings + hybrid RAG."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response schemas
# ---------------------------------------------------------------------------

class SuggestRequest(BaseModel):
    conversation: list[ConversationTurn]
    top_k: int = FINAL_TOP_K
    skip_rerank: bool = False

    class Config:
        json_schema_extra = {
            "example": {
                "conversation": [
                    {"role": "customer", "content": "Hi, I'm looking to open a chequing account."},
                    {"role": "agent", "content": "Of course! What's most important to you — unlimited transactions, no fees, or special perks?"},
                    {"role": "customer", "content": "I'm a student and don't want to pay monthly fees."},
                ]
            }
        }


class SuggestResponse(BaseModel):
    suggestion: AgentSuggestion
    retrieval_count: int
    latency_ms: float
    query_analysis: dict


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResult(BaseModel):
    id: str
    title: str
    product_category: str
    hierarchy_path: str
    score: float
    retrieval_method: str
    snippet: str


class IngestResponse(BaseModel):
    message: str
    units_ingested: int




# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Serve the frontend index.html."""
    return FileResponse("index.html")




@app.post("/suggest", response_model=SuggestResponse, tags=["Agent Assist"])
async def suggest(req: SuggestRequest) -> SuggestResponse:
    """
    Main endpoint — takes a conversation and returns structured agent coaching.

    The last message in the conversation must be from the customer.
    """
    if not req.conversation:
        raise HTTPException(status_code=422, detail="Conversation cannot be empty")

    if req.conversation[-1].role != "customer":
        raise HTTPException(
            status_code=422,
            detail="Last message in conversation must be from 'customer'",
        )

    store = get_store()
    t0 = time.perf_counter()

    # If store is empty, return a graceful fallback
    if store.is_empty():
        from pipeline.retrieval import analyse_query
        try:
            analysis = analyse_query(req.conversation)
        except Exception:
            from config import QueryAnalysis
            analysis = QueryAnalysis(
                intent="general inquiry",
                product_categories=[],
                product_name="",
                rewritten_query=req.conversation[-1].content,
                keywords=[],
                conversation_summary="Customer needs assistance",
                urgency="low",
            )
        suggestion = generate_fallback_suggestion(analysis)
        latency = (time.perf_counter() - t0) * 1000
        return SuggestResponse(
            suggestion=suggestion,
            retrieval_count=0,
            latency_ms=round(latency, 1),
            query_analysis=analysis.model_dump(),
        )

    try:
        # Full pipeline
        chunks, analysis = await async_retrieve(
            req.conversation,
            store,
            final_top_k=req.top_k,
            skip_rerank=req.skip_rerank,
        )
        suggestion = generate_suggestions(req.conversation, chunks, analysis)
        latency = (time.perf_counter() - t0) * 1000

        return SuggestResponse(
            suggestion=suggestion,
            retrieval_count=len(chunks),
            latency_ms=round(latency, 1),
            query_analysis=analysis.model_dump(),
        )

    except Exception as exc:
        console.print(f"[red]Suggest error: {exc}[/red]")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(exc)}")


@app.post("/suggest_stream", tags=["Agent Assist"])
async def suggest_stream(req: SuggestRequest):
    """
    Streaming version of the suggest endpoint.
    Returns a stream of JSON tokens from Gemini.
    """
    if not req.conversation or req.conversation[-1].role != "customer":
        raise HTTPException(status_code=422, detail="Invalid conversation")

    store = get_store()
    
    # Run retrieval asynchronously (speculative parallel)
    chunks, analysis = await async_retrieve(
        req.conversation,
        store,
        final_top_k=req.top_k,
        skip_rerank=req.skip_rerank,
    )

    async def stream_generator():
        try:
            async for chunk in generate_suggestions_stream(req.conversation, chunks, analysis):
                yield chunk
        except Exception as exc:
            console.print(f"[red]Streaming error: {exc}[/red]")
            yield f"\n\nERROR: {str(exc)}"

    return StreamingResponse(
        stream_generator(),
        media_type="text/plain",
    )


@app.get("/health", tags=["System"])
async def health():
    """Health check + active model status."""
    from config import GENERATION_MODEL, FALLBACK_MODEL
    return {
        "status": "online",
        "model": GENERATION_MODEL,
        "fallback_model": FALLBACK_MODEL,
        "uptime": time.time() - start_time,
        "timestamp": time.ctime()
    }


@app.post("/search", response_model=list[SearchResult], tags=["Debug"])
async def search(req: SearchRequest) -> list[SearchResult]:
    """
    Debug endpoint — runs raw hybrid retrieval and returns ranked chunks
    without the generation step. Useful for evaluating retrieval quality.
    """
    store = get_store()
    if store.is_empty():
        raise HTTPException(status_code=503, detail="Vector store is empty. Run ingestion first.")

    from config import ConversationTurn as CT
    fake_conv = [CT(role="customer", content=req.query)]

    try:
        chunks, _ = await async_retrieve(fake_conv, store, final_top_k=req.top_k)
        return [
            SearchResult(
                id=rc.unit.id,
                title=rc.unit.title,
                product_category=rc.unit.product_category,
                hierarchy_path=rc.unit.hierarchy_path,
                score=round(rc.score, 6),
                retrieval_method=rc.retrieval_method,
                snippet=rc.unit.atomic_text[:200] + "…",
            )
            for rc in chunks
        ]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats", tags=["System"])
async def stats() -> dict:
    """Vector store statistics broken down by product category."""
    store = get_store()
    if store.is_empty():
        return {"total": 0, "message": "Store is empty"}

    # Pull all metadata to aggregate by category
    results = store._chunks_col.get(include=["metadatas"])
    from collections import Counter
    cats = Counter(m.get("product_category", "unknown") for m in results["metadatas"])
    segs = Counter(m.get("customer_segment", "unknown") for m in results["metadatas"])

    return {
        "total_chunks": store.count(),
        "by_product_category": dict(cats),
        "by_customer_segment": dict(segs),
    }


@app.post("/ingest", response_model=IngestResponse, tags=["System"])
async def trigger_ingest(background_tasks: BackgroundTasks, force: bool = False):
    """Trigger background ingestion of all files in the KB directory."""
    background_tasks.add_task(run_background_ingest, force=force)
    return IngestResponse(
        message="Ingestion started in background. Check KB dashboard for status.",
        units_ingested=0,
    )

async def run_background_ingest(force: bool = False):
    """The core ingestion worker used by startup and API."""
    import asyncio
    from pipeline.ingestion import ingest_all_kb_files, KB_DIR
    from pipeline.embedder import embed_documents, embed_summaries
    
    all_files = [f.name for f in KB_DIR.glob("*.txt")]
    for f in all_files:
        _kb_status[f] = {"status": "indexing", "progress": 10}
        
    try:
        # 1. Structure documents with Gemini
        # We run this in a threadpool because it's synchronous/blocking
        loop = asyncio.get_event_loop()
        units = await loop.run_in_executor(None, ingest_all_kb_files, KB_DIR, force)
        
        if units:
            # Map units to their source files to update progress
            files_involved = list(set(u.source_file for u in units))
            for f in files_involved: _kb_status[f]["progress"] = 60
            
            # 2. Embed chunks
            chunk_embs = await loop.run_in_executor(None, embed_documents, [u.atomic_text for u in units])
            for f in files_involved: _kb_status[f]["progress"] = 80
            
            # 3. Embed summaries
            summary_embs = await loop.run_in_executor(None, embed_summaries, [u.summary for u in units])
            for f in files_involved: _kb_status[f]["progress"] = 90
            
            # 4. Storage
            store = get_store()
            await loop.run_in_executor(None, store.upsert_units, units, chunk_embs, summary_embs, force)
            
            for f in files_involved:
                _kb_status[f] = {"status": "indexed", "progress": 100}
                
            console.print(f"[green]Background ingestion complete: {len(units)} units[/green]")
        else:
            # Nothing processed (everything skipped)
            for f in all_files:
                _kb_status[f] = {"status": "indexed", "progress": 100}
    except Exception as e:
        console.print(f"[red]Background ingestion FAILED: {e}[/red]")
        for f in all_files:
            if _kb_status.get(f, {}).get("status") == "indexing":
                _kb_status[f] = {"status": "error", "progress": 0}


# --- Knowledge Base Endpoints ---

@app.get("/kb_internal", tags=["Knowledge Base"])
async def list_kb_files():
    """List all text files in the knowledge base directory with ingestion status."""
    from pipeline.ingestion import KB_DIR, load_ingestion_state
    
    if not KB_DIR.exists():
        return []
    
    state = load_ingestion_state()
    files = []
    
    for f in KB_DIR.glob("*.txt"):
        filename = f.name
        cached = state.get(filename) # Use .get() without default to check existence
        
        # Determine status
        status_info = _kb_status.get(filename, {})
        status = status_info.get("status")
        progress = status_info.get("progress", 0)
        
        # If no active background status, check if it's in the persisted state
        if not status:
            if cached and cached.get("hash"):
                status = "indexed"
                progress = 100
            else:
                status = "new"
                progress = 0

        files.append({
            "filename": filename,
            "title": f.stem.replace("_", " ").title(),
            "size_kb": round(f.stat().st_size / 1024, 1),
            "last_modified": time.ctime(f.stat().st_mtime),
            "status": status,
            "progress": progress,
            "error": status_info.get("error")
        })
    return sorted(files, key=lambda x: x["filename"])


@app.post("/kb_internal/upload", tags=["Knowledge Base"])
async def upload_kb_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    overwrite: bool = Query(False)
):
    """Upload a new KB document and trigger individual indexing."""
    kb_dir = Path("data/kb")
    kb_dir.mkdir(parents=True, exist_ok=True)
    
    filename = file.filename
    if not filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are allowed")
        
    target_path = kb_dir / filename
    if target_path.exists() and not overwrite:
        return JSONResponse(
            status_code=409, 
            content={"message": "File already exists", "filename": filename}
        )

    # Save file
    try:
        content = await file.read()
        target_path.write_bytes(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    # Trigger background ingestion
    _kb_status[filename] = {"status": "processing", "progress": 10}
    
    def run_single_ingest(fname: str):
        from pipeline.ingestion import ingest_single_file
        from pipeline.embedder import embed_documents, embed_summaries
        try:
            _kb_status[fname]["progress"] = 33 # Step 1: Structuring
            units = ingest_single_file(fname)
            
            if units:
                _kb_status[fname]["progress"] = 66 # Step 2: Embedding
                chunk_embs = embed_documents([u.atomic_text for u in units])
                summary_embs = embed_summaries([u.summary for u in units])
                
                _kb_status[fname]["progress"] = 90 # Step 3: Indexing
                store = get_store()
                store.upsert_units(units, chunk_embs, summary_embs, reset_bm25=False)
                
                # Update ingestion state so it shows as indexed after reboot
                from pipeline.ingestion import calculate_file_hash, save_ingestion_state, load_ingestion_state
                state = load_ingestion_state()
                kb_path = Path("data/kb") / fname
                state[fname] = {
                    "hash": calculate_file_hash(kb_path),
                    "last_ingested": time.ctime(),
                    "units": len(units)
                }
                save_ingestion_state(state)
                
            _kb_status[fname] = {"status": "indexed", "progress": 100}
        except Exception as e:
            console.print(f"[red]Failed to ingest {fname}: {e}[/red]")
            _kb_status[fname] = {"status": "error", "progress": 0, "error": str(e)}

    background_tasks.add_task(run_single_ingest, filename)
    return {"message": "Upload successful, indexing started.", "filename": filename}


@app.get("/kb_internal/{filename}", tags=["Knowledge Base"])
async def get_kb_content(filename: str):
    """Get the raw content of a specific KB file."""
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
        
    kb_path = Path("data/kb") / filename
    if not kb_path.exists() or not kb_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        content = kb_path.read_text(encoding="utf-8")
        return {"filename": filename, "content": content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


@app.delete("/kb_internal/{filename}", tags=["Knowledge Base"])
async def delete_kb_file(filename: str):
    """Delete a file from the KB and purge its chunks from the vector store."""
    # Security: prevent directory traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    kb_dir = Path("data/kb")
    target_path = kb_dir / filename
    
    # 1. Delete from Vector Store
    try:
        store = get_store()
        deleted_count = store.delete_by_source(filename)
    except Exception as e:
        console.print(f"[red]Failed to delete chunks for {filename}: {e}[/red]")
        # Continue anyway to try and clean up local files
        deleted_count = 0

    # 2. Delete Physical File
    file_deleted = False
    if target_path.exists():
        try:
            target_path.unlink()
            file_deleted = True
        except Exception as e:
            console.print(f"[red]Failed to delete file {filename}: {e}[/red]")

    # 3. Purge Ingestion State
    from pipeline.ingestion import load_ingestion_state, save_ingestion_state
    state = load_ingestion_state()
    if filename in state:
        del state[filename]
        save_ingestion_state(state)

    # 4. Cleanup in-memory status
    if filename in _kb_status:
        del _kb_status[filename]

    if not file_deleted and deleted_count == 0:
         raise HTTPException(status_code=404, detail="File and chunks not found")

    return {
        "message": f"Successfully deleted {filename}",
        "chunks_purged": deleted_count,
        "file_removed": file_deleted
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info",
    )
