"""
ATB Agent Assist — Hybrid Retrieval Pipeline
=============================================
Implements the full 6-stage retrieval flow described in the architecture doc:

  Stage 1  — Conversation-aware query analysis (Gemini)
  Stage 2  — Coarse retrieval over summary embeddings
  Stage 3  — Fine dense search filtered by product categories
  Stage 4  — BM25 keyword search in parallel
  Stage 5  — Reciprocal Rank Fusion (RRF) hybrid scoring
  Stage 6  — Neural re-ranking with Gemini

The result is a ranked list of KnowledgeUnit objects, each grounded in the
ATB KB and directly relevant to the current customer conversation.
"""

from __future__ import annotations

import json
import re
import asyncio
from typing import Optional

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    FALLBACK_MODEL,
    COARSE_TOP_K,
    DENSE_CANDIDATE_K,
    BM25_CANDIDATE_K,
    RERANK_TOP_K,
    FINAL_TOP_K,
    RRF_K,
    ConversationTurn,
    KnowledgeUnit,
    QueryAnalysis,
    RetrievedChunk,
    ProductCategory,
)
from pipeline.embedder import embed_query, async_embed_query
from pipeline.vector_store import KnowledgeStore

_client = genai.Client(api_key=GEMINI_API_KEY)
_aclient_instance = None

def _get_aclient():
    global _aclient_instance
    if _aclient_instance is None:
        _aclient_instance = _client.aio
    return _aclient_instance


# ---------------------------------------------------------------------------
# Stage 1 — Conversation-aware query analysis
# ---------------------------------------------------------------------------

_QUERY_ANALYSIS_SYSTEM = """\
You are an intent and query extraction engine for an ATB Financial banking
customer service system.

Your job is to analyse the conversation history and extract a structured
QueryAnalysis object that will drive document retrieval.

ATB product categories: chequing, savings, credit_cards, investing,
mortgages, borrowing, customer_service, company_profile.

Customer segments: general, student, senior, newcomer, professional,
business, youth, all.

Urgency guidelines:
  - high: fraud reports, account locked, money missing, complaints
  - medium: application questions, rate comparisons, product decisions
  - low: general information queries

Set requires_escalation=true ONLY for: fraud, unauthorized transactions,
formal complaints, requests to close account.

Respond ONLY with a valid JSON object matching the QueryAnalysis schema.
No markdown, no extra text.\
"""

_QUERY_ANALYSIS_PROMPT = """\
Conversation history:
{conversation}

Latest customer message:
{latest_message}

Extract a QueryAnalysis JSON object with these fields:
{{
  "intent": "<2-4 word label>",
  "product_categories": ["<category>"],
  "product_name": "<specific product or empty string>",
  "customer_segment": "<segment>",
  "rewritten_query": "<clean complete query for embedding>",
  "keywords": ["<keyword1>", "<keyword2>"],
  "conversation_summary": "<one sentence>",
  "urgency": "<low|medium|high>",
  "requires_escalation": false,
  "escalation_reason": ""
}}\
"""


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def analyse_query(conversation: list[ConversationTurn]) -> QueryAnalysis:
    """
    Async: Use Gemini to analyse the conversation and produce structured query metadata.
    """
    if not conversation:
        raise ValueError("Conversation must have at least one turn")

    history_turns = conversation[:-1]
    latest = conversation[-1]

    history_text = "\n".join(
        f"{t.role.upper()}: {t.content}" for t in history_turns
    ) or "(start of conversation)"

    prompt = _QUERY_ANALYSIS_PROMPT.format(
        conversation=history_text,
        latest_message=latest.content,
    )

    aclient = _get_aclient()
    try:
        response = await aclient.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": _QUERY_ANALYSIS_SYSTEM},
        )
    except Exception as exc:
        print(f"Primary model ({GENERATION_MODEL}) failed for query analysis: {exc}. Retrying with fallback ({FALLBACK_MODEL})...")
        response = await aclient.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt,
            config={"system_instruction": _QUERY_ANALYSIS_SYSTEM},
        )

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)
    return QueryAnalysis.model_validate(data)


# ---------------------------------------------------------------------------
# Stage 5 — Reciprocal Rank Fusion
# ---------------------------------------------------------------------------

def _rrf_fuse(
    dense_hits: list[tuple[str, float]],
    bm25_hits: list[tuple[str, float]],
    k: int = RRF_K,
) -> list[tuple[str, float]]:
    """
    Combines dense and BM25 ranked lists using Reciprocal Rank Fusion.

    score(d) = 1/(k + rank_dense(d)) + 1/(k + rank_bm25(d))

    Documents not present in a list get rank = len(list) + 1 (worst possible).
    """
    all_ids = set(uid for uid, _ in dense_hits) | set(uid for uid, _ in bm25_hits)

    dense_rank = {uid: i + 1 for i, (uid, _) in enumerate(dense_hits)}
    bm25_rank = {uid: i + 1 for i, (uid, _) in enumerate(bm25_hits)}
    dense_fallback = len(dense_hits) + 1
    bm25_fallback = len(bm25_hits) + 1

    scores: dict[str, float] = {}
    for uid in all_ids:
        rrf = 1.0 / (k + dense_rank.get(uid, dense_fallback)) + \
              1.0 / (k + bm25_rank.get(uid, bm25_fallback))
        scores[uid] = rrf

    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Stage 6 — Gemini neural re-ranker
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """\
You are a relevance scoring engine for ATB Financial customer service.
Given a customer query and a list of knowledge chunks, return a JSON array
of chunk IDs sorted from MOST to LEAST relevant.

Only include the IDs, in order. No explanations. No markdown.\
"""


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def _gemini_rerank(
    query: str,
    chunks: list[tuple[str, str]],  # (id, text)
) -> list[str]:
    """
    Async: Ask Gemini to re-rank chunks by relevance.
    """
    chunk_str = "\n\n".join(
        f"[{uid}]\n{text[:400]}" for uid, text in chunks
    )
    prompt = (
        f"Customer query: {query}\n\n"
        f"Chunks to rank:\n{chunk_str}\n\n"
        f"Return a JSON array of IDs sorted most-to-least relevant.\n"
        f"Example: [\"id_a\", \"id_b\", \"id_c\"]"
    )
    aclient = _get_aclient()
    try:
        response = await aclient.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": _RERANK_SYSTEM},
        )
    except Exception as exc:
        print(f"Primary model ({GENERATION_MODEL}) failed for reranking: {exc}. Retrying with fallback ({FALLBACK_MODEL})...")
        response = await aclient.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt,
            config={"system_instruction": _RERANK_SYSTEM},
        )
    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Helper — reconstruct KnowledgeUnit from ChromaDB metadata + document text
# ---------------------------------------------------------------------------

def _build_unit(unit_id: str, doc_text: str, meta: dict) -> KnowledgeUnit:
    return KnowledgeUnit(
        id=unit_id,
        title=meta.get("title", ""),
        product_category=meta.get("product_category", "customer_service"),
        product_name=meta.get("product_name", ""),
        topic=meta.get("topic", "general"),
        customer_segment=meta.get("customer_segment", "general"),
        content_type=meta.get("content_type", "definition"),
        atomic_text=doc_text,
        summary=meta.get("summary", ""),
        keywords=json.loads(meta.get("keywords_json", "[]")),
        hierarchy_path=meta.get("hierarchy_path", ""),
        requires_advisor_verification=bool(
            meta.get("requires_advisor_verification", False)
        ),
        source_file=meta.get("source_file", ""),
    )


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

async def async_retrieve(
    conversation: list[ConversationTurn],
    store: KnowledgeStore,
    final_top_k: int = FINAL_TOP_K,
    skip_rerank: bool = False,
) -> tuple[list[RetrievedChunk], QueryAnalysis]:
    """
    High-performance 6-stage hybrid retrieval pipeline using speculative parallel execution.
    """
    # ── Stage 1 & 2: Speculative Parallel Start ──────────────────────────
    raw_query = conversation[-1].content
    
    # Task 1: Semantic analysis of intent (Slowest)
    analysis_task = asyncio.create_task(analyse_query(conversation))
    
    # Task 2: Speculative Raw Embedder + Dense search
    raw_emb_task = asyncio.create_task(async_embed_query(raw_query))
    
    # Await raw embedding to trigger raw search immediately
    raw_query_emb = await raw_emb_task
    
    # Start raw searches in parallel with the ongoing LLM analysis
    raw_dense_task = asyncio.to_thread(store.dense_search, raw_query_emb, top_k=DENSE_CANDIDATE_K)
    raw_bm25_task = asyncio.to_thread(store.bm25_search, raw_query, top_k=BM25_CANDIDATE_K)
    
    # Now await the Analysis to get the refined query
    try:
        analysis = await analysis_task
    except Exception as exc:
        print(f"Query analysis failed: {exc}. Using raw query fallback.")
        analysis = QueryAnalysis(
            intent="general enquiry",
            product_categories=[],
            product_name="",
            customer_segment="general",
            rewritten_query=raw_query,
            keywords=raw_query.split()[:5],
            conversation_summary="Analysis failed.",
            urgency="low"
        )

    # Task 3: Refined Search (using rewritten query if different)
    refined_dense_task = None
    if analysis.rewritten_query != raw_query:
        refined_emb = await async_embed_query(analysis.rewritten_query)
        # Apply category filters if identified
        where_filter: Optional[dict] = None
        if analysis.product_categories and len(analysis.product_categories) <= 3:
            cat_list = list(analysis.product_categories)
            where_filter = {"product_category": cat_list[0]} if len(cat_list) == 1 else {"product_category": {"$in": cat_list}}
        
        refined_dense_task = asyncio.to_thread(store.dense_search, refined_emb, top_k=DENSE_CANDIDATE_K, where=where_filter)

    # ── Join All Searches ────────────────────────────────────────────────
    raw_dense_results, raw_bm25_hits = await asyncio.gather(raw_dense_task, raw_bm25_task)
    refined_dense_results = await refined_dense_task if refined_dense_task else []

    # ── Stage 5: Merged RRF Hybrid Fusion ────────────────────────────────
    # Combine hits from Raw and Refined searches
    dense_hits_map = {uid: score for uid, score, _, _ in raw_dense_results}
    for uid, score, _, _ in refined_dense_results:
        # If already present, keep the better (lower) distance
        dense_hits_map[uid] = min(dense_hits_map.get(uid, score), score)
    
    dense_list = sorted(dense_hits_map.items(), key=lambda x: x[1]) # distances sorted ASC
    fused = _rrf_fuse(dense_list, raw_bm25_hits)
    top_ids_for_rerank = [uid for uid, _ in fused[:RERANK_TOP_K]]

    # Build lookup
    id_to_data = {uid: (doc, meta) for uid, _, doc, meta in raw_dense_results + refined_dense_results}
    bm25_only_ids = [uid for uid in top_ids_for_rerank if uid not in id_to_data]
    if bm25_only_ids:
        bm25_data = await asyncio.to_thread(store.get_by_ids, bm25_only_ids)
        for uid, doc, meta in bm25_data:
            id_to_data[uid] = (doc, meta)

    # ── Stage 6: Async Neural Re-ranking ───────────────────────────────
    chunks_for_rerank = [(uid, id_to_data[uid][0]) for uid in top_ids_for_rerank if uid in id_to_data]
    
    if not skip_rerank and len(chunks_for_rerank) > final_top_k:
        try:
            reranked_ids = await _gemini_rerank(analysis.rewritten_query, chunks_for_rerank)
            ordered_ids = reranked_ids[:final_top_k]
        except Exception:
            ordered_ids = [uid for uid, _ in chunks_for_rerank[:final_top_k]]
    else:
        ordered_ids = [uid for uid, _ in chunks_for_rerank[:final_top_k]]

    # ── Final Assembly ───────────────────────────────────────────────────
    rrf_score_map = dict(fused)
    results: list[RetrievedChunk] = []

    for uid in ordered_ids:
        if uid not in id_to_data: continue
        doc_text, meta = id_to_data[uid]
        results.append(RetrievedChunk(
            unit=_build_unit(uid, doc_text, meta),
            score=rrf_score_map.get(uid, 0.0),
            retrieval_method="reranked" if not skip_rerank else "hybrid"
        ))

    return results, analysis
