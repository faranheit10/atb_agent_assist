"""
ATB Agent Assist — Central Configuration & Shared Data Models
=============================================================
All constants, paths, model names, and Pydantic schemas live here so every
module imports from one source of truth.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
KB_DIR = BASE_DIR / "data" / "kb"
CHROMA_DIR = BASE_DIR / "chroma_db"
BM25_INDEX_PATH = BASE_DIR / "bm25_index.json"
INGESTION_STATE_PATH = BASE_DIR / "data" / "ingestion_state.json"

# ---------------------------------------------------------------------------
# Gemini model identifiers
# ---------------------------------------------------------------------------
GENERATION_MODEL = "gemini-3.1-flash-lite-preview"  # primary: conversation + structuring + reranking
FALLBACK_MODEL = "gemini-2.5-flash"              # fallback for stability
EMBEDDING_MODEL = "gemini-embedding-2-preview"        # all vector embeddings

# Embedding output size — 768 is recommended by Google for storage/speed balance;
# still requires explicit L2 normalisation (only 3072-dim is auto-normalised).
EMBEDDING_DIM = 768

# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# ChromaDB collection names
# ---------------------------------------------------------------------------
COLLECTION_CHUNKS = "atb_kb_chunks"       # dense embeddings of atomic_text
COLLECTION_SUMMARIES = "atb_kb_summaries" # dense embeddings of one-line summaries

# ---------------------------------------------------------------------------
# Retrieval hyper-parameters
# ---------------------------------------------------------------------------
COARSE_TOP_K = 3          # summary hits to determine relevant product categories
DENSE_CANDIDATE_K = 20    # dense candidates before hybrid fusion
BM25_CANDIDATE_K = 20     # BM25 candidates before hybrid fusion
RERANK_TOP_K = 10         # chunks sent to Gemini for neural re-ranking
FINAL_TOP_K = 5           # chunks injected into the generation prompt
RRF_K = 60                # reciprocal rank fusion constant (standard default)
MAX_CONTEXT_TOKENS = 6000 # approximate token budget for retrieved context

# ---------------------------------------------------------------------------
# Pydantic models — Ingestion
# ---------------------------------------------------------------------------

ProductCategory = Literal[
    "chequing", "savings", "credit_cards", "investing",
    "mortgages", "borrowing", "customer_service", "company_profile"
]

ContentType = Literal[
    "definition", "rule", "rate", "exception",
    "faq", "procedure", "comparison", "contact"
]

Topic = Literal[
    "fees", "eligibility", "rates", "benefits", "how_to",
    "faq", "comparison", "limits", "procedure", "contact", "general"
]

CustomerSegment = Literal[
    "general", "student", "senior", "newcomer",
    "professional", "business", "youth", "all"
]


class KnowledgeUnit(BaseModel):
    """A single atomic piece of ATB knowledge extracted by the LLM ingestion engine."""

    id: str = Field(description="Unique identifier, e.g. 'chequing_unlimited_fees_001'")
    title: str = Field(description="Concise human-readable title")
    product_category: ProductCategory
    product_name: str = Field(description="Specific product, e.g. 'Unlimited Chequing Account'")
    topic: Topic
    customer_segment: CustomerSegment = "all"
    content_type: ContentType
    atomic_text: str = Field(
        description="Self-contained answer — includes all context needed to answer without referencing the source doc"
    )
    summary: str = Field(description="One sentence summary for coarse retrieval")
    keywords: list[str] = Field(description="Search keywords for BM25/sparse retrieval")
    hierarchy_path: str = Field(description="e.g. 'chequing > unlimited > fees'")
    requires_advisor_verification: bool = False
    source_file: str = ""


class IngestedChunk(KnowledgeUnit):
    """A KnowledgeUnit that has been embedded and stored."""

    chunk_embedding: Optional[list[float]] = None
    summary_embedding: Optional[list[float]] = None


# ---------------------------------------------------------------------------
# Pydantic models — Retrieval
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    role: Literal["customer", "agent"]
    content: str


class QueryAnalysis(BaseModel):
    """Structured output from the conversation-aware query rewriter."""

    intent: str = Field(description="2-4 word intent label, e.g. 'student account inquiry'")
    product_categories: list[ProductCategory] = Field(
        description="Relevant product categories to filter retrieval"
    )
    product_name: str = Field(description="Specific product if identifiable, else ''")
    customer_segment: CustomerSegment = "general"
    rewritten_query: str = Field(description="Clean, complete query for embedding")
    keywords: list[str] = Field(description="Terms for BM25 search")
    conversation_summary: str = Field(
        description="One sentence summary of the conversation so far"
    )
    urgency: Literal["low", "medium", "high"] = "low"
    requires_escalation: bool = False
    escalation_reason: str = ""


class RetrievedChunk(BaseModel):
    """A chunk returned from the retrieval pipeline with its score."""

    unit: KnowledgeUnit
    score: float
    retrieval_method: str  # "dense", "bm25", "hybrid", "reranked"


# ---------------------------------------------------------------------------
# Pydantic models — Generation (Agent Suggestions)
# ---------------------------------------------------------------------------

class SuggestedResponse(BaseModel):
    label: str
    text: str


class AgentSuggestion(BaseModel):
    """Structured coaching output sent to the agent UI."""

    intent: str
    urgency: Literal["low", "medium", "high"]
    summary: str
    suggested_responses: list[SuggestedResponse]
    key_facts: list[str]
    actions: list[str]
    sources: list[str]
    confidence: Literal["high", "medium", "low"]
    escalate: bool = False
    escalation_reason: str = ""
    escalation_to: str = ""
    requires_advisor_verification: bool = False
