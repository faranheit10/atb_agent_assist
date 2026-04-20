# ATB Agent Assist — Intelligent KB Retrieval System

Real-time AI coaching for ATB Financial customer service agents.
Uses **Gemini 2.5 Flash** for generation + **gemini-embedding-2-preview** for embeddings,
with a full hybrid RAG pipeline (dense + BM25 + Gemini reranker).

---

## Architecture

```
KB Files (.txt)
     │
     ▼
┌─────────────────────────┐
│  LLM Ingestion Engine   │  ← Gemini 2.5 Flash structures raw docs into
│  (pipeline/ingestion.py)│    atomic KnowledgeUnits with rich metadata
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  Embedding Layer  (pipeline/embedder.py)    │
│  • atomic_text  → RETRIEVAL_DOCUMENT        │  gemini-embedding-2-preview
│  • summary      → SEMANTIC_SIMILARITY       │  768-dim, L2-normalised
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  Storage  (pipeline/vector_store.py)        │
│  • ChromaDB  : chunks + summaries (cosine)  │
│  • BM25Okapi : keyword sparse index         │
└────────────┬────────────────────────────────┘
             │
             ▼ (at query time)
┌─────────────────────────────────────────────┐
│  Retrieval Pipeline  (pipeline/retrieval.py)│
│  Stage 1: Gemini query analysis             │
│  Stage 2: Coarse summary search             │
│  Stage 3: Fine dense search (filtered)      │
│  Stage 4: BM25 keyword search               │
│  Stage 5: RRF hybrid fusion                 │
│  Stage 6: Gemini neural re-ranking          │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│  Generation  (pipeline/generation.py)       │
│  Gemini 2.5 Flash → structured JSON         │
│  • 2-3 suggested responses                  │
│  • Key ATB facts                            │
│  • Recommended actions                      │
│  • Escalation flag                          │
│  • Source attribution                       │
└─────────────────────────────────────────────┘
             │
             ▼
        FastAPI  (api.py)
```

---

## Quick Start

### 1. Install dependencies

```bash
cd atb_agent_assist
pip install -r requirements.txt
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

### 3. Ingest the KB (run once, or when KB files change)

```bash
python ingest.py
```

Expected output:
```
Step 1/3 — LLM Structuring
  ✓ atb_kb_01_chequing_savings.txt → 18 knowledge units
  ✓ atb_kb_02_credit_cards.txt     → 22 knowledge units
  ...
Step 2/3 — Generating Embeddings
  ✓ Generated 140 chunk embeddings
  ✓ Generated 140 summary embeddings
Step 3/3 — Storing in Vector DB + BM25 Index
  ✓ Stored 140 units
```

Typical runtime: 3–6 minutes (Gemini structuring + batch embeddings).

### 4. Start the API server

```bash
python api.py
# or, for development with hot reload:
uvicorn api:app --reload --port 8000
```

### 5. Test a suggestion

```bash
curl -s -X POST http://localhost:8000/suggest \
  -H "Content-Type: application/json" \
  -d '{
    "conversation": [
      {"role": "customer", "content": "Hi, I am a student and want a free account"},
      {"role": "agent", "content": "Happy to help! How old are you?"},
      {"role": "customer", "content": "I am 22 years old, studying at UofA"}
    ]
  }' | python -m json.tool
```

---

## API Reference

### `POST /suggest`

Main endpoint. Takes a conversation, returns structured agent coaching.

**Request:**
```json
{
  "conversation": [
    {"role": "customer", "content": "..."},
    {"role": "agent", "content": "..."},
    {"role": "customer", "content": "..."}
  ],
  "top_k": 5,
  "skip_rerank": false
}
```
The **last message must be from `customer`**. This triggers suggestion generation.

**Response:**
```json
{
  "suggestion": {
    "intent": "student account inquiry",
    "urgency": "low",
    "summary": "22-year-old student wants a free chequing account",
    "suggested_responses": [
      {
        "label": "Direct recommendation",
        "text": "Great news! Since you're 25 or under, you qualify for the ATB Generation Account..."
      }
    ],
    "key_facts": [
      "Generation Account: $0/month fee, unlimited transactions, for clients aged 25 and under",
      "Minimum GIC investment for Generation Account holders: $100"
    ],
    "actions": [
      "Confirm student is 25 years of age or younger (Alberta resident)",
      "Proceed with Generation Account application online or in-branch"
    ],
    "sources": ["atb_kb_01_chequing_savings.txt"],
    "confidence": "high",
    "escalate": false,
    "requires_advisor_verification": false
  },
  "retrieval_count": 5,
  "latency_ms": 1842.3,
  "query_analysis": { ... }
}
```

### `POST /search` (debug)

Raw retrieval without generation — useful for evaluating chunk quality.

```bash
curl -X POST http://localhost:8000/search \
  -d '{"query": "student no fee account", "top_k": 5}'
```

### `GET /health`

Returns store status, model names, and API key presence.

### `GET /stats`

Returns chunk counts by product category and customer segment.

### `POST /ingest`

Triggers a background re-ingestion of all KB files. Check `/health` for completion.

---

## File Structure

```
atb_agent_assist/
├── api.py                  ← FastAPI application (main runtime)
├── ingest.py               ← CLI ingestion script (run once)
├── config.py               ← Settings + Pydantic models
├── requirements.txt
├── .env.example
├── pipeline/
│   ├── __init__.py
│   ├── ingestion.py        ← LLM-driven KB structuring
│   ├── embedder.py         ← gemini-embedding-2-preview wrapper
│   ├── vector_store.py     ← ChromaDB + BM25 index
│   ├── retrieval.py        ← 6-stage hybrid retrieval
│   └── generation.py       ← Agent suggestion generation
├── data/
│   └── kb/                 ← Drop .txt KB files here
├── chroma_db/              ← Auto-created by ChromaDB
└── bm25_index.json         ← Auto-created by ingest.py
```

---

## Connecting to the React Demo

Replace the demo's inline API call URL with your local server:

```js
const response = await fetch('http://localhost:8000/suggest', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ conversation: messages })
});
```

---

## Retrieval Design Notes

### Why hierarchical (coarse → fine)?

The coarse stage searches one-line **summaries** to identify which product
categories are relevant (e.g. chequing, mortgages). The fine stage then
searches only within those categories, dramatically reducing noise.

### Why hybrid (dense + BM25)?

Dense embeddings excel at semantic similarity ("what account has no monthly fee?")
but can miss exact terms. BM25 catches exact product names, rates, and fee
amounts that customers often phrase verbatim. RRF combines both without needing
to tune weights.

### Why Gemini re-ranking?

After RRF fusion, Gemini performs a final semantic relevance pass, understanding
nuances like customer segment (student vs senior) that affect which chunks are
actually most useful. This adds ~300ms but meaningfully improves precision.

### Embedding dimensions

`gemini-embedding-2-preview` outputs 3072-dim by default. We use 768-dim (MRL
truncation) for storage/speed. Per Google's benchmarks, 768-dim achieves
67.99 MTEB vs 68.17 for 3072-dim — less than 0.2% quality loss.
Vectors are L2-normalised after generation (required for non-3072-dim).
