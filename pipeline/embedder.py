"""
ATB Agent Assist — Gemini Embedding Utility
===========================================
Wraps `gemini-embedding-2-preview` with:
  - Correct task-type selection (RETRIEVAL_DOCUMENT vs RETRIEVAL_QUERY)
  - Batched embedding to respect API limits
  - L2 normalisation for sub-3072-dim vectors (required for cosine similarity)
  - Exponential-backoff retry via tenacity
"""

from __future__ import annotations

import math
import time
import asyncio
from typing import Literal

import numpy as np
from google import genai
from google.genai import types as genai_types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import GEMINI_API_KEY, EMBEDDING_MODEL, EMBEDDING_DIM

# Gemini embedding API limit: 100 texts per batch call
_BATCH_SIZE = 100
# Small inter-batch sleep to stay below RPM limits
_INTER_BATCH_SLEEP_SEC = 0.5

TaskType = Literal[
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "QUESTION_ANSWERING",
    "FACT_VERIFICATION",
]

_client = genai.Client(api_key=GEMINI_API_KEY)
_aclient_instance = None

def _get_aclient():
    global _aclient_instance
    if _aclient_instance is None:
        _aclient_instance = _client.aio
    return _aclient_instance


def _normalise(vec: list[float]) -> list[float]:
    """L2-normalise a vector so cosine similarity == dot product."""
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm < 1e-10:
        return vec
    return (arr / norm).tolist()


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _embed_batch(texts: list[str], task_type: TaskType) -> list[list[float]]:
    """Embed a single batch of texts (≤100) with retry."""
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [_normalise(e.values) for e in result.embeddings]


@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(5),
    reraise=True,
)
async def _async_embed_batch(texts: list[str], task_type: TaskType) -> list[list[float]]:
    """Embed a single batch of texts (≤100) asynchronously with retry."""
    aclient = _get_aclient()
    result = await aclient.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=genai_types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [_normalise(e.values) for e in result.embeddings]


def embed_texts(
    texts: list[str],
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """
    Embed an arbitrary number of texts, batching automatically.

    Args:
        texts:      List of strings to embed.
        task_type:  Use RETRIEVAL_DOCUMENT for KB chunks being indexed,
                    RETRIEVAL_QUERY for queries, QUESTION_ANSWERING for QA.

    Returns:
        List of L2-normalised embedding vectors (length = EMBEDDING_DIM).
    """
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    num_batches = math.ceil(len(texts) / _BATCH_SIZE)

    for i in range(num_batches):
        batch = texts[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
        embeddings = _embed_batch(batch, task_type)
        all_embeddings.extend(embeddings)
        if i < num_batches - 1:
            time.sleep(_INTER_BATCH_SLEEP_SEC)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Convenience: embed a single retrieval query string."""
    return embed_texts([query], task_type="RETRIEVAL_QUERY")[0]


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Convenience: embed KB document chunks for indexing."""
    return embed_texts(texts, task_type="RETRIEVAL_DOCUMENT")


def embed_summaries(summaries: list[str]) -> list[list[float]]:
    """
    Embed one-line summaries for the coarse-retrieval stage.
    Uses SEMANTIC_SIMILARITY since we're matching high-level intent → summary.
    """
    return embed_texts(summaries, task_type="SEMANTIC_SIMILARITY")


async def async_embed_texts(
    texts: list[str],
    task_type: TaskType = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Async: Embed an arbitrary number of texts, batching automatically."""
    if not texts:
        return []

    all_embeddings: list[list[float]] = []
    num_batches = math.ceil(len(texts) / _BATCH_SIZE)

    for i in range(num_batches):
        batch = texts[i * _BATCH_SIZE : (i + 1) * _BATCH_SIZE]
        embeddings = await _async_embed_batch(batch, task_type)
        all_embeddings.extend(embeddings)
        if i < num_batches - 1:
            await asyncio.sleep(_INTER_BATCH_SLEEP_SEC)

    return all_embeddings


async def async_embed_query(query: str) -> list[float]:
    """Async convenience: embed a single retrieval query string."""
    results = await async_embed_texts([query], task_type="RETRIEVAL_QUERY")
    return results[0]
