"""
ATB Agent Assist — Agent Suggestion Generator
==============================================
Takes the retrieved KB chunks + conversation context and calls Gemini to
produce structured coaching suggestions for the agent:

  - 2-3 suggested response drafts (grounded, ready to send)
  - Key ATB facts relevant to this query
  - Recommended next actions
  - Escalation flag if needed
  - Confidence score based on retrieved evidence quality
  - Source attribution for compliance

The generation prompt is designed to be strict: Gemini is instructed to
ONLY use the retrieved chunks and to flag anything uncertain.
"""

from __future__ import annotations

import json
import re

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    FALLBACK_MODEL,
    MAX_CONTEXT_TOKENS,
    AgentSuggestion,
    ConversationTurn,
    KnowledgeUnit,
    QueryAnalysis,
    RetrievedChunk,
    SuggestedResponse,
)

_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_GENERATION_SYSTEM = """\
You are a real-time AI coaching assistant for ATB Financial customer service
agents. You help agents respond accurately and helpfully to customers.

STRICT RULES:
1. Only use facts explicitly present in the provided KNOWLEDGE CHUNKS.
   Never invent rates, fees, product names, or eligibility criteria.
2. If the chunks do not contain enough information, say so clearly — do not
   guess or fill in gaps from general banking knowledge.
3. For anything requiring current rates or credit approval, always note that
   the agent should verify with an ATB advisor or at atb.com/resources/rates.
4. Suggested responses must be professional, empathetic, and concise.
   Write them as if the agent is speaking directly to the customer.
5. Set escalate=true only for: fraud, unauthorized transactions, formal
   complaints, or requests to speak with a manager.
6. Always include source attribution (source_file) for compliance.

Return ONLY a valid JSON object matching the AgentSuggestion schema.
No markdown fences. No extra text.\
"""

_GENERATION_PROMPT = """\
CONVERSATION HISTORY:
{conversation}

LATEST CUSTOMER MESSAGE:
{latest_message}

CONVERSATION ANALYSIS:
- Intent: {intent}
- Customer Segment: {customer_segment}
- Urgency: {urgency}
- Requires Escalation: {requires_escalation}
{escalation_note}

KNOWLEDGE CHUNKS (use ONLY these to form your response):
{chunks}

Generate an AgentSuggestion JSON object with this exact structure:
{{
  "intent": "{intent}",
  "urgency": "{urgency}",
  "summary": "<one sentence: what the customer needs>",
  "suggested_responses": [
    {{"label": "<option name>", "text": "<full agent response ready to send>"}},
    {{"label": "<option name>", "text": "<full agent response ready to send>"}}
  ],
  "key_facts": [
    "<specific fact from KB relevant to this query>",
    "<another specific fact>"
  ],
  "actions": [
    "<concrete action the agent should take>",
    "<another action>"
  ],
  "sources": ["<source_file_1>", "<source_file_2>"],
  "confidence": "<high|medium|low>",
  "escalate": false,
  "escalation_reason": "",
  "escalation_to": "",
  "requires_advisor_verification": false
}}

Guidelines:
- suggested_responses: 2-3 options covering different approaches (e.g. direct
  answer vs. offer to help more vs. offer to schedule advisor call)
- key_facts: 3-5 specific, precise facts (include exact numbers, rates, limits)
- actions: 2-3 concrete steps (e.g. "Confirm customer's age for Generation Account eligibility")
- confidence: "high" if chunks directly answer the question; "medium" if partial
  match; "low" if chunks don't fully cover the question
- Set requires_advisor_verification=true if rates or credit decisions are involved\
"""

# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks into a numbered context string for the prompt."""
    parts = []
    for i, rc in enumerate(chunks, start=1):
        u = rc.unit
        advisor_note = " ⚠ Verify with advisor" if u.requires_advisor_verification else ""
        parts.append(
            f"[{i}] {u.title}{advisor_note}\n"
            f"    Source: {u.source_file} | {u.hierarchy_path}\n"
            f"    {u.atomic_text}"
        )
    # Rough token budget: ~4 chars per token
    context = "\n\n".join(parts)
    if len(context) > MAX_CONTEXT_TOKENS * 4:
        context = context[: MAX_CONTEXT_TOKENS * 4] + "\n... [context truncated]"
    return context


def _format_conversation(turns: list[ConversationTurn]) -> str:
    return "\n".join(
        f"{'CUSTOMER' if t.role == 'customer' else 'AGENT'}: {t.content}"
        for t in turns
    )


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def generate_suggestions(
    conversation: list[ConversationTurn],
    retrieved_chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
) -> AgentSuggestion:
    """
    Generate structured agent coaching suggestions.

    Args:
        conversation:     Full conversation history
        retrieved_chunks: Output of the retrieval pipeline
        analysis:         Query analysis from Stage 1 of retrieval

    Returns:
        AgentSuggestion with suggested responses, key facts, and actions
    """
    if not conversation:
        raise ValueError("Conversation cannot be empty")

    latest = conversation[-1]
    history = conversation[:-1]

    escalation_note = ""
    if analysis.requires_escalation and analysis.escalation_reason:
        escalation_note = f"- Escalation reason: {analysis.escalation_reason}"

    prompt = _GENERATION_PROMPT.format(
        conversation=_format_conversation(history) or "(start of conversation)",
        latest_message=latest.content,
        intent=analysis.intent,
        customer_segment=analysis.customer_segment,
        urgency=analysis.urgency,
        requires_escalation=analysis.requires_escalation,
        escalation_note=escalation_note,
        chunks=_format_chunks(retrieved_chunks) if retrieved_chunks else "No relevant chunks found.",
    )

    try:
        response = _client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": _GENERATION_SYSTEM},
        )
    except Exception as exc:
        print(f"Primary model ({GENERATION_MODEL}) failed: {exc}. Retrying with fallback ({FALLBACK_MODEL})...")
        response = _client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=prompt,
            config={"system_instruction": _GENERATION_SYSTEM},
        )

    try:
        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)

        # Ensure escalation fields from analysis flow through
        if analysis.requires_escalation and not data.get("escalate"):
            data["escalate"] = True
            data["escalation_reason"] = analysis.escalation_reason

        return AgentSuggestion.model_validate(data)
    except Exception as exc:
        print(f"Suggestion generation failed completely: {exc}. Returning fallback.")
        return generate_fallback_suggestion(analysis)


async def generate_suggestions_stream(
    conversation: list[ConversationTurn],
    retrieved_chunks: list[RetrievedChunk],
    analysis: QueryAnalysis,
):
    """
    Asynchronous generator that yields tokens from Gemini for real-time UI typing.
    Includes fallback logic for model reliability.
    """
    if not conversation:
        raise ValueError("Conversation cannot be empty")

    latest = conversation[-1]
    history = conversation[:-1]

    escalation_note = ""
    if analysis.requires_escalation and analysis.escalation_reason:
        escalation_note = f"- Escalation reason: {analysis.escalation_reason}"

    prompt = _GENERATION_PROMPT.format(
        conversation=_format_conversation(history) or "(start of conversation)",
        latest_message=latest.content,
        intent=analysis.intent,
        customer_segment=analysis.customer_segment,
        urgency=analysis.urgency,
        requires_escalation=analysis.requires_escalation,
        escalation_note=escalation_note,
        chunks=_format_chunks(retrieved_chunks) if retrieved_chunks else "No relevant chunks found.",
    )

    try:
        # Try Primary Model
        response_stream = await _client.aio.models.generate_content_stream(
            model=GENERATION_MODEL,
            contents=prompt,
            config={"system_instruction": _GENERATION_SYSTEM},
        )
        async for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    except Exception as exc:
        print(f"Primary streaming model ({GENERATION_MODEL}) failed: {exc}. Retrying with fallback ({FALLBACK_MODEL})...")
        try:
            # Try Fallback Model
            response_stream = await _client.aio.models.generate_content_stream(
                model=FALLBACK_MODEL,
                contents=prompt,
                config={"system_instruction": _GENERATION_SYSTEM},
            )
            async for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
        except Exception as fallback_exc:
            print(f"Fallback streaming model failed: {fallback_exc}. Yielding fallback JSON.")
            yield json.dumps(generate_fallback_suggestion(analysis).model_dump())


# ---------------------------------------------------------------------------
# Low-latency fallback (no retrieval, just conversation context)
# ---------------------------------------------------------------------------

def generate_fallback_suggestion(analysis: QueryAnalysis) -> AgentSuggestion:
    """
    Returns a safe, minimal suggestion when retrieval fails or store is empty.
    Used for graceful degradation.
    """
    return AgentSuggestion(
        intent=analysis.intent,
        urgency=analysis.urgency,
        summary=analysis.conversation_summary,
        suggested_responses=[
            SuggestedResponse(
                label="Acknowledge and assist",
                text=(
                    "Thank you for reaching out! I want to make sure I give you "
                    "accurate information. Could you please hold for a moment while "
                    "I look into this for you?"
                ),
            )
        ],
        key_facts=["Please consult atb.com or call 1-800-332-8383 for accurate details."],
        actions=["Verify customer identity", "Check relevant product pages at atb.com"],
        sources=[],
        confidence="low",
        escalate=analysis.requires_escalation,
        escalation_reason=analysis.escalation_reason,
        escalation_to="appropriate specialist" if analysis.requires_escalation else "",
    )
