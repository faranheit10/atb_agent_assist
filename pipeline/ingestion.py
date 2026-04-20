"""
ATB Agent Assist — LLM-Driven KB Ingestion Engine
===================================================
Instead of static regex/heuristic chunking, this module calls Gemini to
semantically structure each raw KB document into atomic KnowledgeUnits:

    Raw .txt  →  Gemini structuring  →  [KnowledgeUnit, ...]

Each unit is self-contained, has rich metadata, and a one-line summary.
Those summaries become a separate embedding space for coarse (hierarchical)
retrieval, while the atomic_text fields form the fine-grained search layer.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Iterator, Optional

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from config import GEMINI_API_KEY, GENERATION_MODEL, KB_DIR, INGESTION_STATE_PATH, KnowledgeUnit

console = Console()
_client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Structuring prompt
# ---------------------------------------------------------------------------

_STRUCTURING_SYSTEM = """\
You are a banking knowledge structuring system for ATB Financial.
Your task is to decompose a raw knowledge base document into the smallest
meaningful, self-contained knowledge units — each answering exactly ONE
customer question or covering ONE specific fact.

Rules:
- Each unit's atomic_text must be fully self-contained (no references like
  "as described above" or "see section X").
- Do NOT summarise or omit details — preserve all specific numbers, rates,
  fees, product names, eligibility criteria, and conditions exactly as written.
- Assign customer_segment carefully: "student" for student products,
  "senior" for 59+ products, "newcomer" for newcomer products, etc.
- hierarchy_path uses " > " as separator, e.g. "chequing > unlimited > fees".
- keywords should include product names, synonyms, and key terms a customer
  might use when asking about this topic.
- Set requires_advisor_verification=true for anything involving current rates,
  approvals, credit decisions, or complex eligibility.

Respond ONLY with a valid JSON array. No markdown fences. No extra text.\
"""

_STRUCTURING_PROMPT_TEMPLATE = """\
Document filename: {filename}

Document content:
---
{content}
---

Convert this document into structured knowledge units following the schema:
[
  {{
    "id": "<product_category>_<product_slug>_<topic>_<3-digit-seq>",
    "title": "<concise title>",
    "product_category": "<chequing|savings|credit_cards|investing|mortgages|borrowing|customer_service|company_profile>",
    "product_name": "<specific product name>",
    "topic": "<fees|eligibility|rates|benefits|how_to|faq|comparison|limits|procedure|contact|general>",
    "customer_segment": "<general|student|senior|newcomer|professional|business|youth|all>",
    "content_type": "<definition|rule|rate|exception|faq|procedure|comparison|contact>",
    "atomic_text": "<complete, self-contained factual text>",
    "summary": "<one sentence summary>",
    "keywords": ["keyword1", "keyword2"],
    "hierarchy_path": "<category > product > topic>",
    "requires_advisor_verification": false
  }}
]

Produce as many units as needed — typically 8–25 per document.
"""


# ---------------------------------------------------------------------------
# Core structuring call
# ---------------------------------------------------------------------------

@retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    stop=stop_after_attempt(10),
    reraise=True,
)
def _structure_document(filename: str, content: str) -> list[KnowledgeUnit]:
    """Call Gemini to structure one KB file into atomic knowledge units."""
    prompt = _STRUCTURING_PROMPT_TEMPLATE.format(
        filename=filename, content=content
    )
    response = _client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config={"system_instruction": _STRUCTURING_SYSTEM},
    )

    raw_text = response.text.strip()

    # Strip any accidental markdown fences
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    data: list[dict] = json.loads(raw_text)

    units: list[KnowledgeUnit] = []
    for i, item in enumerate(data):
        # Ensure source file is set
        item["source_file"] = filename
        # Ensure ID uniqueness — suffix with file index if needed
        if not item.get("id"):
            item["id"] = f"unit_{filename}_{i:03d}"
        try:
            unit = KnowledgeUnit.model_validate(item)
            units.append(unit)
        except Exception as exc:
            console.print(
                f"[yellow]  ⚠ Skipping malformed unit #{i} from {filename}: {exc}[/yellow]"
            )
            continue

    return units


# ---------------------------------------------------------------------------
# State & Hashing
# ---------------------------------------------------------------------------

def calculate_file_hash(path: Path) -> str:
    """Calculate MD5 hash of file content for change detection."""
    if not path.exists():
        return ""
    return hashlib.md5(path.read_bytes()).hexdigest()


def load_ingestion_state() -> dict:
    """Load the mapping of filename -> {hash, last_ingested}."""
    if not INGESTION_STATE_PATH.exists():
        return {}
    try:
        return json.loads(INGESTION_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_ingestion_state(state: dict) -> None:
    """Save the ingestion state mapping."""
    INGESTION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    INGESTION_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_kb_files(kb_dir: Path = KB_DIR) -> Iterator[tuple[str, str]]:
    """Yield (filename, content) for each .txt file in the KB directory."""
    txt_files = sorted(kb_dir.glob("*.txt"))
    if not txt_files:
        return  # Graceful handle empty dir
    for path in txt_files:
        yield path.name, path.read_text(encoding="utf-8")


def ingest_single_file(filename: str, kb_dir: Path = KB_DIR) -> list[KnowledgeUnit]:
    """
    Structure a single file from the KB directory.
    Normally called after a new file is uploaded.
    """
    path = kb_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"File {filename} not found in {kb_dir}")
    
    console.print(f"[cyan]➜ Ingesting single file:[/cyan] {filename}")
    units = _structure_document(filename, path.read_text(encoding="utf-8"))
    
    # Update local state for this file
    state = load_ingestion_state()
    state[filename] = {
        "hash": calculate_file_hash(path),
        "last_ingested": time.time(),
        "unit_ids": [u.id for u in units]
    }
    save_ingestion_state(state)
    
    return units


def ingest_all_kb_files(
    kb_dir: Path = KB_DIR, force: bool = False
) -> list[KnowledgeUnit]:
    """
    Main ingestion entry point.
    
    Skips files that haven't changed (based on hash) unless 'force' is True.
    Returns KnowledgeUnits for all files (either freshly structured or skipped).
    """
    state = load_ingestion_state()
    all_units: dict[str, KnowledgeUnit] = {}
    
    # We need a list of files to process or skip
    txt_files = sorted(kb_dir.glob("*.txt"))
    
    console.print(f"\n[bold cyan]🧠 Ingestion Engine[/bold cyan] — checking {len(txt_files)} files\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("Processing...", total=len(txt_files))

        for path in txt_files:
            filename = path.name
            current_hash = calculate_file_hash(path)
            cached = state.get(filename, {})
            
            if not force and cached.get("hash") == current_hash:
                progress.update(task, description=f"[dim]Skipping (unchanged)[/dim] {filename}")
                progress.advance(task)
                # Note: We don't return the units here because ingest_all is often used 
                # to get NEW units to add to the store. 
                # If we want to return ALL units including cached ones, we'd need to 
                # store the unit objects in the state file too, which is bulky.
                # Standard practice: ingest_all returns ONLY the ones that actually 
                # were just structured, unless the caller specifically wants a full reload.
                continue

            progress.update(task, description=f"[cyan]Structuring[/cyan] {filename} …")
            try:
                content = path.read_text(encoding="utf-8")
                units = _structure_document(filename, content)
                for u in units:
                    all_units[u.id] = u
                
                # Update state
                state[filename] = {
                    "hash": current_hash,
                    "last_ingested": time.time(),
                    "unit_ids": [u.id for u in units]
                }
                
                progress.advance(task)
                console.print(f"  [green]✓[/green] {filename}  →  {len(units)} units")
            except Exception as exc:
                progress.advance(task)
                console.print(f"  [red]✗[/red] {filename}  →  FAILED: {exc}")
            
            if path != txt_files[-1]:
                time.sleep(2)

    save_ingestion_state(state)
    result = list(all_units.values())
    if result:
        console.print(f"\n[bold green]Ingestion update complete:[/bold green] {len(result)} new/changed units structured\n")
    else:
        console.print("\n[bold blue]No changes detected.[/bold blue] Everything is up to date.\n")
    
    return result
