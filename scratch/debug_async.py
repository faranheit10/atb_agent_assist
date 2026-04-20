import asyncio
import sys
import os

# Ensure we can import from the root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import ConversationTurn
from pipeline_v2.vector_store import KnowledgeStore
from pipeline_v2.retrieval import async_retrieve

async def main():
    print("Testing async_retrieve...")
    store = KnowledgeStore()
    conv = [ConversationTurn(role="customer", content="What is the monthly fee for the Generation Account?")]
    
    try:
        chunks, analysis = await async_retrieve(conv, store)
        print(f"Success! Found {len(chunks)} chunks.")
        print(f"Intent: {analysis.intent}")
        for c in chunks:
            print(f"- {c.unit.id}: {c.unit.title}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
