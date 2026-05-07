import google.generativeai as genai
from app.core.vectorstore import get_lore_retriever, get_campaign_retriever
from app.services.reranker import rerank
from app.core.config import settings
import asyncio

genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel(settings.llm_model)

SYSTEM_PROMPT = """You are an expert, immersive Dungeon Master running a tabletop RPG session.
You have access to two knowledge bases:

1. **World Lore** – The deep history, factions, geography, religions, magic systems, and lore of the world. Use this for world-building details, NPC backgrounds, location descriptions, and anything that exists in the established fiction.

2. **Campaign History** – The events, decisions, and consequences that have unfolded specifically in this campaign. Use this to maintain continuity with past sessions, remember player choices, and track ongoing plot threads.

Your responsibilities:
- Stay strictly consistent with retrieved lore and campaign history. Never contradict established facts.
- Clearly distinguish world-lore flavor from campaign-specific continuity in your narration.
- If the context does not cover something, improvise creatively but stay tonally consistent with the retrieved material.
- Narrate in vivid, second-person present tense ("You see...", "The guard snarls at you...").
- End each turn with a clear decision point or open question for the players.
- Never break character or acknowledge the RAG system.

Respond only as the Dungeon Master."""


def _format_docs(docs: list, header: str) -> str:
    """Format a list of LangChain Documents into a labeled context block."""
    if not docs:
        return f"[{header}: No relevant information found]\n"

    lines = [f"=== {header} ==="]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(lines)


def _build_prompt(query: str, lore_context: str, campaign_context: str) -> str:
    return f"""{SYSTEM_PROMPT}

---

{lore_context}

{campaign_context}

---

Player action / query:
{query}

Dungeon Master response:"""


async def play_turn(query: str) -> dict:
    """
    Full RAG pipeline:
      1. Retrieve from both vectorstores in parallel
      2. Rerank each result set independently
      3. Build formatted context blocks
      4. Construct prompt and call Gemini
      5. Return structured response
    """
    # Step 1: Retrieve from both stores concurrently
    lore_retriever = get_lore_retriever()
    campaign_retriever = get_campaign_retriever()

    raw_lore_docs, raw_campaign_docs = await asyncio.gather(
        asyncio.to_thread(lore_retriever.get_relevant_documents, query),
        asyncio.to_thread(campaign_retriever.get_relevant_documents, query),
    )

    # Step 2: Rerank each set independently (in parallel, both are CPU-bound)
    lore_docs, campaign_docs = await asyncio.gather(
        asyncio.to_thread(rerank, query, raw_lore_docs),
        asyncio.to_thread(rerank, query, raw_campaign_docs),
    )

    # Step 3: Build context blocks
    lore_context = _format_docs(lore_docs, "World Lore")
    campaign_context = _format_docs(campaign_docs, "Campaign History")

    # Step 4: Build prompt and call LLM
    prompt = _build_prompt(query, lore_context, campaign_context)
    response = await asyncio.to_thread(model.generate_content, prompt)
    answer = response.text.strip()

    # Step 5: Return structured payload
    return {
        "answer": answer,
        "sources": {
            "lore": [
                {"source": d.metadata.get("source"), "snippet": d.page_content[:200]}
                for d in lore_docs
            ],
            "campaign": [
                {"source": d.metadata.get("source"), "snippet": d.page_content[:200]}
                for d in campaign_docs
            ],
        },
    }
