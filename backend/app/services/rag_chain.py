from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain.schema import Document
from app.core.vectorstore import get_lore_retriever, get_campaign_retriever, campaign_vectorstore
from app.services.reranker import rerank
from app.services.evaluator import evaluate_with_reference, evaluate_without_reference
from app.core.config import settings
from datetime import datetime

llm = ChatGoogleGenerativeAI(model=settings.llm_model,
                             google_api_key=settings.gemini_api_key)

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

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{lore_context}\n\n{campaign_context}\n\nPlayer action: {query}")
])

def _format_docs(docs: list, header: str) -> str:
    if not docs:
        return f"[{header}: No relevant information found]\n"
    lines = [f"=== {header} ==="]
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        lines.append(f"[{i}] (source: {source})\n{doc.page_content.strip()}")
    return "\n\n".join(lines)

def _save_turn_to_campaign(player_action: str, dm_response: str, turn: int):
    """Persist the current turn as a document in the campaign vectorstore."""
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    content = f"[Turn {turn} | {timestamp}]\nPlayer: {player_action}\nDM: {dm_response}"
    doc = Document(
        page_content=content,
        metadata={
            "source": "campaign_history",
            "turn": turn,
            "timestamp": timestamp,
            "type": "turn_log",
        },
    )
    campaign_vectorstore.add_documents([doc])


def _get_turn_count() -> int:
    """Count existing turn logs to number the next turn."""
    try:
        results = campaign_vectorstore._collection.get(
            where={"type": {"$eq": "turn_log"}},
            include=["metadatas"],
        )
        return len(results["metadatas"]) + 1
    except Exception:
        return 1

retrieval_step = RunnableParallel(
    lore_raw = RunnableLambda(lambda q: get_lore_retriever().invoke(q)),
    campaign_raw = RunnableLambda(lambda q: get_campaign_retriever().invoke(q)),
    query=RunnablePassthrough()
)

def _rerank_and_format(inputs: dict) -> dict:
    query = inputs["query"]
    lore_docs = rerank(inputs["lore_raw"], query)
    campaign_docs = rerank(inputs["campaign_raw"], query)
    return {
        "query": query,
        "lore_docs": lore_docs,
        "campaign_docs": campaign_docs,
        "lore_context": _format_docs(lore_docs, "World Lore"),
        "campaign_context": _format_docs(campaign_docs, "Campaign History"),
    }

dm_chain = (
    retrieval_step
    | RunnableLambda(_rerank_and_format)
    | RunnablePassthrough.assign(answer=(
        prompt | llm | StrOutputParser()
    ))
)

async def play_turn(query: str, evaluate: bool = False, reference: str | None = None) -> dict:
    result = await dm_chain.ainvoke(query)

    answer = result["answer"]
    lore_docs = result["lore_docs"]
    campaign_docs = result["campaign_docs"]

    turn_number = _get_turn_count()
    _save_turn_to_campaign(query, answer, turn_number)

    response = {
        "turn": turn_number,
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

    if evaluate:
        if reference:
            response["evaluation"] = await evaluate_with_reference(
                query, answer, lore_docs, campaign_docs, reference)
        else:
            response["evaluation"] = await evaluate_without_reference(
                query, answer, lore_docs, campaign_docs)

    return response
