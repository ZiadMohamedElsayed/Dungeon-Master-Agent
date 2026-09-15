from langchain_google_genai import ChatGoogleGenerativeAI
try:
    from langchain.prompts import ChatPromptTemplate
    from langchain.schema.output_parser import StrOutputParser
    from langchain.schema.runnable import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain.schema import Document
except ImportError:  # langchain >= 1.x moved to langchain_core
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
    from langchain_core.documents import Document
from app.core.vectorstore import get_lore_retriever, get_campaign_retriever, campaign_vectorstore
from app.services.reranker import rerank
from app.services.dice import roll_dice
from app.core.config import settings
from datetime import datetime

llm = ChatGoogleGenerativeAI(model=settings.llm_model,
                             google_api_key=settings.gemini_api_key)
llm_with_tools = llm.bind_tools([roll_dice])

DICE_INSTRUCTIONS = """Dice (mandatory): whenever the player attempts an attack, skill check, saving throw, or any action with a chance-based outcome, you MUST call the `roll_dice` tool BEFORE narrating the outcome — never decide success/failure yourself. Pick the fitting notation (attacks/ability checks: '1d20'; weapon damage: e.g. '1d8+2'; pick locks/sneak: '1d20+modifier'). Narrate the outcome strictly from the real rolled total: high roll = success, low roll = failure or complication. State the roll naturally in character (e.g. 'The blade bites deep — a clean strike!'). Never reveal tool-call mechanics in character."""

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

{DICE_INSTRUCTIONS}

Respond only as the Dungeon Master.""".replace("{DICE_INSTRUCTIONS}", DICE_INSTRUCTIONS)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{lore_context}\n\n{campaign_context}\n\nPlayer action: {query}\n\n(Reminder: if this action could succeed or fail — attack, check, save, sneak, persuade — call the roll_dice tool FIRST and narrate strictly from the real result. Never narrate an outcome before rolling.)")
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

_TOOLS_BY_NAME = {"roll_dice": roll_dice}

def _text_of(content) -> str:
    """Extract plain text from an LLM response content (str or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict):
                if b.get("type") in ("text", None) and b.get("text"):
                    parts.append(b["text"])
            else:
                t = getattr(b, "text", "")
                if t:
                    parts.append(t)
        return "".join(parts)
    return str(content)


async def _generate_with_tools(formatted: dict, max_iters: int = 3) -> str:
    """Run the DM LLM with tool calls (dice). Loops until no more tool calls."""
    try:
        from langchain_core.messages import ToolMessage
    except ImportError:
        from langchain.schema import ToolMessage  # type: ignore
    messages = prompt.format_messages(**{k: formatted[k] for k in ("lore_context", "campaign_context", "query") if k in formatted})
    for _ in range(max_iters + 1):
        response = await llm_with_tools.ainvoke(messages)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return _text_of(getattr(response, "content", ""))
        import logging
        logging.getLogger("uvicorn.error").info(
            "DM tool calls: %s", [(c.get("name"), c.get("args")) for c in tool_calls]
        )
        messages.append(response)
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args", {}) or {}
            call_id = call.get("id", "")
            fn = _TOOLS_BY_NAME.get(name)
            if fn is None:
                messages.append(ToolMessage(content=f"Unknown tool '{name}'.", tool_call_id=call_id))
                continue
            try:
                result = fn.invoke(args) if hasattr(fn, "invoke") else fn(**args)
            except Exception as e:
                result = f"Tool error: {e}"
            messages.append(ToolMessage(content=str(result), tool_call_id=call_id))
    # fallback: final plain answer if tools kept looping
    final = await llm.ainvoke(messages)
    return _text_of(getattr(final, "content", ""))

async def play_turn(query: str, evaluate: bool = False, reference: str | None = None) -> dict:
    retrieved = await retrieval_step.ainvoke(query)
    formatted = _rerank_and_format(retrieved)
    answer = await _generate_with_tools(formatted)

    lore_docs = formatted["lore_docs"]
    campaign_docs = formatted["campaign_docs"]

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
