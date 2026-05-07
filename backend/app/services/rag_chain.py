import google.generativeai as genai
from app.core.vectorstore import get_lore_retriever, get_campaign_retriever
from app.services.reranker import rerank
from app.tools.mcp_tools import TOOLS, execute_tool
from app.core.config import settings
import asyncio

genai.configure(api_key=settings.gemini_api_key)
model = genai.GenerativeModel("gemini-2.5-flash")

SYSTEM_PROMPT = """"""


async def play_turn(query: str) -> dict:
    # Step 1: Retrieve
    lore_retriever = get_lore_retriever()
    campaign_retriever = get_campaign_retriever()

    raw_lore_docs = lore_retriever.get_relevant_documents(query)
    raw_campaign_docs = campaign_retriever.get_relevant_documents(query)

    # Step 2: Rerank
    raw_lore_docs = rerank(query, raw_lore_docs)
    raw_campaign_docs = rerank(query, raw_campaign_docs)

    # Step 3: Build context

    # Step 4: Build prompt and call LLM
