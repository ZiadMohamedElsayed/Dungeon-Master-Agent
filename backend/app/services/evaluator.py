"""DM response evaluation via direct Gemini-judge calls (1-5 rubrics).

Replaces RAGAS (0.3.1 and even 0.4.3 hard-import
`langchain_community.chat_models.vertexai`, removed in
langchain-community 0.4.x — unfixable without shims).

Each metric is one async LLM call running a fixed 1-5 rubric over the
turn's query/answer/retrieved contexts (+ reference when provided).
`evaluate_turn` is @traceable so evaluations show up in LangSmith.
"""

import asyncio
import re

from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings

try:
    from langsmith import traceable
except ImportError:  # langsmith not installed (e.g. old docker image)

    def traceable(*dargs, **dkwargs):
        def wrap(fn):
            return fn

        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]
        return wrap


# ── Rubrics (name → score descriptions) ──────────────────────────────────────

RUBRICS: dict[str, dict[str, str]] = {
    "dm_relevance": {
        "1": "The response completely ignores the player's action or contradicts it.",
        "2": "The response barely acknowledges the player's action with little narrative build.",
        "3": "The response addresses the action but the narrative connection is weak or generic.",
        "4": "The response directly addresses the action with good narrative coherence.",
        "5": "The response masterfully builds upon the action in a rich, narratively compelling way.",
    },
    "lore_consistency": {
        "1": "The response directly contradicts multiple established lore or campaign facts.",
        "2": "The response contradicts at least one established fact.",
        "3": "The response is mostly consistent but introduces ambiguous or uncertain details.",
        "4": "The response is consistent with all retrieved lore and campaign history.",
        "5": "The response is perfectly consistent and enriches the established lore naturally.",
    },
    "narrative_quality": {
        "1": "The response is flat, has no sensory detail, and ends with no decision point.",
        "2": "The response has minimal immersion and a weak or missing decision point.",
        "3": "The response is moderately immersive with a clear but uninspired decision point.",
        "4": "The response is vivid and engaging with a clear, meaningful decision point.",
        "5": "The response is exceptionally immersive, uses rich sensory detail, and ends with a compelling, dramatically tense decision point.",
    },
    "context_precision": {
        "1": "None of the retrieved context is relevant to the reference answer.",
        "2": "Only a small fraction of the retrieved context is relevant.",
        "3": "About half of the retrieved context is relevant.",
        "4": "Most of the retrieved context is relevant.",
        "5": "All retrieved context is relevant to the reference answer.",
    },
    "context_recall": {
        "1": "The response covers none of the reference answer's key points.",
        "2": "The response covers only a small fraction of the reference's key points.",
        "3": "The response covers about half of the reference's key points.",
        "4": "The response covers most of the reference's key points.",
        "5": "The response covers all of the reference answer's key points.",
    },
}

BASE_METRICS = ["dm_relevance", "lore_consistency", "narrative_quality"]
REFERENCE_METRICS = ["context_precision", "context_recall"]

_MAX_CONTEXT_CHARS = 4000
_MAX_ANSWER_CHARS = 3000


def _build_contexts(lore_docs: list, campaign_docs: list) -> str:
    texts = [d.page_content for d in (lore_docs or []) + (campaign_docs or [])]
    joined = "\n\n---\n\n".join(texts)
    if len(joined) > _MAX_CONTEXT_CHARS:
        joined = joined[:_MAX_CONTEXT_CHARS] + "… [truncated]"
    return joined or "[no retrieved context]"


def _judge_prompt(
    metric: str, query: str, answer: str, contexts: str, reference: str | None
) -> str:
    rubric_lines = "\n".join(
        f"{score}: {desc}" for score, desc in RUBRICS[metric].items()
    )
    ref_block = f"\nReference answer:\n{reference}\n" if reference else ""
    return (
        f"You are a strict judge of AI Dungeon Master narration. "
        f"Score the DM response on '{metric}' with exactly one integer 1-5.\n\n"
        f"Rubric:\n{rubric_lines}\n\n"
        f"Player action:\n{query}\n\n"
        f"DM response:\n{answer[:_MAX_ANSWER_CHARS]}\n\n"
        f"Retrieved context:\n{contexts}\n"
        f"{ref_block}\n"
        f"Reply with ONLY the integer score (1, 2, 3, 4, or 5), nothing else."
    )


def _parse_score(text: str) -> int | None:
    match = re.search(r"\b([1-5])\b", text or "")
    return int(match.group(1)) if match else None


def _judge_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
    )


async def _score_metric(
    llm: ChatGoogleGenerativeAI,
    metric: str,
    query: str,
    answer: str,
    contexts: str,
    reference: str | None,
) -> tuple[str, int | None]:
    for attempt in range(2):
        try:
            raw = await llm.ainvoke(_judge_prompt(metric, query, answer, contexts, reference))
            content = raw.content if isinstance(raw.content, str) else str(raw.content)
            return metric, _parse_score(content)
        except Exception as e:
            retryable = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if retryable and attempt == 0:
                await asyncio.sleep(25)  # free-tier per-minute budget reset
                continue
            return metric, None
    return metric, None


@traceable(name="dm_evaluation")
async def evaluate_turn(
    query: str,
    answer: str,
    lore_docs: list,
    campaign_docs: list,
    reference: str | None = None,
) -> dict:
    """Score one turn on the base rubrics (+ reference metrics if given)."""
    llm = _judge_llm()
    contexts = _build_contexts(lore_docs, campaign_docs)
    metrics = BASE_METRICS + (REFERENCE_METRICS if reference else [])
    results = await asyncio.gather(*[
        _score_metric(llm, m, query, answer, contexts, reference) for m in metrics
    ])
    scores = dict(results)
    if all(v is None for v in scores.values()):
        raise RuntimeError("all judge calls failed (see server log / quota)")
    return scores


async def evaluate_without_reference(
    query: str, answer: str, lore_docs: list, campaign_docs: list
) -> dict:
    return await evaluate_turn(query, answer, lore_docs, campaign_docs)


async def evaluate_with_reference(
    query: str, answer: str, lore_docs: list, campaign_docs: list, reference: str
) -> dict:
    return await evaluate_turn(query, answer, lore_docs, campaign_docs, reference)
