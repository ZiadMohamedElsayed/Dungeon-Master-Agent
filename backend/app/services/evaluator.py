import asyncio
import math
from functools import partial

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, RubricsScore
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings

# ── Custom DM-appropriate metrics ────────────────────────────────────────────

dm_relevance = RubricsScore(
    name="dm_relevance",
    rubrics={
        "score1_description": "The response completely ignores the player's action or contradicts it.",
        "score2_description": "The response barely acknowledges the player's action with little narrative build.",
        "score3_description": "The response addresses the action but the narrative connection is weak or generic.",
        "score4_description": "The response directly addresses the action with good narrative coherence.",
        "score5_description": "The response masterfully builds upon the action in a rich, narratively compelling way.",
    }
)

lore_consistency = RubricsScore(
    name="lore_consistency",
    rubrics={
        "score1_description": "The response directly contradicts multiple established lore or campaign facts.",
        "score2_description": "The response contradicts at least one established fact.",
        "score3_description": "The response is mostly consistent but introduces ambiguous or uncertain details.",
        "score4_description": "The response is consistent with all retrieved lore and campaign history.",
        "score5_description": "The response is perfectly consistent and enriches the established lore naturally.",
    }
)

narrative_quality = RubricsScore(
    name="narrative_quality",
    rubrics={
        "score1_description": "The response is flat, has no sensory detail, and ends with no decision point.",
        "score2_description": "The response has minimal immersion and a weak or missing decision point.",
        "score3_description": "The response is moderately immersive with a clear but uninspired decision point.",
        "score4_description": "The response is vivid and engaging with a clear, meaningful decision point.",
        "score5_description": "The response is exceptionally immersive, uses rich sensory detail, and ends with a compelling, dramatically tense decision point.",
    }
)


def _build_contexts(lore_docs: list, campaign_docs: list) -> list:
    return [doc.page_content for doc in lore_docs + campaign_docs]


def _sanitize(record: dict) -> dict:
    """Replace nan/inf with None so the response is JSON-safe."""
    return {
        k: (None if isinstance(v, float) and not math.isfinite(v) else v)
        for k, v in record.items()
    }


def _run_evaluation_sync(dataset: EvaluationDataset, metrics: list) -> dict:
    """
    Fully blocking. Instantiates its own LLM/embeddings so that gRPC
    channels are bound to THIS thread's event loop, not FastAPI's.
    """
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
    ))
    embeddings = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.gemini_api_key,
    ))

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
    )
    record = result.to_pandas().to_dict(orient="records")[0]
    return _sanitize(record)


async def _run_evaluation(dataset: EvaluationDataset, metrics: list) -> dict:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(_run_evaluation_sync, dataset, metrics),
    )


async def evaluate_without_reference(
    query: str, answer: str, lore_docs: list, campaign_docs: list
) -> dict:
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=_build_contexts(lore_docs, campaign_docs),
        )
    ])
    return await _run_evaluation(
        dataset, 
        [dm_relevance, lore_consistency, narrative_quality])


async def evaluate_with_reference(      
    query: str, answer: str, lore_docs: list, campaign_docs: list, reference: str
) -> dict:
    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=query,
            response=answer,
            retrieved_contexts=_build_contexts(lore_docs, campaign_docs),
            reference=reference,
        )
    ])
    return await _run_evaluation(
        dataset,
        [dm_relevance, lore_consistency, narrative_quality,
         context_precision, context_recall],
    )