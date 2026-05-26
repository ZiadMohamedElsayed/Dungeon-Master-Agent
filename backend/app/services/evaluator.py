import asyncio
import math
from functools import partial

from ragas import evaluate, EvaluationDataset, SingleTurnSample
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from app.core.config import settings


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
    return await _run_evaluation(dataset, [faithfulness, answer_relevancy])


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
        [faithfulness, answer_relevancy, context_precision, context_recall],
    )