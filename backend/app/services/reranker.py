from sentence_transformers import CrossEncoder
from langchain.schema import Document
from app.core.config import settings

_model = None


def get_reranker() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(settings.rerank_model, device="cpu")
    return _model


def rerank(query: str, docs: list, top_k: int = None) -> list:
    top_k = top_k or settings.top_k_rerank
    if not docs:
        return []
    model = get_reranker()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs, convert_to_numpy=True)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]