from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from app.core.config import settings

reranker = CrossEncoderReranker(
    model=HuggingFaceCrossEncoder(model_name=settings.rerank_model),
    top_n=settings.top_k_rerank,
)

def rerank(docs: list, query: str) -> list:
    return reranker.compress_documents(docs, query)