from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.schema.embeddings import Embeddings
from functools import lru_cache
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.core.config import settings
import tempfile, os


class CachedEmbeddings(Embeddings):
    """LRU wrapper around the real embedding model.

    Identical texts embed once: saves one CPU pass per turn (the query is
    embedded separately by the lore and campaign retrievers) and makes
    repeated player inputs free. Same model + same text always gives the
    same vector, so entries never go stale.
    """

    def __init__(self, inner: Embeddings, query_size: int = 512, docs_size: int = 4096):
        self._inner = inner
        self._query = lru_cache(maxsize=query_size)(inner.embed_query)
        self._docs = lru_cache(maxsize=docs_size)(
            lambda texts: tuple(tuple(v) for v in inner.embed_documents(list(texts)))
        )

    def embed_query(self, text: str) -> list:
        return list(self._query(text))

    def embed_documents(self, texts: list) -> list:
        return [list(v) for v in self._docs(tuple(texts))]

    def cache_info(self) -> dict:
        return {
            "embed_query": self._query.cache_info()._asdict(),
            "embed_documents": self._docs.cache_info()._asdict(),
        }

    def cache_clear(self) -> None:
        self._query.cache_clear()
        self._docs.cache_clear()


embeddings = CachedEmbeddings(
    HuggingFaceEmbeddings(model_name=settings.embed_model),
    query_size=settings.embed_cache_query_size,
    docs_size=settings.embed_cache_docs_size,
)


# Epoch per collection, bumped on every write. Retrieval cache keys include
# the epoch, so uploads/deletes/clears (and turn auto-saves) invalidate it.
_collection_epochs = {"lore": 0, "campaign": 0}


def collection_epoch(prefix: str) -> int:
    return _collection_epochs.get(prefix, 0)


def bump_collection_epoch(prefix: str) -> int:
    _collection_epochs[prefix] = _collection_epochs.get(prefix, 0) + 1
    return _collection_epochs[prefix]

lore_vectorstore = Chroma(
    collection_name="lore_docs",
    embedding_function=embeddings,
    persist_directory=settings.resolved_lore_dir(),
)

campaign_vectorstore = Chroma(
    collection_name="campaign_docs",
    embedding_function=embeddings,
    persist_directory=settings.resolved_campaign_dir(),
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_file(file_bytes: bytes, filename: str, source_name: str) -> tuple:
    """Chunk a PDF, Markdown, or text file."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        suffix = ".pdf"
        loader_cls = PyPDFLoader
    elif lower.endswith(".md"):
        suffix = ".md"
        loader_cls = TextLoader  # markdown is plain-text compatible
    else:
        suffix = ".txt"
        loader_cls = TextLoader
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = loader_cls(tmp_path)
        docs = loader.load()
    finally:
        os.unlink(tmp_path)

    for doc in docs:
        doc.metadata["source"] = source_name
        doc.metadata["filename"] = filename

    chunks = splitter.split_documents(docs)
    return chunks, len(docs)


def get_lore_retriever(k: int = None):
    """Get a retriever for lore documents."""
    k = k or settings.top_k_retrieve
    return lore_vectorstore.as_retriever(search_kwargs={"k": k})


def get_campaign_retriever(k: int = None):
    """Get a retriever for campaign documents."""
    k = k or settings.top_k_retrieve
    return campaign_vectorstore.as_retriever(search_kwargs={"k": k})
