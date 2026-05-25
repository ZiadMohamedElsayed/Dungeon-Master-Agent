from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from app.core.config import settings
import tempfile, os

embeddings = HuggingFaceEmbeddings(model_name=settings.embed_model)

lore_vectorstore = Chroma(
    collection_name="lore_docs",
    embedding_function=embeddings,
    persist_directory=settings.lore_dp_persist_dir,
)

campaign_vectorstore = Chroma(
    collection_name="campaign_docs",
    embedding_function=embeddings,
    persist_directory=settings.campaign_dp_persist_dir,
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_file(file_bytes: bytes, filename: str, source_name: str) -> tuple:
    """Chunk a PDF or text file."""
    suffix = ".pdf" if filename.lower().endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        loader = PyPDFLoader(tmp_path) if suffix == ".pdf" else TextLoader(tmp_path)
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
