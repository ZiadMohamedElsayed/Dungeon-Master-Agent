from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.vectorstore import chunk_file, lore_vectorstore, campaign_vectorstore

router = APIRouter()

SUPPORTED_FORMATS = (".pdf", ".txt", ".md")


def register_document_routes(prefix: str, vectorstore):
    @router.post(f"/{prefix}/upload")
    async def upload_document(file: UploadFile = File(...)):
        # Check if already uploaded
        existing = lore_vectorstore._collection.get(
            where={"source": {"$eq": file.filename}},
            include=["metadatas"],
        )
        if existing["metadatas"]:
            raise HTTPException(409, f"'{file.filename}' already ingested. Delete it first or use a different filename.")
        
        # Check if format not supported
        if not file.filename.lower().endswith(SUPPORTED_FORMATS):
            raise HTTPException(
                400,
                f"Supported formats: {', '.join(f.upper().lstrip('.') for f in SUPPORTED_FORMATS)}",
            )
        content = await file.read()
        chunks, num_docs = chunk_file(content, file.filename, source_name=file.filename)
        vectorstore.add_documents(chunks)
        return {
            "message": "Ingested successfully",
            "chunks_added": len(chunks),
            "pages": num_docs,
        }

    @router.get(f"/{prefix}/list")
    async def list_documents():
        results = vectorstore._collection.get(include=["metadatas"])
        sources = list({m.get("source", "unknown") for m in results["metadatas"]})
        return {"documents": sources, "total_chunks": len(results["metadatas"])}

    @router.delete(f"/{prefix}/clear")
    async def clear_documents():
        vectorstore._collection.delete(where={"source": {"$ne": ""}})
        return {"message": "All documents cleared"}


register_document_routes("lore", lore_vectorstore)
register_document_routes("campaign", campaign_vectorstore)
