from fastapi import APIRouter, UploadFile, File, HTTPException
from app.core.vectorstore import chunk_file, lore_vectorstore, campaign_vectorstore

router = APIRouter()

SUPPORTED_FORMATS = (".pdf", ".txt", ".md")


def register_document_routes(prefix: str, vectorstore):
    @router.post(f"/{prefix}/upload")
    async def upload_document(file: UploadFile = File(...)):
        # Check if format not supported
        if not file.filename or not file.filename.lower().endswith(SUPPORTED_FORMATS):
            raise HTTPException(
                400,
                f"Supported formats: {', '.join(f.upper().lstrip('.') for f in SUPPORTED_FORMATS)}",
            )
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(400, "Empty file")
        if len(content) > 10 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 10 MB)")
        # Check if already uploaded (use this store, not always lore)
        existing = vectorstore._collection.get(
            where={"source": {"$eq": file.filename}},
            include=["metadatas"],
        )
        if existing["metadatas"]:
            raise HTTPException(409, f"'{file.filename}' already ingested. Delete it first or use a different filename.")
        
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
        metadatas = results.get("metadatas") or []
        counts: dict[str, int] = {}
        for m in metadatas:
            src = (m or {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        documents = [{"name": k, "chunks": v} for k, v in sorted(counts.items())]
        # Backwards-compat: keep flat names list + total
        return {
            "documents": [d["name"] for d in documents],
            "details": documents,
            "total_chunks": len(metadatas),
        }

    @router.delete(f"/{prefix}/clear")
    async def clear_documents():
        vectorstore._collection.delete(where={"source": {"$ne": ""}})
        return {"message": "All documents cleared"}

    @router.delete(f"/{prefix}/{{filename}}")
    async def delete_document(filename: str):
        existing = vectorstore._collection.get(
            where={"source": {"$eq": filename}},
            include=["metadatas"],
        )
        if not existing["metadatas"]:
            raise HTTPException(404, f"'{filename}' not found")
        vectorstore._collection.delete(where={"source": {"$eq": filename}})
        return {"message": f"'{filename}' deleted"}


register_document_routes("lore", lore_vectorstore)
register_document_routes("campaign", campaign_vectorstore)
