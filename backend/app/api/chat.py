from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.services.rag_chain import (
    play_turn,
    stream_turn,
    get_turn_count,
    get_campaign_history,
    get_recent_turns,
)
from app.core.vectorstore import campaign_vectorstore, chunk_file
import json

router = APIRouter()

class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    evaluate: bool = False
    reference: str | None = None
    stream: bool = False


@router.post("/")
async def chat(request: ChatRequest):
    if request.stream:
        async def gen():
            async for event in stream_turn(request.query):
                yield f"data: {json.dumps(event)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    try:
        return await play_turn(
            query=request.query,
            evaluate=request.evaluate,
            reference=request.reference,
        )
    except Exception as e:
        raise HTTPException(502, f"DM backend unavailable ({type(e).__name__}: {str(e)[:200]})")


@router.get("/turns/count")
async def turn_count():
    return {"turns": get_turn_count()}


@router.get("/turns/history")
async def turn_history(limit: int = 50):
    return {"turns": get_campaign_history(limit=limit)}


@router.get("/turns/recent")
async def recent_turns(limit: int | None = None):
    return {"turns": get_recent_turns(limit=limit)}


@router.get("/export")
async def export_campaign():
    return {"turns": get_campaign_history(limit=1000)}


@router.post("/import")
async def import_campaign(file: UploadFile = File(...)):
    content = await file.read()
    try:
        data = json.loads(content)
    except Exception:
        raise HTTPException(400, "Invalid JSON export file")
    turns = data.get("turns", data if isinstance(data, list) else [])
    if not isinstance(turns, list):
        raise HTTPException(400, "Invalid export format")
    try:
        from langchain.schema import Document
    except ImportError:
        from langchain_core.documents import Document
    docs = []
    for t in turns:
        text = t.get("content") or t.get("page_content") or ""
        if not text:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source": "campaign_history",
                "turn": t.get("turn", 0),
                "timestamp": t.get("timestamp", ""),
                "type": "turn_log",
            },
        ))
    if docs:
        # chunk long turns so they stay retrievable
        chunks, _ = chunk_file(
            "\n\n".join(d.page_content for d in docs).encode(),
            "campaign_import.txt",
            source_name="campaign_history",
        )
        # preserve turn numbers where possible
        campaign_vectorstore.add_documents(chunks if chunks else docs)
    return {"message": "Campaign imported", "turns_added": len(docs)}