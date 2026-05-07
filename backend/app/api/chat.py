from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_chain import play_turn

router = APIRouter()


class ChatRequest(BaseModel):
    query: str


@router.post("/")
async def chat(request: ChatRequest):
    return await play_turn(request.query)
