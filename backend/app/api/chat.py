from fastapi import APIRouter
from app.services.rag_chain import play_turn

router = APIRouter()


@router.post("/")
async def chat(query: str):
    return await play_turn(query)
