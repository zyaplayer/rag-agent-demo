from fastapi import APIRouter
from pydantic import BaseModel

from backend.agent.agent import RAGAgent

router = APIRouter()
_agent = RAGAgent()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


@router.get("/")
def root():
    return {"msg": "RAG API is running"}


@router.post("/chat")
def chat(req: ChatRequest):
    answer, chunks = _agent.answer(req.query, top_k=req.top_k)
    return {
        "answer": answer,
        "references": [
            {"source": c.source, "score": c.score}
            for c in chunks
        ],
    }
    