from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.core import get_pipeline


router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    top_k: int = 3


@router.get("/")
def root():
    return {"msg": "RAG API is running"}


@router.post("/chat")
def chat(req: ChatRequest):
    try:
        pipeline = get_pipeline()
        docs = pipeline.query(req.query, top_k=req.top_k)

        if not docs:
            return {
                "answer": "未检索到相关内容",
                "references": []
            }

        answer = "\n".join([d.page_content for d in docs])

        references = []
        for d in docs:
            references.append({
                "source": d.metadata.get("source", "unknown"),
                "score": d.metadata.get("score", None)
            })

        return {
            "answer": answer,
            "references": references,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
