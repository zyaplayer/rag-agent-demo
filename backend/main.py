from fastapi import FastAPI
from backend.api.router import router
app = FastAPI(
    title="RAG Agent API",
    version="1.0.0",
)
app.include_router(router, prefix="/api", tags=["rag"])