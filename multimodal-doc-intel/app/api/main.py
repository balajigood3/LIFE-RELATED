from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from app.rag.vectorstore import build_vectorstore, query
from app.utils.config import get_settings

app = FastAPI(title="Multimodal Industrial Document Intelligence", version="1.0")

class QueryRequest(BaseModel):
    question: str
    k: int = 4

@app.on_event("startup")
def startup():
    global store
    store = build_vectorstore()

@app.post("/query")
def ask(req: QueryRequest):
    docs = query(store, req.question, k=req.k)
    return {
        "answer_context": [ {"content": d.page_content[:800], "source": d.metadata.get("source")} for d in docs ],
        "note": "Pass context to an LLM for final answer generation in production."
    }

@app.get("/health")
def health():
    return {"status": "ok"}
