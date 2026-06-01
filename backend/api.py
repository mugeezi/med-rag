import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

import config
import retriever
from database import connect_milvus

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    connect_milvus()
    retriever.load_models()


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = config.TOP_K


class DocResult(BaseModel):
    doc_id: int
    question: str
    answer: str
    summary: str
    score: float
    chunk_snippet: str


class SearchResponse(BaseModel):
    query: str
    results: List[DocResult]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")

    docs = retriever.hybrid_search(req.query, top_k=req.top_k)

    return SearchResponse(
        query=req.query,
        results=[DocResult(**d) for d in docs],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
