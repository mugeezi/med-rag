import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import config
import retriever
from database import connect_milvus

app = FastAPI(title="医疗问答RAG", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
def startup():
    connect_milvus()
    retriever.load_models()
    print("[App] 启动完成")

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = config.TOP_K
    with_answer: bool = True

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
    llm_answer: Optional[str] = None

def generate_answer(query, docs):
    context = "\n\n".join(
        f"参考{i+1}：问题：{d['question']}\n回答：{d['answer'][:500]}"
        for i, d in enumerate(docs[:5])
    )
    prompt = f"你是一名专业医疗助手。请根据以下参考资料回答用户问题。如果参考资料不足，请说明并建议就医。\n\n【参考资料】\n{context}\n\n【用户问题】{query}\n\n【回答】"
    try:
        resp = requests.post(
            config.LLM_API_URL,
            headers={"Authorization": f"Bearer {config.LLM_API_KEY}", "Content-Type": "application/json"},
            json={"model": config.LLM_MODEL, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3, "max_tokens": 800},
            timeout=config.LLM_TIMEOUT,
        )
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[LLM] 调用失败: {e}")
        return "（LLM生成失败，请查看下方参考文档）"

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询不能为空")
    docs = retriever.hybrid_search(req.query, top_k=req.top_k)
    llm_answer = generate_answer(req.query, docs) if req.with_answer and docs else None
    return SearchResponse(query=req.query, results=[DocResult(**d) for d in docs], llm_answer=llm_answer)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
