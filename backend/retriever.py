import re
from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import jieba
import config
from database import mysql_cursor, get_collection

_embedding_model = None
_rerank_model = None

def load_models():
    global _embedding_model, _rerank_model
    if _embedding_model is None:
        print(f"[Retriever] 加载 Embedding 模型: {config.EMBEDDING_MODEL}")
        _embedding_model = SentenceTransformer(config.EMBEDDING_MODEL)
    if _rerank_model is None:
        print(f"[Retriever] 加载 Rerank 模型: {config.RERANK_MODEL}")
        _rerank_model = CrossEncoder(config.RERANK_MODEL)

def get_embedding(text):
    load_models()
    return _embedding_model.encode(text, normalize_embeddings=True).tolist()

def tokenize(text):
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9]", " ", str(text))
    words = jieba.lcut(text)
    return [w.strip() for w in words if len(w.strip()) > 1]

def vector_search(query_vec, limit):
    col = get_collection()
    results = col.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": config.METRIC_TYPE, "params": {"nprobe": 16}},
        limit=limit,
        output_fields=["chunk_id", "doc_id"],
        partition_names=[config.PARTITION_NAME],
    )
    return [{"chunk_id": h.entity.get("chunk_id"), "doc_id": h.entity.get("doc_id")} for h in results[0]]

def bm25_search(query, limit):
    with mysql_cursor() as cur:
        cur.execute("SELECT chunk_id, doc_id, chunk_text FROM chunk ORDER BY chunk_id")
        rows = cur.fetchall()
    if not rows:
        return []
    corpus = [tokenize(r["chunk_text"]) for r in rows]
    scores = BM25Okapi(corpus).get_scores(tokenize(query))
    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)[:limit]
    return [{"chunk_id": r["chunk_id"], "doc_id": r["doc_id"]} for r, s in ranked if s > 0]

def rrf_fusion(vec_hits, bm25_hits):
    scores, doc_map = {}, {}
    for rank, h in enumerate(vec_hits, 1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + config.VECTOR_WEIGHT / (config.RRF_K + rank)
        doc_map[cid] = h["doc_id"]
    for rank, h in enumerate(bm25_hits, 1):
        cid = h["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + config.BM25_WEIGHT / (config.RRF_K + rank)
        doc_map[cid] = h["doc_id"]
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{"chunk_id": cid, "doc_id": doc_map[cid], "rrf_score": s} for cid, s in fused]

def fetch_chunk_texts(chunk_ids):
    if not chunk_ids:
        return {}
    ph = ",".join(["%s"] * len(chunk_ids))
    with mysql_cursor() as cur:
        cur.execute(f"SELECT chunk_id, chunk_text FROM chunk WHERE chunk_id IN ({ph})", chunk_ids)
        return {r["chunk_id"]: r["chunk_text"] for r in cur.fetchall()}

def fetch_docs(doc_ids):
    if not doc_ids:
        return {}
    ph = ",".join(["%s"] * len(doc_ids))
    with mysql_cursor() as cur:
        cur.execute(f"SELECT doc_id, question, answer, summary FROM doc WHERE doc_id IN ({ph})", doc_ids)
        return {r["doc_id"]: r for r in cur.fetchall()}

def hybrid_search(query, top_k=None):
    top_k = top_k or config.TOP_K
    limit = config.CANDIDATE_LIMIT
    query_vec = get_embedding(query)
    vec_hits  = vector_search(query_vec, limit)
    bm25_hits = bm25_search(query, limit)
    fused     = rrf_fusion(vec_hits, bm25_hits)[:limit]
    chunk_texts = fetch_chunk_texts([f["chunk_id"] for f in fused])
    for f in fused:
        f["chunk_text"] = chunk_texts.get(f["chunk_id"], "")
    load_models()
    pairs  = [[query, c["chunk_text"]] for c in fused if c["chunk_text"]]
    if not pairs:
        return []
    scores = _rerank_model.predict(pairs)
    for c, s in zip(fused, scores):
        c["rerank_score"] = float(s)
    fused = sorted(fused, key=lambda x: x.get("rerank_score", 0), reverse=True)
    doc_best = {}
    for c in fused:
        did = c["doc_id"]
        if did not in doc_best or c["rerank_score"] > doc_best[did]["rerank_score"]:
            doc_best[did] = c
    top_docs = sorted(doc_best.values(), key=lambda x: x["rerank_score"], reverse=True)[:top_k]
    docs = fetch_docs([d["doc_id"] for d in top_docs])
    return [{
        "doc_id":        d["doc_id"],
        "question":      docs.get(d["doc_id"], {}).get("question", ""),
        "answer":        docs.get(d["doc_id"], {}).get("answer", ""),
        "summary":       docs.get(d["doc_id"], {}).get("summary", ""),
        "score":         round(d["rerank_score"], 4),
        "chunk_snippet": d.get("chunk_text", "")[:300],
    } for d in top_docs]
