import os
from dotenv import load_dotenv
load_dotenv()
# config.py —— 所有配置集中在这里，按需修改

from pathlib import Path

# ── MySQL ──────────────────────────────────────────
MYSQL_CONFIG = {
    "host":     "127.0.0.1",
    "port":     3306,
    "user":     "root",
    "password": "medrag123",
    "db":       "med_rag",
    "charset":  "utf8mb4",
}

# ── Milvus ────────────────────────────────────────
MILVUS_HOST     = "127.0.0.1"
MILVUS_PORT     = "19530"
COLLECTION_NAME = "med_rag_chunks"
VECTOR_DIM      = 1024
METRIC_TYPE     = "L2"
INDEX_TYPE      = "IVF_FLAT"
INDEX_NLIST     = 128

# ── 模型路径 ──────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
EMBEDDING_MODEL = str(BASE_DIR / "models" / "bge-large-zh-v1.5")
RERANK_MODEL    = str(BASE_DIR / "models" / "bge-reranker-large")

# ── 检索参数 ──────────────────────────────────────
TOP_K           = 8
CANDIDATE_LIMIT = 30
RRF_K           = 60
BM25_WEIGHT     = 0.7
VECTOR_WEIGHT   = 0.3

# ── LLM ──────────────────────────────────────────
LLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL   = "deepseek-ai/DeepSeek-V3"
LLM_TIMEOUT = 30

# ── 数据导入 ──────────────────────────────────────
IMPORT_LIMIT   = 200
PROJECT_ID     = 1
PARTITION_NAME = f"partition_project_{PROJECT_ID}"
