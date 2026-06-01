import sys
from pathlib import Path
import pandas as pd
import pymysql
from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import config
from database import init_mysql, init_milvus

DATA_DIR = Path(__file__).parent

def split_text(text, max_len=400):
    text = str(text).strip()
    chunks = []
    while text:
        chunks.append(text[:max_len])
        text = text[max_len:]
    return chunks or [text]

def load_data(limit):
    q_path = DATA_DIR / "question.csv"
    a_path = DATA_DIR / "answer.csv"
    if not q_path.exists() or not a_path.exists():
        print(f"[ERROR] 找不到 answer.csv / question.csv，请放到 {DATA_DIR}")
        sys.exit(1)
    questions = pd.read_csv(q_path)
    answers   = pd.read_csv(a_path)

    # 每个问题取第一条回答
    first = answers.drop_duplicates("question_id")

    merged = questions.merge(
        first[["question_id", "content"]].rename(columns={"content": "answer"}),
        on="question_id", how="left"
    )
    merged = merged.dropna(subset=["answer", "content"]).rename(columns={"content": "question"})
    return merged.head(limit)

def main():
    print("[Import] 初始化数据库...")
    init_mysql()
    col = init_milvus()
    col.load()

    print(f"[Import] 加载模型: {config.EMBEDDING_MODEL}")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print(f"[Import] 读取数据（最多 {config.IMPORT_LIMIT} 条）...")
    data = load_data(config.IMPORT_LIMIT)
    print(f"[Import] 共 {len(data)} 条，开始导入...")

    conn = pymysql.connect(**config.MYSQL_CONFIG)
    BATCH = 64
    cid_buf, did_buf, vec_buf = [], [], []

    def flush():
        if cid_buf:
            col.insert([cid_buf, did_buf, vec_buf], partition_name=config.PARTITION_NAME)
            cid_buf.clear(); did_buf.clear(); vec_buf.clear()

    skipped = 0
    for _, row in tqdm(data.iterrows(), total=len(data)):
        q = str(row["question"]).strip()
        a = str(row["answer"]).strip()
        if not q or not a:
            skipped += 1
            continue
        with conn.cursor() as cur:
            cur.execute("INSERT INTO doc (question, answer, summary) VALUES (%s, %s, %s)", (q, a, q[:250]))
            conn.commit()
            doc_id = cur.lastrowid
        chunks = split_text(a)
        all_chunks = [f"问题：{q}\n回答：{chunks[0]}"] + chunks[1:]
        for seq, text in enumerate(all_chunks):
            with conn.cursor() as cur:
                cur.execute("INSERT INTO chunk (doc_id, chunk_text, chunk_seq) VALUES (%s,%s,%s)", (doc_id, text, seq))
                conn.commit()
                chunk_id = cur.lastrowid
            vec = model.encode(text, normalize_embeddings=True).tolist()
            cid_buf.append(chunk_id); did_buf.append(doc_id); vec_buf.append(vec)
            if len(cid_buf) >= BATCH:
                flush()

    flush()
    col.flush()
    conn.close()
    print(f"\n[Import] 完成！成功 {len(data)-skipped} 条，跳过 {skipped} 条")

if __name__ == "__main__":
    main()
