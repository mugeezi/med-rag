import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from pymilvus import connections, Collection, utility
import config

def get_mysql_conn():
    return pymysql.connect(**config.MYSQL_CONFIG, cursorclass=DictCursor)

@contextmanager
def mysql_cursor():
    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_mysql():
    conn = get_mysql_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS doc (
                    doc_id      BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    question    TEXT            NOT NULL,
                    answer      TEXT            NOT NULL,
                    summary     VARCHAR(300)    NOT NULL,
                    create_time DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (doc_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS chunk (
                    chunk_id    BIGINT UNSIGNED NOT NULL AUTO_INCREMENT,
                    doc_id      BIGINT UNSIGNED NOT NULL,
                    chunk_text  TEXT            NOT NULL,
                    chunk_seq   INT             NOT NULL DEFAULT 0,
                    create_time DATETIME        NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (chunk_id),
                    KEY idx_doc (doc_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
        conn.commit()
        print("[DB] MySQL 表初始化完成")
    finally:
        conn.close()

def connect_milvus():
    connections.connect(alias="default", host=config.MILVUS_HOST, port=config.MILVUS_PORT)

def get_collection():
    col = Collection(config.COLLECTION_NAME)
    col.load()
    return col

def init_milvus():
    from pymilvus import CollectionSchema, FieldSchema, DataType
    connect_milvus()
    if not utility.has_collection(config.COLLECTION_NAME):
        fields = [
            FieldSchema("id",       DataType.INT64,        is_primary=True, auto_id=True),
            FieldSchema("chunk_id", DataType.INT64),
            FieldSchema("doc_id",   DataType.INT64),
            FieldSchema("vector",   DataType.FLOAT_VECTOR, dim=config.VECTOR_DIM),
        ]
        schema = CollectionSchema(fields, description="医疗问答RAG")
        col = Collection(config.COLLECTION_NAME, schema)
        col.create_index("vector", {
            "index_type": config.INDEX_TYPE,
            "metric_type": config.METRIC_TYPE,
            "params": {"nlist": config.INDEX_NLIST},
        })
        print(f"[DB] 已创建 Collection: {config.COLLECTION_NAME}")
    else:
        col = Collection(config.COLLECTION_NAME)
    if not col.has_partition(config.PARTITION_NAME):
        col.create_partition(config.PARTITION_NAME)
        print(f"[DB] 已创建 Partition: {config.PARTITION_NAME}")
    print("[DB] Milvus 初始化完成")
    return col
