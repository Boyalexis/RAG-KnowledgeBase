
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer

from app.rag.ingest import DB_DIR, MODEL_NAME, COLLECTION_NAME


QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def get_embedding_model():
    """缓存 BGE 模型，避免每次请求重新加载。"""
    return SentenceTransformer(MODEL_NAME, local_files_only=True)


@lru_cache(maxsize=1)
def get_collection():
    """连接已经建立的 ChromaDB 知识库。"""
    if not DB_DIR.exists():
        raise FileNotFoundError("向量数据库不存在，请先执行 ingest.py")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=None,
    )


def search_chunks(question: str, top_k: int = 4) -> list[dict]:
    """根据用户问题检索相关 Markdown 文本片段。"""

    collection = get_collection()
    count = collection.count()

    if count == 0:
        return []

    model = get_embedding_model()

    query_vector = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, count),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []

    for text, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "title": metadata["title"],
            "source": metadata["source"],  # 新增
            "text": text,
            "source_url": metadata["source_url"],
            "distance": float(distance),
        })

    return chunks


if __name__ == "__main__":
    question = "西红柿炒鸡蛋怎么做？"

    for chunk in search_chunks(question):
        print(chunk)
