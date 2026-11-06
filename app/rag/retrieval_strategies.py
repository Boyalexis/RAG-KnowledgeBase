
from collections import Counter

from app.rag.search import (
    QUERY_INSTRUCTION,
    get_collection,
    get_embedding_model,
)


def select_diverse_chunks(
    candidates: list[dict],
    top_k: int = 5,
    max_chunks_per_doc: int = 1,
) -> list[dict]:
    """从已排序的候选 Chunk 中选择不同文档。"""

    selected = []
    source_counts = Counter()
    # 用于记录每份菜谱已经选中了多少个 Chunk

    for chunk in candidates:
        source = chunk["source"]

        if source_counts[source] >= max_chunks_per_doc:
            continue

        selected.append(chunk)
        source_counts[source] += 1

        if len(selected) >= top_k:
            break

    return selected


def search_chunks_diverse(
    question: str,
    top_k: int = 5,
    fetch_k: int = 30,
    max_chunks_per_doc: int = 1,
) -> list[dict]:
    """
    先扩大候选池，再执行文档级去重。

    top_k: 最终返回 Chunk 数
    fetch_k: 向 ChromaDB 请求的候选 Chunk 数
    max_chunks_per_doc: 单份文档最多保留的 Chunk 数
    """

    if top_k < 1 or fetch_k < 1 or max_chunks_per_doc < 1:
        raise ValueError("检索参数必须大于 0")

    collection = get_collection()

    count = collection.count()
    if count == 0:
        return []

    model = get_embedding_model()

    query_vector = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
    )[0].tolist()

    candidate_count = min(
        count,
        max(fetch_k, top_k),
    )

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=candidate_count,
        include=["documents", "metadatas", "distances"],
    )

    candidates = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        candidates.append({
            "title": metadata["title"],
            "source": metadata["source"],
            "text": document,
            "source_url": metadata["source_url"],
            "distance": float(distance),
        })

    return select_diverse_chunks(
        candidates,
        top_k=top_k,
        max_chunks_per_doc=max_chunks_per_doc,
    )
