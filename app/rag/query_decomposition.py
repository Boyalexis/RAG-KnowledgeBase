
from app.rag.retrieval_strategies import search_chunks_diverse


def search_multi_query(
    sub_queries: list[str],
    top_k: int = 5,
) -> list[dict]:
    """
    对多个子查询分别检索，再按轮次合并结果。

    每个子查询保留独立的候选列表；
    合并时优先兼顾不同查询意图。
    """

    if not sub_queries or top_k < 1:
        raise ValueError("子查询不能为空，top_k 必须大于 0")

    all_results = []

    for query in sub_queries:
        results = search_chunks_diverse(
            question=query,
            top_k=top_k,
            fetch_k=30,
            max_chunks_per_doc=1,
        )
        all_results.append(results)

    merged = []
    seen_sources = set()

    # 轮流选取各子查询的检索结果
    for rank in range(top_k):
        for results in all_results:

            if rank >= len(results):
                continue

            chunk = results[rank]
            source = chunk["source"]

            if source in seen_sources:
                continue

            merged.append(chunk)
            seen_sources.add(source)

            if len(merged) == top_k:
                return merged

    return merged
