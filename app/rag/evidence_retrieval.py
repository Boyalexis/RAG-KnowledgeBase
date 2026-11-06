
from app.rag.search import get_collection


def expand_evidence(
    retrieved_docs: list[dict],
    max_docs: int = 3,
    max_chunks_per_doc: int = 2,
) -> list[dict]:
    """对已召回的文档补充有用的章节 Chunk。"""

    if max_docs < 1 or max_chunks_per_doc < 1:
        raise ValueError("参数必须大于 0")

    collection = get_collection()
    evidence = []

    for doc in retrieved_docs[:max_docs]:
        source = doc["source"]

        # 只查询已经被检索选中的菜谱
        result = collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )

        documents = result["documents"] or []
        metadatas = result["metadatas"] or []

        # 优先保留最初的语义检索结果
        selected = [doc.copy()]
        seen_texts = {doc["text"]}

        # 以操作、计算等章节作为补充证据
        priorities = [
            "操作",
            "计算",
            "必备原料和工具",
            "附加内容",
        ]

        for section in priorities:
            if len(selected) >= max_chunks_per_doc:
                break

            for text, metadata in zip(documents, metadatas):
                if len(selected) >= max_chunks_per_doc:
                    break

                if text in seen_texts:
                    continue

                if f"章节：{section}\n" not in text:
                    continue

                selected.append({
                    "title": metadata["title"],
                    "source": source,
                    "text": text,
                    "source_url": metadata["source_url"],
                    "origin": "document_expansion",
                })

                seen_texts.add(text)

        evidence.extend(selected)

    return evidence
