
from app.rag.search import (
    QUERY_INSTRUCTION,
    get_collection,
    get_embedding_model,
)




def title_match_score(title: str, query: str) -> float:
    """
    带置信度门槛的 LCS 标题匹配。

    防止短标题因少量公共字符而获得过高分数。
    """
    title = title.strip()
    query = query.strip()

    if not title or not query:
        return 0.0

    if title in query:
        return 1.0

    previous = [0] * (len(query) + 1)

    for char in title:
        current = [0]

        for j, query_char in enumerate(query, 1):
            if char == query_char:
                current.append(previous[j - 1] + 1)
            else:
                current.append(
                    max(previous[j], current[-1])
                )

        previous = current

    lcs_length = previous[-1]
    coverage = lcs_length / len(title)

    # 拒绝只有少量字符巧合的标题匹配
    if lcs_length < 3:
        return 0.0

    if coverage < 0.75:
        return 0.0

    return coverage




def choose_documents(
    sub_queries: list[str],
    candidates: list[dict],
    max_docs: int = 2,
) -> list[tuple[str, dict]]:
    """根据子查询选择文档，优先保护唯一完整菜名匹配。"""

    if len(sub_queries) > max_docs:
        raise ValueError("max_docs 不能小于子查询数量")

    unique = {}
    for item in candidates:
        unique.setdefault(item["source"], item)

    docs = list(unique.values())
    if not docs:
        return []

    model = get_embedding_model()

    texts = (
        [QUERY_INSTRUCTION + q for q in sub_queries]
        + [doc["text"] for doc in docs]
    )

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_vectors = vectors[:len(sub_queries)]
    doc_vectors = vectors[len(sub_queries):]

    selected = []
    used_sources = set()

    for index, query in enumerate(sub_queries):

        # 1. 检查完整菜名是否直接出现在子查询中
        exact_sources = {
            doc["source"]
            for doc in docs
            if doc["title"] and doc["title"] in query
        }

        # 仅当完整匹配唯一时，给予优先权
        exact_source = (
            next(iter(exact_sources))
            if len(exact_sources) == 1
            else None
        )

        ranked = []

        for j, doc in enumerate(docs):
            semantic = float(
                query_vectors[index] @ doc_vectors[j]
            )

            lexical = title_match_score(
                doc["title"],
                query,
            )

            score = 0.45 * semantic + 0.55 * lexical

            # 唯一完整标题命中，获得优先级
            exact_priority = int(
                doc["source"] == exact_source
            )

            ranked.append(
                (exact_priority, score, j, doc)
            )

        # 排序顺序：
        # 1. 唯一完整菜名匹配
        # 2. 综合相关性评分
        # 3. 原始候选顺序（仅用于分数相等时）
        ranked.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                item[2],
            )
        )

        for priority, score, _, doc in ranked:
            if doc["source"] in used_sources:
                continue

            selected.append((query, doc))
            used_sources.add(doc["source"])

            print(
                f"子查询：{query}\n"
                f"选中文档：{doc['title']} "
                f"(score={score:.4f}, "
                f"exact={bool(priority)})"
            )
            break

    return selected



def rank_document_chunks(
    query: str,
    source: str,
) -> list[dict]:
    """获取文档所有 Chunk，并根据子查询重新排序。"""

    collection = get_collection()
    model = get_embedding_model()

    result = collection.get(
        where={"source": source},
        include=["documents", "metadatas"],
    )

    documents = result["documents"] or []
    metadatas = result["metadatas"] or []

    if not documents:
        return []

    query_vec = model.encode(
        [QUERY_INSTRUCTION + query],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    chunk_vecs = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    ranked = []

    action_question = any(
        word in query
        for word in ("怎么", "如何", "怎样", "处理", "制作", "更浓", "汤汁")
    )

    quantity_question = any(
        word in query
        for word in ("多少", "用量", "几克", "毫升", "几份")
    )

    ingredient_question = any(
        word in query
        for word in ("原料", "食材", "准备哪些")
    )

    for text, metadata, vec in zip(
        documents, metadatas, chunk_vecs
    ):
        section = (
            text.split("章节：", 1)[1].split("\n", 1)[0].strip()
            if "章节：" in text
            else "未知"
        )

        semantic = float(query_vec @ vec)
        bonus = 0.0

        if section == "操作" and action_question:
            bonus += 0.18

        if section == "计算" and quantity_question:
            bonus += 0.18

        if section == "必备原料和工具" and ingredient_question:
            bonus += 0.18

        ranked.append({
            "title": metadata["title"],
            "source": source,
            "source_url": metadata["source_url"],
            "text": text,
            "section": section,
            "score": semantic + bonus,
        })

    return sorted(
        ranked,
        key=lambda x: x["score"],
        reverse=True,
    )


def choose_complementary_chunks(
    ranked: list[dict],
    limit: int = 2,
) -> list[dict]:
    """选相关片段，同时优先补充不同章节的证据。"""

    if not ranked or limit <= 0:
        return []

    selected = [ranked[0]]

    if limit == 1:
        return selected

    first_section = ranked[0]["section"]

    # 操作和用量常常是互补证据
    preferred = {
        "操作": "计算",
        "计算": "操作",
    }.get(first_section)

    remaining = ranked[1:]

    if preferred:
        for item in remaining:
            if item["section"] == preferred:
                selected.append(item)
                break

    if len(selected) == 1:
        for item in remaining:
            if item["section"] != first_section:
                selected.append(item)
                break

    for item in remaining:
        if len(selected) >= limit:
            break
        if item not in selected:
            selected.append(item)

    return selected[:limit]


def select_evidence(
    sub_queries: list[str],
    candidates: list[dict],
    max_docs: int = 2,
    max_chunks_per_doc: int = 2,
    max_chars: int = 4000,
) -> list[dict]:
    """
    动态选择文档和 Chunk。

    max_chars 是字符预算，不是精确 Token 数。
    """

    if not sub_queries or not all(q.strip() for q in sub_queries):
        raise ValueError("子查询不能为空")
    if min(max_docs, max_chunks_per_doc, max_chars) <= 0:
        raise ValueError("预算参数必须大于 0")

    selected_docs = choose_documents(
        sub_queries,
        candidates,
        max_docs=max_docs,
    )

    groups = []

    for query, doc in selected_docs:
        ranked = rank_document_chunks(
            query,
            doc["source"],
        )

        group = choose_complementary_chunks(
            ranked,
            limit=max_chunks_per_doc,
        )

        groups.append(group)

    # 按轮次选择，每份文档优先获得一个证据位置
    evidence = []
    used_chars = 0

    for level in range(max_chunks_per_doc):
        for group in groups:
            if level >= len(group):
                continue

            item = group[level]
            length = len(item["text"])

            if used_chars + length > max_chars:
                continue

            evidence.append(item)
            used_chars += length

    return evidence
