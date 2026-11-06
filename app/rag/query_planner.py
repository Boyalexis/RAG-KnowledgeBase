
import json
import logging
import os

from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI
from app.rag.llm_throttle import limited_chat_completion


logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """
你是 HowToCook 菜谱知识库的检索规划器。

你的任务不是回答用户，而是为向量数据库生成检索查询。

规则：
1. 如果用户只询问一道菜，即使包含食材、用量、
   步骤、保存方式等多个要求，也只保留一个查询。
2. 如果用户需要比较或分别了解多道菜，
   为每个独立的菜谱目标生成一个子查询。
3. 每个子查询都要保留对应的菜谱线索和具体需求。
4. 保留用户问题中的数量、时间、否定条件等关键信息。
5. 不得凭空添加用户没有提到的菜名、食材或做法。
6. 用户没有明确提到完整菜名时，保留其自然语言描述。
7. 最多返回三个查询。
8. 不要回答烹饪问题，也不要解释检索结果。

严格输出 JSON，不要输出 Markdown：

{
  "sub_queries": [
    "子查询1",
    "子查询2"
  ]
}
"""


@dataclass
class QueryPlan:
    sub_queries: list[str]
    mode: str
    error: str | None = None


@lru_cache(maxsize=1)
def get_planner_client() -> OpenAI:
    load_dotenv()

    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")

    if not api_key:
        raise RuntimeError("未配置 LLM_API_KEY")

    return OpenAI(
        api_key=api_key,
        base_url=base_url or None,
        timeout=30.0,
        max_retries=0,
    )


def parse_sub_queries(
    content: str,
    original_question: str,
) -> list[str]:
    """校验 LLM 输出，不允许静默截断子查询。"""

    content = content.strip()

    # 兼容少数模型输出的 ```json 包裹
    if content.startswith("```"):
        lines = content.splitlines()

        if len(lines) < 3 or not lines[-1].startswith("```"):
            raise ValueError("JSON 代码块不完整")

        content = "\n".join(lines[1:-1])

    data = json.loads(content)

    if not isinstance(data, dict):
        raise ValueError("输出必须是 JSON 对象")

    queries = data.get("sub_queries")

    if not isinstance(queries, list):
        raise ValueError("sub_queries 必须是数组")

    if not 1 <= len(queries) <= 3:
        raise ValueError("子查询数量必须为 1 到 3")

    cleaned = []

    for query in queries:
        if not isinstance(query, str):
            raise ValueError("子查询必须是字符串")

        query = query.strip()

        if not query or len(query) > 250:
            raise ValueError("子查询为空或过长")

        if query not in cleaned:
            cleaned.append(query)

    # 单查询时直接保留原始用户问题，
    # 防止 LLM 改写丢失否定条件或细节。
    if len(cleaned) == 1:
        return [original_question]

    return cleaned


def plan_queries(question: str) -> QueryPlan:
    """自动判断是否需要查询分解。"""

    question = question.strip()

    if not question:
        raise ValueError("用户问题不能为空")

    try:
        load_dotenv()
        model = os.getenv("LLM_MODEL")

        if not model:
            raise RuntimeError("未配置 LLM_MODEL")

        client = get_planner_client()

        response = limited_chat_completion(
            client,
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": question,
                },
            ],
        )

        content = response.choices[0].message.content

        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM 未返回有效文本")

        queries = parse_sub_queries(
            content,
            original_question=question,
        )

        return QueryPlan(
            sub_queries=queries,
            mode="multi" if len(queries) > 1 else "single",
        )

    except Exception as exc:
        # 查询规划失败不应使整个 RAG 无法检索。
        logger.warning(
            "Query Planner 失败，回退到原始问题：%s",
            exc,
        )

        return QueryPlan(
            sub_queries=[question],
            mode="fallback",
            error=f"{type(exc).__name__}: {exc}",
        )
