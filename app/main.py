
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag.search import search_chunks
from app.rag.generator import generate_answer
from app.rag.rag_pipeline import answer_question


logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG-KnowledgeBase",
    description="基于 HowToCook 菜谱的 RAG 智能问答系统",
    version="0.3.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=500,
        description="用户提出的菜谱问题",
    )
    top_k: int = Field(default=4, ge=1, le=8)


class Source(BaseModel):
    index: int
    title: str
    source_url: str
    distance: float
    excerpt: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "RAG-KnowledgeBase",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="问题不能为空",
        )

    # 第一步：检索相关菜谱
    try:
        chunks = search_chunks(question, request.top_k)
    except Exception as exc:
        logger.exception("知识库检索失败")
        raise HTTPException(
            status_code=503,
            detail="知识库检索失败，请检查 ChromaDB 和 Embedding 模型",
        ) from exc

    if not chunks:
        return AskResponse(
            question=question,
            answer="当前知识库没有可用的菜谱资料。",
            sources=[],
        )

    # 第二步：LLM 生成答案
    try:
        answer = generate_answer(question, chunks)
    except ValueError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("LLM 调用失败")
        raise HTTPException(
            status_code=502,
            detail="大模型调用失败，请检查 API Key、模型配置和网络",
        ) from exc

    # 第三步：返回答案和检索参考资料
    sources = [
        Source(
            index=index,
            title=chunk["title"],
            source_url=chunk["source_url"],
            distance=chunk["distance"],
            excerpt=chunk["text"],
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    return AskResponse(
        question=question,
        answer=answer,
        sources=sources,
    )

@app.post("/ask-v2")
def ask_v2(request: AskRequest):
    return answer_question(request.question)

