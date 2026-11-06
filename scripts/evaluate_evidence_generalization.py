"""Lesson 8: evidence-level generalization evaluation for RAG-KnowledgeBase.
Place under scripts/evaluate_evidence_generalization.py and run with `uv run python -m scripts.evaluate_evidence_generalization`.
Requires existing app.rag.search, app.rag.query_decomposition and app.rag.evidence_selector.
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from app.rag.search import search_chunks
from app.rag.query_decomposition import search_multi_query
from app.rag.evidence_selector import select_evidence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/evaluation/questions_generalization.jsonl"
DEFAULT_PLAN = ROOT / "data/evaluation/query_plan.jsonl"
DEFAULT_REPORT = ROOT / "data/evaluation/reports/evidence_generalization.jsonl"
DEFAULT_REVIEWS = ROOT / "data/evaluation/reports/evidence_review_template.jsonl"


def normalized(path):
    return str(path).replace("\\", "/").removeprefix("./")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as stream:
        for i, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path} 第 {i} 行不是有效 JSON") from exc


def section_of(item):
    if item.get("section"):
        return str(item["section"]).strip()
    match = re.search(r"(?m)^章节：([^\r\n]+)", item.get("text", ""))
    return match.group(1).strip() if match else "未知"


def normalize_evidence(chunks):
    return [
        {
            "evidence_id": i,
            "source": normalized(chunk["source"]),
            "title": chunk.get("title", ""),
            "section": section_of(chunk),
            "text": chunk["text"],
            "score": chunk.get("score"),
            "distance": chunk.get("distance"),
        }
        for i, chunk in enumerate(chunks, 1)
    ]


def requirement_matches(item, requirement):
    if item["source"] != normalized(requirement["doc"]):
        return False
    if item["section"] != requirement["section"]:
        return False
    sub = requirement.get("subsection")
    return sub is None or sub in item["text"]


def score_structure(case, evidence):
    """Structural proxy only: does NOT establish factual entailment."""
    relevant = {normalized(path) for path in case["relevant_docs"]}
    sources = {item["source"] for item in evidence}
    requirements = case["evidence_requirements"]
    n_facts = len(case["reference_facts"])
    possible_indices = set()
    matched_requirements = []
    for req in requirements:
        matched = any(requirement_matches(item, req) for item in evidence)
        matched_requirements.append(matched)
        if matched:
            possible_indices.update(req["fact_indices"])

    matching_chunk_count = sum(
        any(requirement_matches(item, req) for req in requirements)
        for item in evidence
    )
    return {
        "doc_recall": len(relevant & sources) / len(relevant),
        "required_section_coverage": (
            sum(matched_requirements) / len(requirements)
        ),
        "potential_fact_coverage_proxy": (
            len(possible_indices) / n_facts
        ),
        "required_section_chunk_precision_proxy": (
            matching_chunk_count / len(evidence) if evidence else 0.0
        ),
        "n_chunks": len(evidence),
        "context_chars": sum(len(x["text"]) for x in evidence),
        "matched_requirement_flags": matched_requirements,
    }


def validate_case(case):
    required = ("id", "question", "question_type", "relevant_docs",
                "reference_facts", "evidence_requirements", "reviewed")
    for key in required:
        if key not in case:
            raise ValueError(f"{case.get('id', '<unknown>')} 缺少字段 {key}")
    if not case["reviewed"]:
        return False
    if not case["relevant_docs"] or not case["reference_facts"]:
        raise ValueError(f"{case['id']} 缺少正确文档或参考事实")
    if not case["evidence_requirements"]:
        raise ValueError(f"{case['id']} 缺少 evidence_requirements")
    n = len(case["reference_facts"])
    for req in case["evidence_requirements"]:
        if normalized(req["doc"]) not in {normalized(p) for p in case["relevant_docs"]}:
            raise ValueError(f"{case['id']}: evidence doc 不在 relevant_docs 中")
        if not req.get("section") or not req.get("fact_indices"):
            raise ValueError(f"{case['id']}: 不完整的 evidence requirement")
        if any(not isinstance(i, int) or i < 0 or i >= n for i in req["fact_indices"]):
            raise ValueError(f"{case['id']}: fact_indices 越界")
    covered_indices = {idx for req in case["evidence_requirements"] for idx in req["fact_indices"]}
    if covered_indices != set(range(n)):
        raise ValueError(f"{case['id']}: evidence_requirements 未完整覆盖 reference_facts")
    return True


def check_commit(cases, allow_mismatch):
    declared = {c.get("source_commit") for c in cases if c.get("source_commit")}
    if len(declared) != 1:
        print("警告：数据集未声明唯一 source_commit，不能验证菜谱版本")
        return
    repo = ROOT / "data/sources/HowToCook"
    try:
        local = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("警告：无法读取本地 HowToCook Commit，请自行核对菜谱版本")
        return
    expected = next(iter(declared))
    print(f"菜谱版本：数据集={expected[:12]} / 当前={local[:12]}")
    if local != expected and not allow_mismatch:
        raise RuntimeError(
            "知识库源文件 Git Commit 与评测集不一致。"
            "确认已重新建索引后可带 --allow-commit-mismatch 运行探索实验，"
            "但不要将结果与锁定版本的基线直接比较。"
        )


def run(args):
    cases = [c for c in read_jsonl(args.dataset) if validate_case(c)]
    if not cases:
        raise ValueError("没有 reviewed=true 的有效评测案例")
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("评测案例 ID 重复")
    plans = {p["id"]: p["sub_queries"] for p in read_jsonl(args.plan)}
    for c in cases:
        qs = plans.get(c["id"])
        if not isinstance(qs, list) or not qs or any(not isinstance(q, str) or not q.strip() for q in qs):
            raise ValueError(f"{c['id']} 缺少非空 sub_queries")
        if len(qs) > args.max_docs:
            raise ValueError(f"{c['id']}: 子查询数量超过 max_docs")
    check_commit(cases, args.allow_commit_mismatch)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    records = []
    review_rows = []
    for i, case in enumerate(cases, 1):
        q = case["question"]
        queries = plans[case["id"]]
        print(f"[{i}/{len(cases)}] {case['id']} ({case['question_type']})")
        try:
            # True original-question baseline: 5 nearest chunks.
            baseline = normalize_evidence(search_chunks(q, top_k=args.top_k))
            # No ground truth is given to the retrieval pipeline.
            candidates = search_multi_query(queries, top_k=args.candidate_docs)
            dynamic = normalize_evidence(select_evidence(
                sub_queries=queries, candidates=candidates,
                max_docs=args.max_docs,
                max_chunks_per_doc=args.max_chunks_per_doc,
                max_chars=args.max_chars,
            ))
            err = None
        except Exception as exc:
            baseline, dynamic = [], []
            err = f"{type(exc).__name__}: {exc}"
            print("  ERROR:", err)

        record = {
            "id": case["id"], "question": q,
            "question_type": case["question_type"],
            "sub_queries": queries, "relevant_docs": case["relevant_docs"],
            "reference_facts": case["reference_facts"],
            "evidence_requirements": case["evidence_requirements"],
            "source_commit": case.get("source_commit"),
            "error": err,
            "baseline": {
                "metrics": score_structure(case, baseline) if not err else None,
                "evidence": baseline,
            },
            "dynamic": {
                "metrics": score_structure(case, dynamic) if not err else None,
                "evidence": dynamic,
            },
        }
        records.append(record)
        review_rows.append({
            "id": case["id"], "question": q,
            "fact_reviews": {
                strategy: [
                    {
                        "fact_index": j, "fact": fact,
                        "supported": None,
                        "supporting_evidence_ids": [],
                    }
                    for j, fact in enumerate(case["reference_facts"])
                ]
                for strategy in ("baseline", "dynamic")
            },
        })

    with args.report.open("w", encoding="utf-8") as stream:
        for rec in records:
            stream.write(json.dumps(rec, ensure_ascii=False) + "\n")
    if args.reviews.exists():
        print(f"人工标注模板已存在，不覆盖：{args.reviews}")
    else:
        with args.reviews.open("w", encoding="utf-8") as stream:
            for rec in review_rows:
                stream.write(json.dumps(rec, ensure_ascii=False) + "\n")

    valid = [r for r in records if r["error"] is None]
    print(f"\n评测成功：{len(valid)}/{len(records)}")
    for strategy in ("baseline", "dynamic"):
        if not valid:
            break
        metrics = [r[strategy]["metrics"] for r in valid]
        print(f"\n{strategy}（注意：以下是结构性代理指标，不是事实覆盖率）：")
        for key in ("doc_recall", "required_section_coverage", "potential_fact_coverage_proxy", "required_section_chunk_precision_proxy", "n_chunks", "context_chars"):
            print(f"  {key}: {sum(m[key] for m in metrics) / len(metrics):.4f}")
    print(f"\n逐题证据及指标：{args.report}")
    print(f"人工事实核对模板：{args.reviews}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-docs", type=int, default=10)
    parser.add_argument("--max-docs", type=int, default=2)
    parser.add_argument("--max-chunks-per-doc", type=int, default=2)
    parser.add_argument("--max-chars", type=int, default=4000)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    run(parser.parse_args())
