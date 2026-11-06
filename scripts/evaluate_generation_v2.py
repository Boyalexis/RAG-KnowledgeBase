"""End-to-end generation evaluation for RAG-KnowledgeBase.

Runs each question against /ask-v2 (or /ask), saves complete responses, and
creates an *empty* human review template. No LLM-as-judge or label leakage.

Use: uv run python -m scripts.evaluate_generation_v2 --max-cases 4
     uv run python -m scripts.evaluate_generation_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "evaluation"
REPORTS = DATA / "reports"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} 不是有效 JSON") from exc
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def unique_index(rows: list[dict], key: str = "id") -> dict[str, dict]:
    indexed = {}
    for row in rows:
        k = row[key]
        if k in indexed:
            raise ValueError(f"发现重复 {key}：{k}")
        indexed[k] = row
    return indexed


def response_hash(row: dict) -> str:
    payload = {
        "question": row["question"],
        "answer": row.get("answer"),
        "sources": row.get("sources", []),
        "planner_mode": row.get("planner_mode"),
        "retrieval_mode": row.get("retrieval_mode"),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def citation_summary(answer: str, sources: list[dict]) -> dict:
    """Mechanical citation pointer check; not citation *accuracy*."""
    cited = sorted({int(x) for x in re.findall(r"\[(\d+)\]", answer)})
    source_ids = {v for s in sources for v in [s.get("id", s.get("index"))] if type(v) is int}
    return {
        "cited_ids": cited,
        "invalid_ids": [x for x in cited if x not in source_ids],
        "has_citations": bool(cited),
    }


def make_review(row: dict) -> dict:
    return {
        "id": row["id"],
        "strategy": row["strategy"],
        "question": row["question"],
        "response_hash": response_hash(row),
        "fact_checks": [
            {"fact_index": i, "fact": fact, "covered": None, "notes": ""}
            for i, fact in enumerate(row["reference_facts"])
        ],
        "faithfulness": None,
        "citation_accuracy": None,
        "answer_relevance": None,
        "unsupported_claims": [],
        "notes": "",
    }


def call_api(url: str, question: str, timeout: float, strategy: str) -> dict:
    data = {"question": question}
    if strategy == "baseline":
        data["top_k"] = 5
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urlopen(request, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = resp.status
        result = json.loads(raw)
        if not isinstance(result, dict) or not isinstance(result.get("answer"), str) or not result["answer"].strip():
            raise ValueError("HTTP 成功，但响应缺少非空 answer 字段")
        sources = result.get("sources", [])
        if not isinstance(sources, list):
            raise ValueError("sources 必须为数组")
        return {
            "ok": True,
            "http_status": status,
            "error": None,
            "answer": result["answer"],
            "sources": sources,
            "planner_mode": result.get("planner_mode"),
            "retrieval_mode": result.get("retrieval_mode"),
            "sub_queries": result.get("sub_queries"),
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
    except HTTPError as exc:
        body = exc.read(1000).decode("utf-8", errors="replace")
        return {
            "ok": False, "http_status": exc.code,
            "error": f"HTTP {exc.code}: {body[:800]}",
            "answer": None, "sources": [],
            "planner_mode": None, "retrieval_mode": None, "sub_queries": None,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
    except (URLError, TimeoutError, ValueError, json.JSONDecodeError, OSError) as exc:
        return {
            "ok": False, "http_status": None,
            "error": f"{type(exc).__name__}: {str(exc)[:800]}",
            "answer": None, "sources": [],
            "planner_mode": None, "retrieval_mode": None, "sub_queries": None,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }


def sync_reviews(report_rows: list[dict], review_file: Path) -> None:
    # Existing reviews are preserved ONLY if the answer + evidence fingerprint matches.
    existing = unique_index(read_jsonl(review_file)) if review_file.exists() else {}
    synced = []
    for row in report_rows:
        if not row["ok"]:
            continue
        old = existing.get(row["id"])
        if old is not None and old.get("response_hash") == response_hash(row):
            synced.append(old)
        else:
            synced.append(make_review(row))
    write_jsonl_atomic(review_file, synced)


def make_readable(report_rows: list[dict], filename: Path) -> None:
    parts = []
    for row in report_rows:
        parts.extend(["=" * 76, f"ID: {row['id']} | type={row['question_type']} | ok={row['ok']}",
                      f"问题：{row['question']}", f"耗时：{row['elapsed_seconds']} 秒",
                      f"规划：{row.get('planner_mode')} | 检索：{row.get('retrieval_mode')}"])
        if row.get("error"):
            parts.append(f"失败原因：{row['error']}")
            continue
        parts.extend(["\n【参考事实：仅用于审核，不参与请求】"])
        for i, fact in enumerate(row["reference_facts"]):
            parts.append(f"{i}. {fact}")
        parts.extend(["\n【最终回答】", row["answer"], "\n【实际返回的证据】"])
        for i, s in enumerate(row["sources"], 1):
            sid = s.get("id", s.get("index", i))
            parts.append(f"\n[{sid}] {s.get('title', '')} | {s.get('section', '')} | {s.get('source', '')}")
            parts.append(s.get("excerpt", ""))
    filename.parent.mkdir(parents=True, exist_ok=True)
    filename.write_text("\n".join(parts) + "\n", encoding="utf-8")


def run(args) -> None:
    if args.max_cases is not None and args.max_cases < 1:
        raise ValueError("--max-cases 必须大于 0")
    if args.interval < 0:
        raise ValueError("--interval 不能为负")
    cases = [c for c in read_jsonl(args.dataset) if c.get("reviewed") is True]
    if not cases:
        raise ValueError("没有 reviewed=true 的评测样本")
    unique_index(cases)
    strategies = [args.strategy] if args.strategy != "both" else ["v2", "baseline"]
    urls = {"v2": args.v2_url, "baseline": args.baseline_url}

    existing = unique_index(read_jsonl(args.report)) if args.report.exists() else {}
    if existing:
        unexpected = [x for x in existing.values() if x.get("strategy") not in strategies]
        if unexpected:
            raise ValueError("报告文件已含其他策略数据：请使用不同 --report 文件")
        for case in cases:
            old = existing.get(case["id"])
            if old and old.get("question") != case["question"]:
                raise ValueError(f"缓存题目变更：{case['id']}，请使用新的报告文件")

    pending = []
    selected_cases = cases[:args.max_cases] if args.max_cases is not None else cases
    for strategy in strategies:
        for case in selected_cases:
            old = existing.get(case["id"])
            if old and old.get("strategy") == strategy and not args.refresh:
                if old.get("ok") or not args.retry_errors:
                    continue
            pending.append((strategy, case))

    print(f"测试题：{len(selected_cases)}，待请求：{len(pending)}，已有报告：{len(existing)}", flush=True)
    successes_since_start = 0
    for i, (strategy, case) in enumerate(pending, 1):
        if i > 1 and args.interval:
            print(f"请求间隔控制：{args.interval:g} 秒", flush=True)
            time.sleep(args.interval)
        print(f"[{i}/{len(pending)}] {strategy}: {case['id']}", flush=True)
        result = call_api(urls[strategy], case["question"], args.timeout, strategy)
        row = {
            "id": case["id"],
            "strategy": strategy,
            "question": case["question"],
            "question_type": case["question_type"],
            "reference_facts": case["reference_facts"],
            "reference_answer": case.get("reference_answer"),
            "source_commit": case.get("source_commit"),
            "answerability": case.get("answerability"),
            "tested_at": datetime.now(timezone.utc).isoformat(),
            **result,
        }
        if row["ok"]:
            row["citation_pointer_check"] = citation_summary(row["answer"], row["sources"])
            successes_since_start += 1
        else:
            row["citation_pointer_check"] = None
        existing[case["id"]] = row
        rows = [existing[k] for k in sorted(existing)]
        write_jsonl_atomic(args.report, rows)  # per-case checkpoint
        sync_reviews(rows, args.reviews)
        make_readable(rows, args.readable)
        print(f"  HTTP={row['http_status']} | ok={row['ok']} | {row['elapsed_seconds']}s", flush=True)
        if row["error"]:
            print(f"  {row['error'][:180]}", flush=True)

    rows = [existing[k] for k in sorted(existing)]
    write_jsonl_atomic(args.report, rows)
    sync_reviews(rows, args.reviews)
    make_readable(rows, args.readable)
    subset = [existing[c["id"]] for c in selected_cases if c["id"] in existing]
    good = [r for r in subset if r["ok"]]
    durations = [r["elapsed_seconds"] for r in good]
    fallbacks = [r for r in good if r.get("planner_mode") == "fallback"]
    invalid_cites = [r["id"] for r in good if r.get("citation_pointer_check", {}).get("invalid_ids")]
    print("\n========== Generation 运行结果 ==========")
    print(f"API 成功：{len(good)}/{len(subset)} (仅已运行题目)")
    print(f"Planner fallback：{len(fallbacks)} | 引用指针非法的题目：{invalid_cites}")
    if durations:
        print(f"平均 API 耗时：{sum(durations) / len(durations):.2f}s (仅成功请求)")
    print(f"回答报告：{args.report}\n人工审核表：{args.reviews}\n可读审核文本：{args.readable}")
    print("注意：这些输出未自动判定 Faithfulness、Completeness 或 Citation Accuracy。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=DATA / "questions_generalization.jsonl")
    ap.add_argument("--strategy", choices=("v2", "baseline"), default="v2")
    ap.add_argument("--v2-url", default="http://127.0.0.1:8001/ask-v2")
    ap.add_argument("--baseline-url", default="http://127.0.0.1:8001/ask")
    ap.add_argument("--report", type=Path, default=REPORTS / "generation_v2.jsonl")
    ap.add_argument("--reviews", type=Path, default=REPORTS / "generation_reviews_v2.jsonl")
    ap.add_argument("--readable", type=Path, default=REPORTS / "generation_review_readable.txt")
    ap.add_argument("--timeout", type=float, default=240.0, help="单次 HTTP 请求超时秒数")
    ap.add_argument("--interval", type=float, default=55.0, help="客户端请求之间最小固定间隔秒数；配合服务端限速器")
    ap.add_argument("--max-cases", type=int, help="只测试前 N 道题，首次建议 4")
    ap.add_argument("--retry-errors", action="store_true", help="只重跑已缓存但失败的题目")
    ap.add_argument("--refresh", action="store_true", help="重新请求全部已运行题目（可能覆盖旧回答）")
    run(ap.parse_args())
