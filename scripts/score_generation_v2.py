"""Score human-reviewed Generation v2 output. Requires manual ratings, never guesses them."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from scripts.evaluate_generation_v2 import DATA, REPORTS, read_jsonl, response_hash, unique_index


def evaluate(reports: list[dict], reviews: list[dict]) -> dict:
    review_map = unique_index(reviews)
    ok_rows = [r for r in reports if r["ok"]]
    completed = []
    incomplete = []
    for r in ok_rows:
        review = review_map.get(r["id"])
        if not review or review.get("response_hash") != response_hash(r):
            incomplete.append((r["id"], "未审核或答案/证据已更改"))
            continue
        checks = review.get("fact_checks", [])
        if len(checks) != len(r["reference_facts"]):
            incomplete.append((r["id"], "参考事实数不匹配"))
            continue
        if any(
            check.get("fact_index") != i
            or check.get("fact") != r["reference_facts"][i]
            or type(check.get("covered")) is not bool
            for i, check in enumerate(checks)
        ):
            incomplete.append((r["id"], "存在未标注或错位的参考事实"))
            continue
        fields = ("faithfulness", "citation_accuracy", "answer_relevance")
        if any(type(review.get(name)) is not bool for name in fields):
            incomplete.append((r["id"], "存在未审核的三项人工质量判断"))
            continue
        completed.append((r, review))

    supported = sum(sum(int(x["covered"]) for x in rv["fact_checks"]) for _, rv in completed)
    total_facts = sum(len(rv["fact_checks"]) for _, rv in completed)
    n = len(completed)
    coverage_macro = (
        sum(sum(int(x["covered"]) for x in rv["fact_checks"]) / len(rv["fact_checks"])
            for _, rv in completed if rv["fact_checks"])
        / n if n else None
    )
    by_type = defaultdict(list)
    for row, rv in completed:
        by_type[row["question_type"]].append((row, rv))
    return {
        "all": len(reports),
        "api_success": len(ok_rows),
        "reviewed": n,
        "incomplete": incomplete,
        "covered_facts": supported,
        "total_facts": total_facts,
        "micro": supported / total_facts if total_facts else None,
        "macro": coverage_macro,
        "faithfulness": sum(int(rv["faithfulness"]) for _, rv in completed) / n if n else None,
        "citation_accuracy": sum(int(rv["citation_accuracy"]) for _, rv in completed) / n if n else None,
        "answer_relevance": sum(int(rv["answer_relevance"]) for _, rv in completed) / n if n else None,
        "by_type": {t: {
            "questions": len(pairs),
            "covered": sum(sum(int(x["covered"]) for x in rv["fact_checks"]) for _, rv in pairs),
            "facts": sum(len(rv["fact_checks"]) for _, rv in pairs),
        } for t, pairs in sorted(by_type.items())},
    }


def pct(x):
    return "待审核" if x is None else f"{x:.2%}"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, default=REPORTS / "generation_v2.jsonl")
    p.add_argument("--reviews", type=Path, default=REPORTS / "generation_reviews_v2.jsonl")
    p.add_argument("--expected", type=int, default=24, help="预期完整评测样本数量")
    args = p.parse_args()
    rows = read_jsonl(args.report)
    reviews = read_jsonl(args.reviews)
    unique_index(rows)
    result = evaluate(rows, reviews)
    print("========== Generation Evaluation ==========")
    print(f"已请求 {result['all']}/{args.expected} | API 成功 {result['api_success']}/{result['all']}")
    print(f"完成逐题审核：{result['reviewed']}/{result['api_success']} 条成功回答")
    print(f"Answer Completeness (micro)：{result['covered_facts']}/{result['total_facts']} = {pct(result['micro'])}")
    print(f"Answer Completeness (macro)：{pct(result['macro'])}")
    print(f"Faithfulness（逐题全部事实忠实比例）：{pct(result['faithfulness'])}")
    print(f"Citation Accuracy（逐题引用完整且正确比例）：{pct(result['citation_accuracy'])}")
    print(f"Answer Relevance（逐题切题比例）：{pct(result['answer_relevance'])}")
    print("\n按类型参考事实覆盖（已审核成功回答）：")
    for t, rec in result["by_type"].items():
        print(f"{t}: {rec['covered']}/{rec['facts']} = {pct(rec['covered']/rec['facts'] if rec['facts'] else None)}")
    if result["incomplete"]:
        print("\n待审核：")
        for case_id, why in result["incomplete"]:
            print(f"- {case_id}: {why}")
    if result["all"] != args.expected or result["reviewed"] != result["api_success"]:
        print("\n注意：上述人工指标仅针对已审核成功回答，不能作为完整 24 题结果。")
    if result["all"] and result["api_success"] < result["all"]:
        print("注意：API 失败题目没有被默认为生成正确；完整系统的端到端成功率须另行统计。")


if __name__ == "__main__":
    main()
