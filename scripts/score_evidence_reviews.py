"""Lesson 8: summarize human-verified evidence coverage, paired by question."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "data/evaluation/reports/evidence_generalization.jsonl"
DEFAULT_REVIEWS = ROOT / "data/evaluation/reports/evidence_review_template.jsonl"


def load(path):
    with path.open(encoding="utf-8") as file:
        return {rec["id"]: rec for line in file if line.strip()
                for rec in [json.loads(line)]}


def summarize(report, reviews):
    completed = []
    for id_, record in report.items():
        if record["error"] is not None or id_ not in reviews:
            continue
        fact_reviews = reviews[id_].get("fact_reviews", {})
        a, b = fact_reviews.get("baseline", []), fact_reviews.get("dynamic", [])
        n = len(record["reference_facts"])
        if len(a) != n or len(b) != n:
            continue
        if any(type(x.get("supported")) is not bool for x in a + b):
            continue
        if [x["fact_index"] for x in a] != list(range(n)) or [x["fact_index"] for x in b] != list(range(n)):
            raise ValueError(f"{id_}: fact_index 顺序与参考事实不一致")
        for strategy, graded in (("baseline", a), ("dynamic", b)):
            allowed = {e["evidence_id"] for e in record[strategy]["evidence"]}
            for item in graded:
                if item["supported"] and not item.get("supporting_evidence_ids"):
                    raise ValueError(f"{id_} {strategy} fact {item['fact_index']} 标记支持却无引用编号")
                if not set(item.get("supporting_evidence_ids", [])).issubset(allowed):
                    raise ValueError(f"{id_} {strategy}: 证据编号不存在")
        completed.append((record, a, b))

    print(f"已完成人工核对的配对题目：{len(completed)}/{len(report)}")
    if not completed:
        print("先在 evidence_review_template.jsonl 里填 supported=true/false，且为 true 的事实填写 supporting_evidence_ids。")
        return
    for strategy, pos in (("baseline", 1), ("dynamic", 2)):
        per_case = [sum(x["supported"] for x in pair[pos]) / len(pair[pos]) for pair in completed]
        facts_total = sum(len(pair[pos]) for pair in completed)
        facts_supported = sum(sum(x["supported"] for x in pair[pos]) for pair in completed)
        chars = [pair[0][strategy]["metrics"]["context_chars"] for pair in completed]
        print(f"{strategy}: macro Evidence Coverage={sum(per_case)/len(per_case):.2%}, "
              f"micro={facts_supported/facts_total:.2%}, "
              f"平均上下文字符数={sum(chars)/len(chars):.0f}")
    print("注意：只有全部事实完成人工核查的题目会参与统计；两个策略使用同一组题目。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    args = parser.parse_args()
    summarize(load(args.report), load(args.reviews))
