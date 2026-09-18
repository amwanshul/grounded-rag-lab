import json
from pathlib import Path

from app.pipeline import build_demo_pipeline


def reciprocal_rank(relevant: set[str], results: list[dict]) -> float:
    for rank, item in enumerate(results, start=1):
        if item["document_id"] in relevant:
            return 1.0 / rank
    return 0.0


def main():
    cases = json.loads(Path(__file__).with_name("golden.json").read_text())
    pipeline = build_demo_pipeline()

    hits = 0
    mrr_total = 0.0

    for case in cases:
        results = pipeline.search(case["query"], k=5)
        relevant = set(case["relevant"])
        hits += int(any(r["document_id"] in relevant for r in results))
        mrr_total += reciprocal_rank(relevant, results)

    recall_at_5 = hits / len(cases)
    mrr = mrr_total / len(cases)

    print(f"queries={len(cases)}")
    print(f"recall@5={recall_at_5:.3f}")
    print(f"mrr={mrr:.3f}")


if __name__ == "__main__":
    main()
