"""Page-level pilot metrics. Missing human grades remain unknown, never zero."""
import statistics


def page_metrics(gold_pages, evidence, top_k=5):
    if not gold_pages:
        return {"hit_at_5": None, "mrr_at_5": None}
    rank = next((i for i, hit in enumerate(evidence[:top_k], 1) if hit["page_number"] in gold_pages), None)
    return {"hit_at_5": float(rank is not None), "mrr_at_5": 1 / rank if rank else 0.0}


def average(values):
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


def summarize(rows):
    groups = {}
    for row in rows:
        for group in ("all", row["document"], row["kind"]):
            groups.setdefault((row["variant"], group), []).append(row)
    results = []
    for (variant, group), items in sorted(groups.items()):
        modes = [r["answer_mode"] for r in items if r["answer_mode"] != "not_requested"]
        results.append({
            "variant": variant, "group": group, "questions": len(items),
            "answerable_questions": sum(r["hit_at_5"] is not None for r in items),
            "hit_at_5": average([r["hit_at_5"] for r in items]),
            "mrr_at_5": average([r["mrr_at_5"] for r in items]),
            "mean_latency_seconds": average([r["latency_seconds"] for r in items]),
            "fallback_rate": modes.count("extractive") / len(modes) if modes else None,
            "answer_support_rate": average([r.get("answer_supported") for r in items]),
            "citation_correctness_rate": average([r.get("citations_correct") for r in items]),
            "unanswerable_abstention_rate": average([r.get("abstained") for r in items if r["kind"] == "unanswerable"]),
            "human_graded_answers": sum(r.get("answer_supported") is not None for r in items),
        })
    return results
