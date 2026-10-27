import json
from pathlib import Path
import unittest

from evaluation.metrics import page_metrics, summarize


class EvaluationTest(unittest.TestCase):
    def test_page_hit_and_reciprocal_rank_use_first_relevant_chunk(self):
        self.assertEqual(page_metrics([3], [{"page_number": 1}, {"page_number": 3}, {"page_number": 3}]), {"hit_at_5": 1.0, "mrr_at_5": 0.5})
        self.assertEqual(page_metrics([], []), {"hit_at_5": None, "mrr_at_5": None})
        self.assertEqual(page_metrics([3], [{"page_number": 1}]), {"hit_at_5": 0.0, "mrr_at_5": 0.0})

    def test_unanswerable_questions_and_missing_grades_do_not_become_zero_scores(self):
        rows = [
            {"variant": "text-off", "document": "doc", "kind": "text", "answer_mode": "generated", "latency_seconds": 1, **page_metrics([3], [{"page_number": 3}])},
            {"variant": "text-off", "document": "doc", "kind": "unanswerable", "answer_mode": "extractive", "latency_seconds": 2, **page_metrics([], [])},
        ]
        summary = next(r for r in summarize(rows) if r["group"] == "all")
        self.assertEqual(summary["hit_at_5"], 1)
        self.assertEqual(summary["fallback_rate"], 0.5)
        self.assertIsNone(summary["answer_support_rate"])
        self.assertIsNone(summary["unanswerable_abstention_rate"])

    def test_pilot_has_thirty_unique_questions_and_valid_source_locations(self):
        root = Path(__file__).resolve().parents[1] / "evaluation"
        corpus = {d["id"]: d for d in json.loads((root / "corpus.json").read_text())}
        questions = [json.loads(line) for line in (root / "questions.jsonl").read_text().splitlines()]
        self.assertEqual(len(questions), 30)
        self.assertEqual(len({q["id"] for q in questions}), 30)
        self.assertEqual({k: sum(q["kind"] == k for q in questions) for k in ("text", "visual", "unanswerable")}, {"text": 20, "visual": 6, "unanswerable": 4})
        for question in questions:
            self.assertTrue(question["question"] and question["reference_answer"])
            self.assertEqual(not question["gold_pages"], question["kind"] == "unanswerable")
            self.assertTrue(all(1 <= p <= corpus[question["document"]]["pages"] for p in question["gold_pages"]))
