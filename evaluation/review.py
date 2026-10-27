"""Validate human_review.csv and produce a separate graded summary."""
import argparse
import csv
import json
from pathlib import Path

from evaluation.metrics import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run_dir / "predictions.jsonl").read_text().splitlines()]
    by_key = {(r["variant"], r["id"]): r for r in rows}
    seen = set()
    with (args.run_dir / "human_review.csv").open() as handle:
        for grade in csv.DictReader(handle):
            key = (grade["variant"], grade["id"])
            if key in seen or key not in by_key:
                parser.error(f"Duplicate or unknown review row: {key}")
            seen.add(key)
            for field in ("answer_supported", "citations_correct", "abstained"):
                value = grade[field].strip().lower()
                if value not in ("", "yes", "no"):
                    parser.error(f"{key}: {field} must be yes, no, or blank")
                if value and not grade["reviewer"].strip():
                    parser.error(f"{key}: identify the reviewer before recording grades")
                by_key[key][field] = {"yes": 1.0, "no": 0.0, "": None}[value]
            by_key[key]["reviewer"] = grade["reviewer"]
            by_key[key]["review_notes"] = grade["notes"]
    (args.run_dir / "graded_summary.json").write_text(json.dumps(summarize(rows), indent=2, ensure_ascii=False) + "\n")
    print("Wrote graded_summary.json; unfilled grades remain null.")


if __name__ == "__main__":
    main()
