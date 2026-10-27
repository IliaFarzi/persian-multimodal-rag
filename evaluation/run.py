"""Run isolated, reproducible pilot variants without changing the application's data."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from config import ROOT
from evaluation.metrics import summarize


def write_report(output, rows):
    summary = summarize(rows)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    fields = ["variant", "id", "answer_supported", "citations_correct", "abstained", "reviewer", "notes"]
    with (output / "human_review.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captions", choices=("both", "on", "off"), default="both")
    parser.add_argument("--retrievers", choices=("both", "text", "simsearch"), default="both")
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--allow-different-pdfs", action="store_true", help="Record changed hashes; this is a different corpus, not an exact reproduction.")
    args = parser.parse_args()
    corpus = json.loads((ROOT / "evaluation/corpus.json").read_text())
    inputs = []
    for document in corpus:
        path = args.corpus_dir.resolve() / document["filename"]
        if not path.is_file():
            parser.error(f"Missing {path}. Obtain the paper from {document['source']}.")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != document["sha256"] and not args.allow_different_pdfs:
            parser.error(f"Hash mismatch for {path.name}. Use the documented copy, or explicitly pass --allow-different-pdfs and review every gold location.")
        inputs.append({**document, "actual_sha256": actual, "matches_reference": actual == document["sha256"]})
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "inputs.json").write_text(json.dumps(inputs, indent=2, ensure_ascii=False) + "\n")
    rows = []
    for captions in (["off", "on"] if args.captions == "both" else [args.captions]):
        variant_dir = output / f"captions-{captions}"
        env = {**os.environ, "APP_DATA_DIR": str(variant_dir / "data"), "TOKENIZERS_PARALLELISM": "false"}
        command = [sys.executable, "-m", "evaluation.worker", "--corpus-dir", str(args.corpus_dir.resolve()), "--output-dir", str(variant_dir), "--captions", captions, "--retrievers", args.retrievers]
        if args.retrieval_only:
            command.append("--retrieval-only")
        subprocess.run(command, env=env, cwd=ROOT, check=True)
        rows.extend(json.loads(line) for line in (variant_dir / "predictions.jsonl").read_text().splitlines())
    with (output / "predictions.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_report(output, rows)
    print(f"Results: {output}. Human answer/citation grades remain unfilled until reviewed.")


if __name__ == "__main__":
    main()
