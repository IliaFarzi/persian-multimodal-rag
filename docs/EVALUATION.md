# Pilot evaluation protocol

This pilot assesses the integrated application. It is not a new benchmark or evidence of state-of-the-art performance.

## Inputs and acquisition

Obtain authorized copies of the following papers and name them `1.pdf` and `2.pdf` in a local directory:

| Local name | Paper | PDF pages | SHA-256 of the supplied evaluation copy |
|---|---|---|---|
| `1.pdf` | [Elman vs deep neural networks for mild Alzheimer's diagnosis](https://doi.org/10.52547/shefa.10.1.1) | 11 | `af820a1baf1793b317f322467ae871408af0348969ea3864aa24e9f54623739e` |
| `2.pdf` | [MLP-based DDoS detection](https://doi.org/10.22034/tjee.2022.15567) | 10 | `4a266ce29bf42a464901d7dbf042cee3a17bf67e4da04fc2f614944a2d6fbae3` |

The first paper is Persian with an English abstract; the second is English. All questions are Persian. Pages use **one-based physical PDF page numbers**, not journal pagination.

The local copies were produced with PDFsharp; a publisher download may have different bytes. The runner rejects mismatched hashes by default. If the supplied copies are unavailable, pass `--allow-different-pdfs` explicitly and independently review every gold page and figure against your copy. Such a run is a changed corpus, not an exact reproduction. The actual hashes and mismatch status are recorded in `inputs.json`. We do not assume access implies redistribution permission.

## Gold annotations

`evaluation/questions.jsonl` contains 15 questions per paper: ten text, three visual-associated, and two unsupported. Each entry identifies a reference answer, supporting PDF pages, and a source note. The annotations were checked against the local source text; the Persian page-eight chart and table were also visually inspected. `source_checked` does not mean independently human-reviewed. Verify the visual cells and unsupported-answer absence claims before treating the gold set as finalized.

The questions deliberately preserve source inconsistencies. For example, the DDoS training paragraph says 17 epochs while its figure caption says ten. The evaluation must not silently correct the underlying paper.

## Experiment

Run from the repository root after installing the requirements and model files:

```bash
python -m evaluation.run --corpus-dir /path/to/papers --output-dir evaluation_runs/pilot
```

The output directory must be new. Each caption condition runs in a separate process and has its own SQLite database, crops, and in-memory Qdrant collection. The same OCR languages, rendering DPI, chunking settings, model configurations and top-k are used in all four conditions. Queries are scoped to their annotated document, matching the app's explicit document selection.

| Condition | Retrieval | Visual captions |
|---|---|---|
| `text-captions-off` | BM25 | Disabled |
| `simsearch-captions-off` | E5 cosine similarity | Disabled |
| `text-captions-on` | BM25 | Enabled |
| `simsearch-captions-on` | E5 cosine similarity | Enabled |

For a quick OCR/retrieval-only check without Gemma or E5:

```bash
python -m evaluation.run --corpus-dir /path/to/papers --output-dir evaluation_runs/lexical-check --captions off --retrievers text --retrieval-only
```

That command produces no generated answers and must not be reported as a full four-condition evaluation.

## Artifacts and metrics

- `inputs.json`: source identities and actual input hashes.
- `captions-*/manifest.json`: package versions, hardware, model receipts or explicitly unverified local provenance, generation settings and peak process/GPU allocations.
- `captions-*/ingestion.json`: ingestion duration, chunk counts and warnings. Caption or indexing failures stop a condition rather than silently becoming a successful caption experiment.
- `predictions.jsonl`: question-level answers, retrieved source text, locations, scores, generation mode, warnings and timing.
- `summary.json` / `summary.csv`: metrics grouped by condition, document, and question kind.
- `human_review.csv`: blank answer-support, citation-correctness and abstention grades.

**Hit@5** is one if any of the first five retrieved chunks falls on an annotated supporting page, and zero otherwise. **MRR@5** is the reciprocal rank of the first such chunk, or zero. Unsupported questions have no retrieval gold pages and are excluded from these means. These are page-level metrics; they do not establish that the precise answer passage was retrieved.

Latency covers retrieval and generation for full runs, or retrieval alone for retrieval-only runs. It includes cold model loading when triggered; conditions run in a fixed order, so timing is descriptive rather than a controlled performance comparison. Ingestion time is separate. GPU memory is PyTorch's peak allocated memory, not total device consumption; process RSS includes Python and CPU runtime allocations.

Fallback rate counts `extractive` outcomes among all answer attempts, including insufficient-evidence outcomes in the denominator. Retrieval-only runs have no fallback rate. A generated refusal still has mode `generated`; abstention is graded separately rather than guessed from the mode.

## Human review

Review the source PDF, full evidence and answer for each question. Fill `yes`, `no`, or leave unknown fields blank, and identify the reviewer:

- `answer_supported`: the substantive answer is supported by the document and matches the question/reference. For unsupported questions, a correct refusal qualifies; an invented answer does not.
- `citations_correct`: every source citation actually used in the answer supports its associated claim and identifies the correct page/paragraph. A ranked evidence list alone is insufficient. Leave this blank for a justified refusal with no factual source claim.
- `abstained`: for unsupported questions, the answer explicitly recognizes that the requested information is not supplied. Leave blank for answerable questions.

```bash
python -m evaluation.review evaluation_runs/pilot
```

The review command writes a separate `graded_summary.json` and keeps raw predictions unchanged. Missing grades stay `null`. Human review is not replaced by an LLM judge. Audit gold annotations independently before reporting finalized results.

## Interpretation

There are only two documents. Visual-associated questions can sometimes be answered from surrounding OCR text, so this is not a strictly visual-only ablation. Page-level labels, overlapping contextual chunks, document-scoped search, limited gold pages and author-created questions constrain conclusions. Report the Persian and cross-language groups separately, include failure examples, and publish negative or neutral caption effects as measured. Do not tune prompts on these 30 questions and then describe them as an untouched test set.
