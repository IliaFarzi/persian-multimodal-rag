# Validation record

Validation is performed on Linux with Python 3.12.3 and the dependencies pinned in `requirements.txt`. GPU checks use an NVIDIA RTX 2060 with 6 GiB VRAM and PyTorch 2.11.0+cu130.

## Completed checks

- The original suite passed 33 tests in the initial CPU-only review.
- The expanded suite passed 47 tests from a fresh source-only clone in a newly installed, isolated Python 3.12 environment with CUDA hidden, offline Hugging Face settings, and no model downloads or Qdrant server. `pip check` passed. Installation used locally cached wheels because external package downloads were unreliable. `constraints.txt` records all 109 resolved distributions, including conditional CUDA dependencies. Every dependency version matches the environment used for the pilot.
- Real OCR and layout extraction succeeded on the first page of each supplied evaluation PDF.
- A real Gemma answer correctly extracted the AUC value 0.91 from an original synthetic evidence passage. Cold execution took 33.79 seconds, with 3,413 MiB peak PyTorch GPU allocation. This is a single smoke observation, not aggregate QA accuracy.
- Qdrant 1.17.1 was pulled and started by its pinned image digest. The real server probe passed upsert, `query_points`, document filtering, deletion and document isolation; it removed its temporary collection afterward.

## Provenance and limits

[Environment and completed checks](../validation/environment.json) record the CPU, system-tool versions, resolved packages, tested source commit and real Qdrant probe results.

The locally available model directories predate this release and have no download revision receipts. Their inference-relevant artifacts were independently hashed and compared with the pinned upstream Hugging Face Git/LFS metadata: **13 of 13 Gemma files and eight of eight E5 files matched**. [Artifact checksums](../validation/model-artifacts.json) record the expected and observed SHA-256 or Git blob hashes. Duplicate E5 `.bin` weights and model-card files are excluded from this inference check.

Run manifests preserve the receipt state observed at startup. An absent receipt alone does not establish a model revision; the separately recorded artifact verification establishes the revision for the local files used in this validation. New downloads receive revision receipts from the helper.

## Completed full pilot

All four conditions completed, with 30 questions each and no ingestion warnings. All 120 attempts are retained in the local run directory. [Aggregate results](../validation/pilot-summary.csv), [question-level retrieval metadata](../validation/pilot-retrieval.jsonl), and [runtime manifests](../validation/pilot-runtime.json) are included in the public source. Raw answers, extracted passages, page images and crops remain in the ignored local `evaluation_runs` directory.

| Retrieval | Captions | Hit@5 | MRR@5 | Mean end-to-end QA seconds |
|---|---|---:|---:|---:|
| BM25 | Off | 0.769 | 0.649 | 27.04 |
| E5 | Off | 0.846 | 0.633 | 46.62 |
| BM25 | On | 0.692 | 0.571 | 7.59 |
| E5 | On | 0.885 | 0.579 | 10.34 |

Retrieval means exclude the four unsupported questions per condition. Timing includes all 30 attempts, generation and loading. These runs occurred sequentially during release preparation, with differing cache and host-load conditions; the large timing differences must not be attributed to captions or used as a controlled speed claim.

Caption effects were mixed. For the English paper, BM25 Hit@5 dropped from 0.692 to 0.462, while E5 Hit@5 increased from 0.846 to 0.923. For the Persian paper, BM25 Hit@5 increased from 0.846 to 0.923, while E5 stayed at 0.846. These are small, document-scoped page-retrieval observations. The gold annotations and answer/citation grades still need independent human review.

## Fresh-clone GPU smoke

The [GPU smoke report](../validation/gpu-smoke.json) comes from the clean source clone and independently installed environment. It passed in 59.12 seconds on the RTX 2060, with a peak PyTorch allocation of 3,711.45 MiB. It processed two original synthetic pages into eight chunks and three captioned visual regions. Both BM25 and semantic QA generated the expected AUC value 0.91, and document/vector deletion was verified.

This controlled synthetic check verifies integration and a known simple fact; it does not establish real-document answer accuracy. The separately recorded app walkthrough demonstrates usability.

## Recorded walkthrough

The [two-minute recording](assets/demo.webm) uses the actual local Streamlit application and the original synthetic PDF. It includes successful ingestion, visual-caption inspection, generated BM25 and semantic answers, and source evidence. [Capture metadata](../validation/demo-recording.json) records the observed elapsed time and playback speed. Screenshots were inspected against the rendered app.
