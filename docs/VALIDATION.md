# Validation record

Validation is performed on Linux with Python 3.12.3 and the dependencies pinned in `requirements.txt`. GPU checks use an NVIDIA RTX 2060 with 6 GiB VRAM and PyTorch 2.11.0+cu130.

## Completed checks

- The original suite passed 33 tests in the initial CPU-only review.
- The expanded suite passed 46 tests in a newly installed, isolated Python 3.12 environment with CUDA hidden, offline Hugging Face settings, and no model downloads or Qdrant server. `pip check` passed. Installation used locally cached wheels because external package downloads were unreliable. `constraints.txt` records all 109 resolved distributions, including conditional CUDA dependencies.
- Real OCR and layout extraction succeeded on the first page of each supplied evaluation PDF.
- A real Gemma answer correctly extracted the AUC value 0.91 from an original synthetic evidence passage. Cold execution took 33.79 seconds, with 3,413 MiB peak PyTorch GPU allocation. This is a single smoke observation, not aggregate QA accuracy.
- Qdrant 1.17.1 was pulled and started by its pinned image digest. The real server probe passed upsert, `query_points`, document filtering, deletion and document isolation; it removed its temporary collection afterward.

## Provenance and limits

The locally available model directories predate this release and have no download revision receipts. Their inference-relevant artifacts were independently hashed and compared with the pinned upstream Hugging Face Git/LFS metadata: **13 of 13 Gemma files and eight of eight E5 files matched**. [Artifact checksums](../validation/model-artifacts.json) record the expected and observed SHA-256 or Git blob hashes. Duplicate E5 `.bin` weights and model-card files are excluded from this inference check.

Run manifests preserve the receipt state observed at startup. An absent receipt alone does not establish a model revision; the separately recorded artifact verification establishes the revision for the local files used in this validation. New downloads receive revision receipts from the helper.

The full pilot, generated-answer smoke check and recorded demonstration are separate release checks. Do not infer their completion from passing unit tests. Measured pilot results and any remaining checks are added to this record after execution.
