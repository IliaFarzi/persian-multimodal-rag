"""One caption condition per process, including separate SQLite and vector stores."""
import argparse
import hashlib
import importlib.metadata
import json
import platform
import resource
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from config import ROOT, APP_DATA_DIR, EMBEDDING_MODEL_PATH, VGEN_MODEL_PATH
from diagnostics import setup_issues
from evaluation.metrics import page_metrics


def runtime_manifest():
    import torch
    models = {}
    for name, path in (("vgen", VGEN_MODEL_PATH), ("embedding", EMBEDDING_MODEL_PATH)):
        receipt = path / ".model-revision.json"
        models[name] = {
            "revision_receipt": json.loads(receipt.read_text()) if receipt.is_file() else None,
            "config_sha256": hashlib.sha256((path / "config.json").read_bytes()).hexdigest() if (path / "config.json").is_file() else None,
            "provenance": "pinned download receipt" if receipt.is_file() else "local artifact revision unverified",
        }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True))
    except subprocess.CalledProcessError:
        commit, dirty = None, True
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "cpu": platform.processor(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2 if torch.cuda.is_available() else None,
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "bitsandbytes", "qdrant-client", "PyMuPDF", "hazm", "pytesseract")},
        "git_commit": commit, "git_dirty": dirty, "models": models,
        "questions_sha256": hashlib.sha256((ROOT / "evaluation/questions.jsonl").read_bytes()).hexdigest(),
        "inference": {"quantization": "fp4", "do_sample": False, "max_new_tokens": 256, "embedding_max_length": 512, "top_k": 5},
        "chunking": {"target_min": 700, "target_max": 1000, "hard_max": 1400},
        "ocr": {"languages": "fas+eng", "render_dpi": 200},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--captions", choices=("on", "off"), required=True)
    parser.add_argument("--retrievers", choices=("both", "text", "simsearch"), required=True)
    parser.add_argument("--retrieval-only", action="store_true")
    args = parser.parse_args()
    if APP_DATA_DIR.resolve() != (args.output_dir / "data").resolve():
        parser.error("Use evaluation.run; worker APP_DATA_DIR must point at its isolated output/data directory.")
    issues = setup_issues("ingestion", include_visual_captions=args.captions == "on")
    if not args.retrieval_only:
        issues += setup_issues("qa")
    if args.retrievers != "text":
        issues += setup_issues("semantic")
    if issues:
        parser.error("\n".join(dict.fromkeys(issues)))
    from qdrant_client import QdrantClient
    import torch
    from document_app_service import DocumentIngestionService, QASessionService
    from knowledge_base import KnowledgeBase
    from qdrant_store import QdrantChunkStore
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = runtime_manifest()
    manifest.update({"captions": args.captions, "retrievers": args.retrievers, "retrieval_only": args.retrieval_only, "vector_backend": "Qdrant client local mode, isolated in-memory collection"})
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    kb = KnowledgeBase()
    client = QdrantClient(":memory:")
    store = QdrantChunkStore(client=client, collection_name="pilot_chunks")
    ingestion = DocumentIngestionService(kb, vector_store=store, include_visual_captions=args.captions == "on", index_vectors=args.retrievers != "text")
    documents = json.loads((ROOT / "evaluation/corpus.json").read_text())
    questions = [json.loads(line) for line in (ROOT / "evaluation/questions.jsonl").read_text().splitlines()]
    document_ids, ingestion_report = {}, []
    try:
        for document in documents:
            started = time.perf_counter()
            print(f"[{args.captions}] Ingesting {document['filename']}", flush=True)
            result = ingestion.ingest_pdf(args.corpus_dir / document["filename"], original_name=document["filename"], progress_callback=lambda page, total: print(f"  page {page}/{total}", flush=True))
            document_ids[document["id"]] = result["document"]["document_id"]
            ingestion_report.append({"document": document["id"], "seconds": time.perf_counter() - started, "warnings": result.get("warnings", []), "chunks": result["document"]["chunk_count"]})
            (args.output_dir / "ingestion.json").write_text(json.dumps(ingestion_report, indent=2) + "\n")
            if result.get("warnings"):
                raise RuntimeError("Caption condition degraded; inspect ingestion.json rather than reporting it as a successful caption experiment.")
            if args.retrievers != "text" and result["document"]["vector_status"] != "indexed":
                raise RuntimeError("Semantic indexing failed: " + str(result["document"].get("vector_error")))
        qa = QASessionService(kb, vector_store=store)
        with (args.output_dir / "predictions.jsonl").open("w") as handle:
            for mode in (["text", "simsearch"] if args.retrievers == "both" else [args.retrievers]):
                for question in questions:
                    started = time.perf_counter()
                    ids = [document_ids[question["document"]]]
                    if args.retrieval_only:
                        result = {"evidence": qa.retrieve(ids, question["question"], top_k=5, retrieval_mode=mode), "answer": "", "answer_mode": "not_requested", "warnings": []}
                    else:
                        result = qa.answer(ids, question["question"], top_k=5, retrieval_mode=mode)
                    row = {**question, **result, **page_metrics(question["gold_pages"], result["evidence"]), "variant": f"{mode}-captions-{args.captions}", "latency_seconds": time.perf_counter() - started}
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    print(f"  {row['variant']} {question['id']} {row['answer_mode']}", flush=True)
    finally:
        manifest["peak_process_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        manifest["peak_torch_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        client.close()


if __name__ == "__main__":
    main()
