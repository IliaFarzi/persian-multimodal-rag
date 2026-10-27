"""Exercise real OCR, captions, both retrieval modes, generated QA and deletion."""
import argparse
from contextlib import closing
import json
import os
from pathlib import Path
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("evaluation_runs/gpu-smoke.json"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="persian-qa-smoke-") as directory:
        os.environ["APP_DATA_DIR"] = str(Path(directory) / "data")
        # Import only after assigning the isolated data directory.
        from diagnostics import setup_issues
        issues = setup_issues() + setup_issues("semantic")
        if issues:
            parser.error("\n".join(dict.fromkeys(issues)))
        import torch
        from qdrant_client import QdrantClient
        from document_app_service import DocumentIngestionService, QASessionService
        from evaluation.worker import runtime_manifest
        from examples.make_demo_pdf import make_demo_pdf
        from knowledge_base import KnowledgeBase
        from qdrant_store import QdrantChunkStore
        from runtime_lifecycle import release_all_model_memory
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        pdf = make_demo_pdf(Path(directory) / "demo.pdf")
        kb = KnowledgeBase()
        with closing(QdrantClient(":memory:")) as client:
            store = QdrantChunkStore(client=client, collection_name="smoke")
            ingestion = DocumentIngestionService(kb, vector_store=store)
            result = ingestion.ingest_pdf(pdf)
            if result.get("warnings") or result["document"]["vector_status"] != "indexed":
                raise RuntimeError("Smoke ingestion degraded: " + json.dumps(result, ensure_ascii=False))
            document_id = result["document"]["document_id"]
            qa = QASessionService(kb, vector_store=store)
            answers = {}
            for mode in ("text", "simsearch"):
                answer = qa.answer([document_id], "مقدار AUC روش آلفا چقدر است؟", retrieval_mode=mode)
                if answer["answer_mode"] != "generated" or not answer["evidence"]:
                    raise RuntimeError(f"{mode} smoke did not produce a generated answer with citations: {answer}")
                answers[mode] = answer
            deleted = ingestion.delete_document(document_id)
            if not deleted["deleted"] or deleted["vector_error"] or kb.list_documents() or store.search([document_id], "آلفا"):
                raise RuntimeError("Document or vector deletion failed")
            report = {"status": "passed", "seconds": time.perf_counter() - started, "page_count": result["document"]["page_count"], "chunk_count": result["document"]["chunk_count"], "answers": answers, "deletion_verified": True, "runtime": runtime_manifest(), "peak_torch_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2}
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
            print(f"GPU smoke passed. Report: {args.output}")
        release_all_model_memory()


if __name__ == "__main__":
    main()
