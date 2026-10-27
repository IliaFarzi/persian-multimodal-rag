import hashlib
import json
import logging
import re
import shutil
from pathlib import Path
from uuid import uuid4

from captioning import FarsiCaptioner
from chunking import ContextualScientificChunker
from knowledge_base import KnowledgeBase
from layout_extraction import PDFLayoutExtractor
from ocr import PersianOCR
from preprocessing import PersianPreprocessor
from qa_generation import FarsiQAGenerator
from qdrant_store import QdrantChunkStore
from retrieval import BM25Retriever
from config import APP_DATA_DIR
from utils import rendered_pdf_pages


UPLOAD_DIR = APP_DATA_DIR / "uploads"
PAGE_IMAGE_DIR = APP_DATA_DIR / "page_images"
REGION_DIR = APP_DATA_DIR / "regions"


def ensure_runtime_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    REGION_DIR.mkdir(parents=True, exist_ok=True)


def compute_file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_uploaded_pdf(uploaded_file):
    ensure_runtime_dirs()
    safe_name = Path(uploaded_file.name).name
    target = UPLOAD_DIR / f"{uuid4().hex}_{safe_name}"
    with open(target, "wb") as handle:
        handle.write(uploaded_file.getbuffer())
    return target


def is_within_path(path, root):
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def remove_app_path(path, directory=False):
    path = Path(path)
    allowed_roots = (UPLOAD_DIR, PAGE_IMAGE_DIR, REGION_DIR)
    if not any(is_within_path(path, root) for root in allowed_roots):
        return []
    if not path.exists():
        return []

    if directory:
        if path.is_dir():
            shutil.rmtree(path)
            return [str(path)]
        return []

    if path.is_file():
        path.unlink()
        return [str(path)]
    return []


def remove_uploaded_files_by_hash(file_hash, skip=None):
    if not file_hash or not UPLOAD_DIR.exists():
        return []

    skipped = {str(Path(path)) for path in (skip or [])}
    removed = []
    for candidate in UPLOAD_DIR.glob("*.pdf"):
        if str(candidate) in skipped:
            continue
        try:
            if compute_file_hash(candidate) == file_hash:
                candidate.unlink()
                removed.append(str(candidate))
        except OSError:
            continue
    return removed


class DocumentIngestionService:
    def __init__(
        self,
        knowledge_base=None,
        ocr=None,
        captioner=None,
        preprocessor=None,
        chunker=None,
        vector_store=None,
        include_visual_captions=True,
        index_vectors=True,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self._ocr = ocr
        self._captioner = captioner
        self._preprocessor = preprocessor
        self._chunker = chunker
        self._vector_store = vector_store
        self.include_visual_captions = include_visual_captions
        self.index_vectors = index_vectors

    @property
    def ocr(self):
        if self._ocr is None:
            self._ocr = PersianOCR()
        return self._ocr

    @property
    def captioner(self):
        if self._captioner is None:
            self._captioner = FarsiCaptioner()
        return self._captioner

    @property
    def preprocessor(self):
        if self._preprocessor is None:
            self._preprocessor = PersianPreprocessor()
        return self._preprocessor

    @property
    def chunker(self):
        if self._chunker is None:
            self._chunker = ContextualScientificChunker()
        return self._chunker

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = QdrantChunkStore()
        return self._vector_store

    def ingest_pdf(self, pdf_path, original_name=None, progress_callback=None):
        pdf_path = Path(pdf_path)
        file_hash = compute_file_hash(pdf_path)
        existing = self.knowledge_base.get_document_by_hash(file_hash)
        if existing and existing["status"] == "ready":
            message = "Document already ingested."
            if str(pdf_path) != existing.get("source_path"):
                remove_app_path(pdf_path)
            if self.index_vectors and existing.get("vector_status") != "indexed":
                chunks = self.knowledge_base.load_chunks([existing["document_id"]])
                vector_error = self._index_vectors(
                    existing["document_id"],
                    existing["file_name"],
                    chunks,
                )
                existing = self.knowledge_base.get_document(existing["document_id"])
                if vector_error:
                    message += f" SimSearch indexing failed: {vector_error}"
                else:
                    message += " SimSearch index refreshed."
            return {
                "status": "skipped",
                "document": existing,
                "message": message,
            }

        document_id = existing["document_id"] if existing else uuid4().hex
        file_name = original_name or pdf_path.name
        self.knowledge_base.start_document(
            document_id,
            file_name,
            file_hash,
            source_path=str(pdf_path),
        )

        try:
            ensure_runtime_dirs()
            warnings = []
            with rendered_pdf_pages(pdf_path) as image_paths:
                stored_images = self._persist_page_images(document_id, image_paths)
            chunks = []
            region_records = []
            chunk_index = 0

            with PDFLayoutExtractor(
                pdf_path,
                document_id,
                region_root=REGION_DIR,
            ) as layout_extractor:
                for page_number, image_path in enumerate(stored_images, start=1):
                    if progress_callback:
                        progress_callback(page_number, len(stored_images))

                    page_layout = layout_extractor.extract_page(
                        page_number,
                        image_path,
                    )
                    for region in page_layout.regions if self.include_visual_captions else []:
                        try:
                            region.caption = str(self.captioner.generate_caption(
                                region.crop_path,
                                region_type=region.region_type,
                                prompt=region.prompt,
                            ))
                            if not region.caption.strip():
                                warnings.append(f"Page {page_number}, region {region.region_index}: no visual caption generated.")
                        except Exception:
                            logging.getLogger(__name__).exception("Visual caption failed on page %s", page_number)
                            warnings.append(f"Page {page_number}, region {region.region_index}: visual caption generation failed.")

                    paragraph_layouts = page_layout.paragraphs()
                    region_records.extend(
                        region.to_record() for region in page_layout.regions
                    )

                    page_chunks = self.chunker.build_page_chunks(
                        paragraph_layouts,
                        file_name=file_name,
                    )
                    for page_chunk in page_chunks:
                        retrieval_text = self.preprocessor.normalize(
                            page_chunk["retrieval_text"]
                        )
                        display_text = self.preprocessor.normalize(
                            page_chunk["display_text"]
                        )
                        chunks.append(
                            {
                                "document_id": document_id,
                                "page_number": page_number,
                                "paragraph_index": page_chunk.get("paragraph_index"),
                                "paragraph_span": page_chunk.get("paragraph_span"),
                                "heading_text": self.preprocessor.normalize(
                                    page_chunk.get("heading_text", "")
                                ),
                                "region_indices": json.dumps(
                                    page_chunk.get("region_indices", []),
                                    ensure_ascii=False,
                                ),
                                "chunk_index": chunk_index,
                                "image_path": str(image_path),
                                "ocr_text": self.preprocessor.normalize(
                                    page_chunk.get("ocr_text", "")
                                ),
                                "image_caption": self.preprocessor.normalize(
                                    page_chunk.get("image_caption", "")
                                ),
                                "display_text": display_text,
                                "retrieval_text": retrieval_text,
                                "combined_text": retrieval_text,
                            }
                        )
                        chunk_index += 1

            if not chunks:
                raise ValueError("No searchable content was extracted. Check OCR language data and PDF page quality.")
            self.knowledge_base.replace_regions(document_id, region_records)
            self.knowledge_base.replace_chunks(document_id, chunks)
            self.knowledge_base.mark_document_ready(
                document_id,
                page_count=len(stored_images),
                chunk_count=len(chunks),
            )
            vector_error = self._index_vectors(document_id, file_name, chunks) if self.index_vectors else None
            document = self.knowledge_base.get_document_by_hash(file_hash)
            message = "Document ingested successfully."
            if vector_error:
                message += f" SimSearch indexing failed: {vector_error}"
            return {
                "status": "ingested",
                "document": document,
                "message": message,
                "warnings": warnings,
            }
        except Exception as exc:
            self.knowledge_base.mark_document_failed(document_id, exc)
            raise

    def _index_vectors(self, document_id, file_name, chunks):
        if not self.vector_store.is_available():
            error = "Qdrant is unavailable; text search remains available."
            self.knowledge_base.mark_vector_failed(document_id, error)
            return error
        self.knowledge_base.mark_vector_indexing(document_id)
        try:
            vector_chunks = []
            for chunk in chunks:
                vector_chunk = dict(chunk)
                vector_chunk["file_name"] = file_name
                vector_chunks.append(vector_chunk)
            self.vector_store.upsert_chunks(vector_chunks)
            self.knowledge_base.mark_vector_indexed(document_id)
            return None
        except Exception as exc:
            logging.getLogger(__name__).exception("Vector indexing failed for %s", document_id)
            self.knowledge_base.mark_vector_failed(document_id, exc)
            return str(exc)

    def _persist_page_images(self, document_id, image_paths):
        document_dir = PAGE_IMAGE_DIR / document_id
        document_dir.mkdir(parents=True, exist_ok=True)
        stored_paths = []

        for index, image_path in enumerate(image_paths, start=1):
            source = Path(image_path)
            target = document_dir / f"page_{index:04d}{source.suffix or '.jpg'}"
            shutil.copyfile(source, target)
            stored_paths.append(target)

        return stored_paths

    def delete_document(self, document_id):
        document = self.knowledge_base.get_document(document_id)
        if not document:
            return {"deleted": False, "message": "Document not found.", "removed_paths": []}

        removed_paths = []
        vector_error = None
        try:
            self.vector_store.delete_document_vectors(document_id)
        except Exception as exc:
            vector_error = str(exc)

        removed_paths.extend(remove_app_path(PAGE_IMAGE_DIR / document_id, directory=True))
        removed_paths.extend(remove_app_path(REGION_DIR / document_id, directory=True))

        source_path = document.get("source_path")
        if source_path:
            removed_paths.extend(remove_app_path(source_path))

        removed_paths.extend(
            remove_uploaded_files_by_hash(document.get("file_hash"), skip=removed_paths)
        )
        self.knowledge_base.delete_document(document_id)

        return {
            "deleted": True,
            "message": f"Deleted {document['file_name']}.",
            "removed_paths": removed_paths,
            "vector_error": vector_error,
        }

    def purge_dev_data(self):
        vector_error = None
        try:
            self.vector_store.purge_collection()
        except Exception as exc:
            vector_error = str(exc)

        removed_paths = []
        for path in (UPLOAD_DIR, PAGE_IMAGE_DIR, REGION_DIR):
            if path.exists():
                shutil.rmtree(path)
                removed_paths.append(str(path))

        ensure_runtime_dirs()
        self.knowledge_base.purge_all()
        return {
            "purged": True,
            "removed_paths": removed_paths,
            "vector_error": vector_error,
        }


class QASessionService:
    def __init__(
        self,
        knowledge_base=None,
        preprocessor=None,
        answer_generator=None,
        vector_store=None,
    ):
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.preprocessor = preprocessor or PersianPreprocessor()
        self._answer_generator = answer_generator
        self._vector_store = vector_store

    @property
    def answer_generator(self):
        if self._answer_generator is None:
            self._answer_generator = FarsiQAGenerator()
        return self._answer_generator

    @property
    def vector_store(self):
        if self._vector_store is None:
            self._vector_store = QdrantChunkStore()
        return self._vector_store

    def simsearch_available(self):
        return self.vector_store.is_available()

    def retrieve(self, document_ids, question, top_k=5, retrieval_mode="text"):
        if retrieval_mode not in ("text", "simsearch"):
            raise ValueError("retrieval_mode must be text or simsearch")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not document_ids or not question.strip():
            return []
        normalized = self.preprocessor.normalize(question)
        method = self._semantic_evidence if retrieval_mode == "simsearch" else self._text_evidence
        return method(document_ids, normalized, top_k)

    def answer(self, document_ids, question, top_k=5, retrieval_mode="text"):
        if retrieval_mode not in ("text", "simsearch"):
            raise ValueError("retrieval_mode must be text or simsearch")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if not document_ids or not question.strip():
            return {"answer": "سند و پرسش را انتخاب کنید.", "evidence": [], "answer_mode": "insufficient_evidence", "warnings": []}
        normalized_question = self.preprocessor.normalize(question)
        evidence = self.retrieve(document_ids, normalized_question, top_k, retrieval_mode)

        if not evidence or not any(is_useful_evidence(item) for item in evidence):
            return {
                "answer": "برای سندهای انتخاب‌شده دانشی در پایگاه داده پیدا نشد.",
                "evidence": evidence,
                "answer_mode": "insufficient_evidence",
                "warnings": [],
            }

        warnings = []
        answer = self._generate_answer(normalized_question, evidence, warnings)
        answer_mode = "generated" if answer else "extractive"
        if answer_mode == "extractive":
            answer = compose_grounded_answer(normalized_question, evidence)
            warnings.append("Showing source excerpts because answer generation was unavailable or returned no answer.")
        return {"answer": answer, "evidence": evidence, "answer_mode": answer_mode, "warnings": warnings}

    def _text_evidence(self, document_ids, normalized_question, top_k):
        chunks = self.knowledge_base.load_chunks(document_ids)
        if not chunks:
            return []

        retriever = BM25Retriever(tokenizer=fast_tokenize)
        retrieval_documents = [
            chunk.get("retrieval_text") or chunk["combined_text"] for chunk in chunks
        ]
        retriever.build_index(retrieval_documents)
        tokenized_query = fast_tokenize(normalized_question)
        ranked = retriever.search_with_scores(normalized_question, top_k)

        evidence = []
        for index, score in ranked[:top_k]:
            chunk = dict(chunks[index])
            chunk["score"] = float(score)
            chunk["match_count"] = len(
                set(tokenized_query) & set(retriever.tokenized_docs[index])
            )
            chunk["retrieval_mode"] = "text"
            evidence.append(chunk)

        return evidence

    def _semantic_evidence(self, document_ids, normalized_question, top_k):
        hits = self.vector_store.search(document_ids, normalized_question, top_k)
        refs = [(hit["document_id"], hit["chunk_index"]) for hit in hits]
        chunks = self.knowledge_base.load_chunks_by_refs(refs)
        chunk_map = {
            (chunk["document_id"], chunk["chunk_index"]): chunk
            for chunk in chunks
        }
        tokenized_query = set(fast_tokenize(normalized_question))
        evidence = []

        for hit in hits:
            key = (hit["document_id"], hit["chunk_index"])
            chunk = chunk_map.get(key)
            if not chunk:
                continue

            chunk = dict(chunk)
            chunk_text = chunk.get("retrieval_text") or chunk["combined_text"]
            chunk["score"] = hit["score"]
            chunk["semantic_score"] = hit["score"]
            chunk["match_count"] = len(tokenized_query & set(fast_tokenize(chunk_text)))
            chunk["retrieval_mode"] = "simsearch"
            evidence.append(chunk)

        return evidence

    def _generate_answer(self, question, evidence, warnings):
        useful = [item for item in evidence if is_useful_evidence(item)]
        if not useful:
            return ""

        try:
            answer = self.answer_generator.generate_answer(question, useful)
            if getattr(self.answer_generator, "last_error", None):
                warnings.append("Answer generation failed. See the application log for details.")
            return answer
        except Exception:
            logging.getLogger(__name__).exception("Answer generation failed")
            warnings.append("Answer generation failed. See the application log for details.")
            return ""


def compose_grounded_answer(question, evidence):
    useful = [item for item in evidence if is_useful_evidence(item)]
    if not useful:
        return (
            "پاسخ روشنی برای این پرسش در سندهای انتخاب‌شده پیدا نشد. "
            "بهتر است سندهای بیشتری انتخاب شود یا پرسش دقیق‌تر نوشته شود."
        )

    snippets = []
    seen = set()
    for item in useful[:3]:
        snippet = normalize_whitespace(
            item.get("display_text") or item["combined_text"]
        )
        snippet = truncate_text(snippet, 420)
        source = format_evidence_location(item)
        key = (snippet, source)
        if key in seen:
            continue
        seen.add(key)
        snippets.append(f"بر اساس {source}: {snippet}")

    return " ".join(snippets)


def is_useful_evidence(item):
    return (
        item.get("match_count", 0) > 0
        or item.get("retrieval_mode") == "simsearch"
        or item.get("semantic_score") is not None
    )


def format_evidence_location(item):
    source = f'{item["file_name"]}، صفحه {item["page_number"]}'
    paragraph_span = item.get("paragraph_span") or item.get("paragraph_index")
    if paragraph_span:
        source += f"، پاراگراف {paragraph_span}"
    return source


def normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def truncate_text(text, max_chars):
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def fast_tokenize(text):
    return re.findall(r"[\w\u0600-\u06FF]+", text.lower())
