from pathlib import Path
import tempfile
import unittest

from knowledge_base import KnowledgeBase


class KnowledgeBaseSchemaTest(unittest.TestCase):
    def test_fresh_schema_has_contextual_chunk_columns_without_versioning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge_base.sqlite3"
            knowledge_base = KnowledgeBase(db_path)

            with knowledge_base.connect() as conn:
                document_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(documents)").fetchall()
                }
                chunk_columns = {
                    row["name"]
                    for row in conn.execute("PRAGMA table_info(chunks)").fetchall()
                }

            self.assertNotIn("ingestion_version", document_columns)
            self.assertTrue(
                {
                    "vector_status",
                    "vector_error",
                    "vector_updated_at",
                }.issubset(document_columns)
            )
            self.assertTrue(
                {
                    "display_text",
                    "retrieval_text",
                    "heading_text",
                    "paragraph_span",
                }.issubset(chunk_columns)
            )

    def test_contextual_chunk_fields_are_persisted_and_loaded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge_base.sqlite3"
            knowledge_base = KnowledgeBase(db_path)
            knowledge_base.start_document(
                "doc-1",
                "paper.pdf",
                "hash-1",
                source_path="/tmp/paper.pdf",
            )
            knowledge_base.replace_chunks(
                "doc-1",
                [
                    {
                        "document_id": "doc-1",
                        "page_number": 2,
                        "chunk_index": 0,
                        "paragraph_index": 9,
                        "paragraph_span": "9-11",
                        "heading_text": "Results",
                        "region_indices": "[3]",
                        "image_path": "/tmp/page.jpg",
                        "ocr_text": "Clean display text",
                        "image_caption": "[توضیح جدول: values]",
                        "display_text": "Results\n\nClean display text",
                        "retrieval_text": "File / سند: paper.pdf\n\nResults",
                        "combined_text": "File / سند: paper.pdf\n\nResults",
                    }
                ],
            )
            knowledge_base.mark_vector_indexing("doc-1")
            knowledge_base.mark_vector_indexed("doc-1")

            rows = knowledge_base.load_chunks(["doc-1"])
            ref_rows = knowledge_base.load_chunks_by_refs([("doc-1", 0)])
            document = knowledge_base.get_document("doc-1")

            self.assertEqual(len(rows), 1)
            self.assertEqual(len(ref_rows), 1)
            self.assertEqual(rows[0]["display_text"], "Results\n\nClean display text")
            self.assertEqual(rows[0]["retrieval_text"], rows[0]["combined_text"])
            self.assertEqual(rows[0]["heading_text"], "Results")
            self.assertEqual(rows[0]["paragraph_span"], "9-11")
            self.assertEqual(ref_rows[0]["paragraph_span"], "9-11")
            self.assertEqual(document["vector_status"], "indexed")

    def test_region_records_are_persisted_and_loaded_in_page_order(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge_base.sqlite3"
            knowledge_base = KnowledgeBase(db_path)
            knowledge_base.start_document("doc-1", "paper.pdf", "hash-1")
            knowledge_base.replace_regions(
                "doc-1",
                [
                    {
                        "document_id": "doc-1",
                        "page_number": 2,
                        "region_index": 1,
                        "region_type": "table",
                        "bbox": "[10, 20, 30, 40]",
                        "crop_path": "/tmp/page2-table.png",
                        "prompt": "Summarize this table.",
                        "paragraph_index": 4,
                        "caption": "page 2 table caption",
                    },
                    {
                        "document_id": "doc-1",
                        "page_number": 1,
                        "region_index": 2,
                        "region_type": "formula",
                        "bbox": "[50, 60, 70, 80]",
                        "crop_path": "/tmp/page1-formula.png",
                        "prompt": "Describe this formula.",
                        "paragraph_index": 3,
                        "caption": "page 1 formula caption",
                    },
                    {
                        "document_id": "doc-1",
                        "page_number": 1,
                        "region_index": 1,
                        "region_type": "image",
                        "bbox": "[1, 2, 3, 4]",
                        "crop_path": "/tmp/page1-image.png",
                        "prompt": "Describe this image.",
                        "paragraph_index": 2,
                        "caption": "page 1 image caption",
                    },
                ],
            )

            rows = knowledge_base.load_regions("doc-1")

            self.assertEqual(
                [(row["page_number"], row["region_index"]) for row in rows],
                [(1, 1), (1, 2), (2, 1)],
            )
            self.assertEqual(rows[0]["crop_path"], "/tmp/page1-image.png")
            self.assertEqual(rows[0]["caption"], "page 1 image caption")
            self.assertEqual(rows[0]["paragraph_index"], 2)
            self.assertEqual(knowledge_base.load_regions(""), [])

    def test_purge_all_clears_documents_chunks_and_regions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "knowledge_base.sqlite3"
            knowledge_base = KnowledgeBase(db_path)
            knowledge_base.start_document("doc-1", "paper.pdf", "hash-1")
            knowledge_base.replace_chunks(
                "doc-1",
                [
                    {
                        "document_id": "doc-1",
                        "page_number": 1,
                        "chunk_index": 0,
                        "paragraph_index": 1,
                        "paragraph_span": "1",
                        "heading_text": "",
                        "region_indices": "[]",
                        "image_path": "/tmp/page.jpg",
                        "ocr_text": "text",
                        "image_caption": "",
                        "display_text": "text",
                        "retrieval_text": "text",
                        "combined_text": "text",
                    }
                ],
            )

            knowledge_base.purge_all()

            self.assertEqual(knowledge_base.list_documents(), [])
            self.assertEqual(knowledge_base.load_chunks(["doc-1"]), [])


if __name__ == "__main__":
    unittest.main()
