from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from document_app_service import DocumentIngestionService
from knowledge_base import KnowledgeBase


class FakeVectorStore:
    def __init__(self):
        self.purged = False

    def purge_collection(self):
        self.purged = True


class DevPurgeTest(unittest.TestCase):
    def test_purge_clears_sqlite_files_and_qdrant_collection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            upload_dir = root / "uploads"
            page_image_dir = root / "page_images"
            region_dir = root / "regions"
            upload_dir.mkdir()
            page_image_dir.mkdir()
            region_dir.mkdir()
            (upload_dir / "paper.pdf").write_text("pdf")
            (page_image_dir / "page.jpg").write_text("image")
            (region_dir / "crop.png").write_text("crop")

            knowledge_base = KnowledgeBase(root / "knowledge_base.sqlite3")
            knowledge_base.start_document("doc-1", "paper.pdf", "hash-1")
            vector_store = FakeVectorStore()

            with patch("document_app_service.UPLOAD_DIR", upload_dir), patch(
                "document_app_service.PAGE_IMAGE_DIR", page_image_dir
            ), patch("document_app_service.REGION_DIR", region_dir):
                service = DocumentIngestionService(
                    knowledge_base=knowledge_base,
                    vector_store=vector_store,
                )
                result = service.purge_dev_data()

            self.assertTrue(result["purged"])
            self.assertTrue(vector_store.purged)
            self.assertEqual(knowledge_base.list_documents(), [])
            self.assertEqual(list(upload_dir.iterdir()), [])
            self.assertEqual(list(page_image_dir.iterdir()), [])
            self.assertEqual(list(region_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
