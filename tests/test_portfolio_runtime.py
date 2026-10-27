from contextlib import nullcontext
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from diagnostics import model_files_present, setup_issues
from document_app_service import DocumentIngestionService, QASessionService
from knowledge_base import KnowledgeBase
from layout_extraction import PageLayout, LayoutSegment
from utils import rendered_pdf_pages


class IdentityPreprocessor:
    def normalize(self, text):
        return text


class PortfolioRuntimeTest(unittest.TestCase):
    def test_startup_without_cuda_or_models_renders_without_loading_weights(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from config import ROOT
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch("knowledge_base.APP_DATA_DIR", Path(directory)), patch(
            "diagnostics.setup_issues", return_value=["CUDA unavailable", "Model files missing"]
        ), patch("vgen_runtime.VGenRuntime.prepare", side_effect=AssertionError("eager Gemma load")) as vgen, patch(
            "embedding_runtime.E5EmbeddingRuntime.prepare", side_effect=AssertionError("eager E5 load")
        ) as embedding:
            app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
            self.assertEqual(list(app.exception), [])
            self.assertIn("CUDA unavailable", [warning.value for warning in app.warning])
            self.assertFalse(any("Purge" in button.label for button in app.button))
            vgen.assert_not_called()
            embedding.assert_not_called()

    def test_model_check_rejects_missing_weight_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
                (root / name).write_text("{}")
            (root / "model.safetensors.index.json").write_text('{"weight_map":{"a":"part1.safetensors","b":"part2.safetensors"}}')
            (root / "part1.safetensors").write_bytes(b"weights")
            self.assertFalse(model_files_present(root))
            (root / "part2.safetensors").write_bytes(b"weights")
            self.assertTrue(model_files_present(root))

    def test_failed_vector_deletion_notice_survives_the_ui_rerun(self):
        import streamlit as st
        from streamlit.testing.v1 import AppTest
        from config import ROOT
        st.cache_resource.clear()
        self.addCleanup(st.cache_resource.clear)
        with tempfile.TemporaryDirectory() as directory, patch("knowledge_base.APP_DATA_DIR", Path(directory)), patch(
            "diagnostics.setup_issues", return_value=[]
        ):
            kb = KnowledgeBase()
            kb.start_document("doc", "paper.pdf", "hash")
            kb.mark_document_ready("doc", 1, 0)
            store = MagicMock()
            store.delete_document_vectors.side_effect = RuntimeError("probe unavailable")
            with patch("document_app_service.QdrantChunkStore", return_value=store):
                app = AppTest.from_file(str(ROOT / "streamlit_app.py")).run(timeout=30)
                next(button for button in app.button if button.label == "Delete").click().run()
            self.assertEqual(list(app.exception), [])
            self.assertTrue(any("probe unavailable" in warning.value for warning in app.warning))
            self.assertEqual(kb.list_documents(), [])

    def test_vision_setup_accepts_jinja_chat_templates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json"):
                (root / name).write_text("{}")
            (root / "model.safetensors").write_bytes(b"weights")
            self.assertFalse(model_files_present(root, vision=True))
            (root / "chat_template.jinja").write_text("template")
            self.assertTrue(model_files_present(root, vision=True))

    def test_missing_cuda_and_model_have_actionable_setup_messages(self):
        with patch("torch.cuda.is_available", return_value=False), patch("diagnostics.model_files_present", return_value=False):
            issues = setup_issues("qa")
        self.assertTrue(any("NVIDIA CUDA" in issue for issue in issues))
        self.assertTrue(any("scripts.download_models" in issue for issue in issues))

    def test_rendering_storage_is_cleaned_on_success_and_processing_error(self):
        def render(*args, **kwargs):
            path = Path(kwargs["output_folder"]) / "page.jpg"
            path.write_bytes(b"page")
            return [str(path)]
        with patch("utils.convert_from_path", side_effect=render):
            with rendered_pdf_pages("example.pdf") as paths:
                success_path = paths[0]
                self.assertTrue(success_path.exists())
            self.assertFalse(success_path.parent.exists())
            with self.assertRaisesRegex(RuntimeError, "processing"):
                with rendered_pdf_pages("example.pdf") as paths:
                    failure_path = paths[0]
                    raise RuntimeError("processing")
            self.assertFalse(failure_path.parent.exists())

    def test_rendering_storage_is_cleaned_when_converter_fails(self):
        directories = []
        def render(*args, **kwargs):
            directories.append(Path(kwargs["output_folder"]))
            (directories[-1] / "partial.jpg").touch()
            raise RuntimeError("conversion")
        with patch("utils.convert_from_path", side_effect=render), self.assertRaisesRegex(RuntimeError, "conversion"):
            with rendered_pdf_pages("example.pdf"):
                pass
        self.assertFalse(directories[0].exists())

    def test_unavailable_qdrant_preserves_text_document_without_loading_e5(self):
        with tempfile.TemporaryDirectory() as directory:
            kb = KnowledgeBase(Path(directory) / "kb.sqlite3")
            kb.start_document("doc", "paper.pdf", "hash")
            kb.mark_document_ready("doc", 1, 1)
            store = MagicMock()
            store.is_available.return_value = False
            service = DocumentIngestionService(kb, vector_store=store)
            with patch("embedding_runtime.AutoModel.from_pretrained", side_effect=AssertionError("unexpected E5 load")) as load:
                error = service._index_vectors("doc", "paper.pdf", [])
            self.assertIn("text search remains available", error)
            self.assertEqual(kb.get_document("doc")["status"], "ready")
            self.assertEqual(kb.get_document("doc")["vector_status"], "failed")
            store.upsert_chunks.assert_not_called()
            load.assert_not_called()

    def test_caption_ablation_ingests_without_touching_models(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pdf, image = root / "paper.pdf", root / "page.jpg"
            pdf.write_bytes(b"pdf fixture")
            image.write_bytes(b"image fixture")
            page = PageLayout(page_number=1, text_segments=[LayoutSegment(
                kind="text", bbox=(10, 10, 500, 100), text="A substantial scientific paragraph about experimental evidence and model evaluation. " * 3,
                block_num=1, par_num=1, line_num=1,
            )], region_segments=[], regions=[])
            layout = MagicMock()
            layout.__enter__.return_value.extract_page.return_value = page
            kb = KnowledgeBase(root / "kb.sqlite3")
            with patch("document_app_service.UPLOAD_DIR", root / "uploads"), patch(
                "document_app_service.PAGE_IMAGE_DIR", root / "pages"
            ), patch("document_app_service.REGION_DIR", root / "regions"), patch(
                "document_app_service.rendered_pdf_pages", return_value=nullcontext([image])
            ), patch("document_app_service.PDFLayoutExtractor", return_value=layout), patch(
                "vgen_runtime.VGenRuntime.prepare", side_effect=AssertionError("unexpected Gemma")
            ), patch("embedding_runtime.E5EmbeddingRuntime.prepare", side_effect=AssertionError("unexpected E5")):
                result = DocumentIngestionService(kb, preprocessor=IdentityPreprocessor(), include_visual_captions=False, index_vectors=False).ingest_pdf(pdf)
            self.assertEqual(result["document"]["status"], "ready")
            self.assertGreater(result["document"]["chunk_count"], 0)

    def test_empty_question_never_loads_a_generator(self):
        service = QASessionService(knowledge_base=MagicMock(), preprocessor=IdentityPreprocessor())
        with patch("document_app_service.FarsiQAGenerator", side_effect=AssertionError("generator")):
            result = service.answer(["doc"], " ")
        self.assertEqual(result["answer_mode"], "insufficient_evidence")

    def test_generation_failure_is_explicitly_an_extractive_answer(self):
        kb = MagicMock()
        kb.load_chunks.return_value = [{"file_name": "paper.pdf", "page_number": 1, "combined_text": "biomarker evidence", "display_text": "biomarker evidence"}]
        generator = MagicMock()
        generator.generate_answer.side_effect = RuntimeError("GPU failure")
        service = QASessionService(kb, preprocessor=IdentityPreprocessor(), answer_generator=generator)
        with self.assertLogs("document_app_service", level="ERROR"):
            result = service.answer(["doc"], "biomarker")
        self.assertEqual(result["answer_mode"], "extractive")
        self.assertTrue(result["warnings"])
        self.assertIn("biomarker evidence", result["answer"])


if __name__ == "__main__":
    unittest.main()
