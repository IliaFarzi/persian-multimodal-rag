import unittest
from unittest.mock import patch

from document_app_service import QASessionService


class FakeKnowledgeBase:
    def load_chunks(self, document_ids):
        return [
            {
                "document_id": "doc-1",
                "file_name": "paper.pdf",
                "page_number": 3,
                "chunk_index": 0,
                "paragraph_index": 4,
                "paragraph_span": "4-5",
                "combined_text": "legacy text without the query term",
                "retrieval_text": "File / سند: paper.pdf\n\nrareterm biomarker",
                "display_text": "Clean citation text for the user.",
            }
        ]

    def load_chunks_by_refs(self, refs):
        rows = self.load_chunks(["doc-1"])
        row_map = {
            (row["document_id"], row.get("chunk_index", 0)): row
            for row in rows
        }
        return [row_map[ref] for ref in refs if ref in row_map]


class IdentityPreprocessor:
    def normalize(self, text):
        return text


class FakeAnswerGenerator:
    def __init__(self, answer="", error=None):
        self.answer = answer
        self.error = error
        self.calls = []

    def generate_answer(self, question, evidence):
        self.calls.append((question, evidence))
        if self.error:
            raise self.error
        return self.answer


class FakeVectorStore:
    def search(self, document_ids, question, top_k):
        return [
            {
                "document_id": "doc-1",
                "chunk_index": 0,
                "score": 0.87,
            }
        ]


class QARetrievalTest(unittest.TestCase):
    def setUp(self):
        self.activate_qa = patch("embedding_runtime.E5EmbeddingRuntime.prepare").start()
        self.addCleanup(patch.stopall)

    def test_bm25_indexes_retrieval_text_and_answer_uses_display_text(self):
        service = QASessionService(
            knowledge_base=FakeKnowledgeBase(),
            preprocessor=IdentityPreprocessor(),
            answer_generator=FakeAnswerGenerator(),
        )

        result = service.answer(["doc-1"], "rareterm", top_k=1)

        self.assertEqual(len(result["evidence"]), 1)
        self.assertGreater(result["evidence"][0]["match_count"], 0)
        self.assertIn("Clean citation text", result["answer"])
        self.assertIn("پاراگراف 4-5", result["answer"])
        self.activate_qa.assert_not_called()

    def test_generated_answer_uses_retrieved_evidence(self):
        answer_generator = FakeAnswerGenerator("پاسخ تولیدشده فارسی.")
        service = QASessionService(
            knowledge_base=FakeKnowledgeBase(),
            preprocessor=IdentityPreprocessor(),
            answer_generator=answer_generator,
        )

        result = service.answer(["doc-1"], "rareterm", top_k=1)

        self.assertEqual(result["answer"], "پاسخ تولیدشده فارسی.")
        self.assertEqual(answer_generator.calls[0][0], "rareterm")
        self.assertEqual(answer_generator.calls[0][1][0]["file_name"], "paper.pdf")
        self.assertIn("Clean citation text", answer_generator.calls[0][1][0]["display_text"])
        self.activate_qa.assert_not_called()

    def test_simsearch_evidence_can_generate_without_lexical_match(self):
        answer_generator = FakeAnswerGenerator("پاسخ معنایی.")
        service = QASessionService(
            knowledge_base=FakeKnowledgeBase(),
            preprocessor=IdentityPreprocessor(),
            answer_generator=answer_generator,
            vector_store=FakeVectorStore(),
        )

        result = service.answer(
            ["doc-1"],
            "semantic-only-query",
            top_k=1,
            retrieval_mode="simsearch",
        )

        self.assertEqual(result["answer"], "پاسخ معنایی.")
        self.assertEqual(result["evidence"][0]["retrieval_mode"], "simsearch")
        self.assertEqual(result["evidence"][0]["match_count"], 0)
        self.assertEqual(result["evidence"][0]["semantic_score"], 0.87)
        self.assertEqual(len(answer_generator.calls), 1)
        self.activate_qa.assert_not_called()

    def test_generation_failure_falls_back_to_grounded_answer(self):
        service = QASessionService(
            knowledge_base=FakeKnowledgeBase(),
            preprocessor=IdentityPreprocessor(),
            answer_generator=FakeAnswerGenerator(error=RuntimeError("generation failed")),
        )

        result = service.answer(["doc-1"], "rareterm", top_k=1)

        self.assertIn("Clean citation text", result["answer"])
        self.assertIn("پاراگراف 4-5", result["answer"])
        self.activate_qa.assert_not_called()


if __name__ == "__main__":
    unittest.main()
