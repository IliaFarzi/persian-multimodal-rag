import unittest

from qdrant_client import QdrantClient

from qdrant_store import QdrantChunkStore, VECTOR_SIZE


class FixedEmbedder:
    def embed_passages(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)

    def _vector(self, text):
        vector = [0.0] * VECTOR_SIZE
        if "second" in text:
            vector[0] = 0.5
            vector[1] = 0.5
        else:
            vector[0] = 1.0
        return vector


class QdrantChunkStoreTest(unittest.TestCase):
    def test_upserts_filters_ranks_and_deletes_document_vectors(self):
        store = QdrantChunkStore(
            client=QdrantClient(":memory:"),
            embedder=FixedEmbedder(),
            collection_name="test_chunks",
        )
        chunks = [
            {
                "document_id": "doc-1",
                "chunk_index": 0,
                "file_name": "paper.pdf",
                "page_number": 1,
                "paragraph_index": 1,
                "paragraph_span": "1",
                "retrieval_text": "best passage",
            },
            {
                "document_id": "doc-1",
                "chunk_index": 1,
                "file_name": "paper.pdf",
                "page_number": 1,
                "paragraph_index": 2,
                "paragraph_span": "2",
                "retrieval_text": "second passage",
            },
            {
                "document_id": "doc-2",
                "chunk_index": 0,
                "file_name": "other.pdf",
                "page_number": 1,
                "paragraph_index": 1,
                "paragraph_span": "1",
                "retrieval_text": "best passage",
            },
        ]

        self.assertEqual(store.upsert_chunks(chunks), 3)

        hits = store.search(["doc-1"], "best query", top_k=3)

        self.assertEqual([hit["document_id"] for hit in hits], ["doc-1", "doc-1"])
        self.assertEqual(hits[0]["chunk_index"], 0)
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])

        store.delete_document_vectors("doc-1")

        self.assertEqual(store.search(["doc-1"], "best query", top_k=3), [])


if __name__ == "__main__":
    unittest.main()
