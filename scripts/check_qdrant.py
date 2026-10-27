"""Test the configured Qdrant server with a temporary, uniquely named collection."""
import argparse
from contextlib import closing
import json
from uuid import uuid4

from qdrant_client import QdrantClient

from qdrant_store import QdrantChunkStore, VECTOR_SIZE, DEFAULT_QDRANT_URL


class ProbeEmbedder:
    def embed_query(self, text):
        return [1.0] + [0.0] * (VECTOR_SIZE - 1)

    def embed_passages(self, texts):
        return [self.embed_query(text) for text in texts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_QDRANT_URL)
    args = parser.parse_args()
    collection = "portfolio_probe_" + uuid4().hex
    with closing(QdrantClient(url=args.url, timeout=10)) as client:
        store = QdrantChunkStore(client=client, embedder=ProbeEmbedder(), collection_name=collection)
        try:
            store.upsert_chunks([{"document_id": doc, "chunk_index": 0, "retrieval_text": "probe", "page_number": 1} for doc in ("first", "second")])
            hits = store.search(["first"], "probe")
            if [hit["document_id"] for hit in hits] != ["first"]:
                raise RuntimeError("Document filtering failed")
            store.delete_document_vectors("first")
            if store.search(["first"], "probe") or not store.search(["second"], "probe"):
                raise RuntimeError("Vector deletion or isolation failed")
            print(json.dumps({"status": "passed", "server_version": client.info().version, "checks": ["upsert", "query_points", "document_filter", "delete", "document_isolation"]}))
        finally:
            if client.collection_exists(collection):
                client.delete_collection(collection)


if __name__ == "__main__":
    main()
