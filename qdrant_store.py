import os
import uuid
import warnings

from qdrant_client import QdrantClient
from qdrant_client.http import models

from embedding_runtime import get_embedding_runtime


DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_COLLECTION = os.getenv("QDRANT_COLLECTION", "uder_grad_chunks")
VECTOR_SIZE = 384


class QdrantChunkStore:
    def __init__(
        self,
        client=None,
        embedder=None,
        collection_name=DEFAULT_COLLECTION,
        url=DEFAULT_QDRANT_URL,
        timeout=2,
    ):
        self._client = client
        self._embedder = embedder
        self._uses_default_embedder = embedder is None
        self.collection_name = collection_name
        self.url = url
        self.timeout = timeout

    @property
    def client(self):
        if self._client is None:
            self._client = QdrantClient(url=self.url, timeout=self.timeout)
        return self._client

    @property
    def embedder(self):
        if self._embedder is None:
            self._embedder = get_embedding_runtime()
        return self._embedder

    def is_available(self):
        try:
            self.client.get_collections()
            return True
        except Exception:
            return False

    def ensure_collection(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                ),
            )
            self._create_payload_indexes()

    def upsert_chunks(self, chunks):
        chunks = [chunk for chunk in chunks if self._chunk_text(chunk)]
        if not chunks:
            return 0

        self.ensure_collection()
        if self._uses_default_embedder:
            from runtime_lifecycle import activate_embedding_runtime
            activate_embedding_runtime()
        vectors = self.embedder.embed_passages(
            [self._chunk_text(chunk) for chunk in chunks]
        )
        points = [
            models.PointStruct(
                id=self._point_id(chunk["document_id"], chunk["chunk_index"]),
                vector=vector,
                payload=self._payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors)
        ]
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search(self, document_ids, query, top_k=5):
        if not document_ids:
            return []

        self.ensure_collection()
        query_vector = self.embedder.embed_query(query)
        result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchAny(any=list(document_ids)),
                    )
                ]
            ),
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )
        return [
            {
                "document_id": point.payload["document_id"],
                "chunk_index": point.payload["chunk_index"],
                "score": float(point.score),
            }
            for point in result.points
        ]

    def delete_document_vectors(self, document_id):
        if not self.client.collection_exists(self.collection_name):
            return

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.Filter(
                must=[
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )

    def purge_collection(self):
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def _create_payload_indexes(self):
        for field_name, schema in (
            ("document_id", models.PayloadSchemaType.KEYWORD),
            ("chunk_index", models.PayloadSchemaType.INTEGER),
        ):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name=field_name,
                        field_schema=schema,
                    )
            except Exception:
                pass

    def _payload(self, chunk):
        return {
            "document_id": chunk["document_id"],
            "chunk_index": chunk["chunk_index"],
            "file_name": chunk.get("file_name", ""),
            "page_number": chunk.get("page_number"),
            "paragraph_index": chunk.get("paragraph_index"),
            "paragraph_span": chunk.get("paragraph_span"),
        }

    def _chunk_text(self, chunk):
        return (
            chunk.get("retrieval_text")
            or chunk.get("combined_text")
            or chunk.get("display_text")
            or ""
        )

    def _point_id(self, document_id, chunk_index):
        value = f"uder-grad:{document_id}:{chunk_index}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, value))
