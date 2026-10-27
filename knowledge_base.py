import sqlite3
from pathlib import Path

from config import APP_DATA_DIR


class KnowledgeBase:
    def __init__(self, db_path=None):
        db_path = db_path if db_path is not None else APP_DATA_DIR / "knowledge_base.sqlite3"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL UNIQUE,
                    page_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    error TEXT,
                    vector_status TEXT NOT NULL DEFAULT 'not_indexed',
                    vector_error TEXT,
                    vector_updated_at TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            self._ensure_column(
                conn,
                "documents",
                "source_path",
                "ALTER TABLE documents ADD COLUMN source_path TEXT",
            )
            self._ensure_column(
                conn,
                "documents",
                "vector_status",
                "ALTER TABLE documents ADD COLUMN vector_status TEXT NOT NULL DEFAULT 'not_indexed'",
            )
            self._ensure_column(
                conn,
                "documents",
                "vector_error",
                "ALTER TABLE documents ADD COLUMN vector_error TEXT",
            )
            self._ensure_column(
                conn,
                "documents",
                "vector_updated_at",
                "ALTER TABLE documents ADD COLUMN vector_updated_at TEXT",
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    paragraph_index INTEGER,
                    region_indices TEXT,
                    image_path TEXT NOT NULL,
                    ocr_text TEXT NOT NULL,
                    image_caption TEXT NOT NULL,
                    combined_text TEXT NOT NULL,
                    display_text TEXT NOT NULL,
                    retrieval_text TEXT NOT NULL,
                    heading_text TEXT NOT NULL DEFAULT '',
                    paragraph_span TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                )
                """
            )
            self._ensure_column(
                conn,
                "chunks",
                "paragraph_index",
                "ALTER TABLE chunks ADD COLUMN paragraph_index INTEGER",
            )
            self._ensure_column(
                conn,
                "chunks",
                "region_indices",
                "ALTER TABLE chunks ADD COLUMN region_indices TEXT",
            )
            self._ensure_column(
                conn,
                "chunks",
                "display_text",
                "ALTER TABLE chunks ADD COLUMN display_text TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "chunks",
                "retrieval_text",
                "ALTER TABLE chunks ADD COLUMN retrieval_text TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "chunks",
                "heading_text",
                "ALTER TABLE chunks ADD COLUMN heading_text TEXT NOT NULL DEFAULT ''",
            )
            self._ensure_column(
                conn,
                "chunks",
                "paragraph_span",
                "ALTER TABLE chunks ADD COLUMN paragraph_span TEXT",
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS regions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    region_index INTEGER NOT NULL,
                    region_type TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    crop_path TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    paragraph_index INTEGER,
                    caption TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                )
                """
            )
            self._ensure_column(
                conn,
                "regions",
                "paragraph_index",
                "ALTER TABLE regions ADD COLUMN paragraph_index INTEGER",
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_chunks_document_id
                ON chunks(document_id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_document_chunk
                ON chunks(document_id, chunk_index)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_regions_document_id
                ON regions(document_id)
                """
            )
            conn.execute(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_regions_document_region
                ON regions(document_id, page_number, region_index)
                """
            )

    def _ensure_column(self, conn, table_name, column_name, ddl):
        columns = [
            row["name"]
            for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        ]
        if column_name not in columns:
            conn.execute(ddl)

    def get_document_by_hash(self, file_hash):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            return dict(row) if row else None

    def get_document(self, document_id):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()
            return dict(row) if row else None

    def list_documents(self, ready_only=False):
        query = "SELECT * FROM documents"
        params = ()
        if ready_only:
            query += " WHERE status = ?"
            params = ("ready",)
        query += " ORDER BY created_at DESC"
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]

    def start_document(
        self,
        document_id,
        file_name,
        file_hash,
        source_path=None,
    ):
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id,
                    file_name,
                    file_hash,
                    status,
                    source_path,
                    vector_status,
                    vector_error,
                    vector_updated_at,
                    updated_at
                )
                VALUES (?, ?, ?, 'processing', ?, 'not_indexed', NULL, NULL, CURRENT_TIMESTAMP)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status = 'processing',
                    source_path = excluded.source_path,
                    vector_status = 'not_indexed',
                    vector_error = NULL,
                    vector_updated_at = NULL,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (document_id, file_name, file_hash, source_path),
            )

    def replace_chunks(self, document_id, chunks):
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO chunks (
                    document_id,
                    page_number,
                    chunk_index,
                    paragraph_index,
                    region_indices,
                    image_path,
                    ocr_text,
                    image_caption,
                    combined_text,
                    display_text,
                    retrieval_text,
                    heading_text,
                    paragraph_span
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk["document_id"],
                        chunk["page_number"],
                        chunk["chunk_index"],
                        chunk.get("paragraph_index"),
                        chunk.get("region_indices", "[]"),
                        chunk["image_path"],
                        chunk["ocr_text"],
                        chunk["image_caption"],
                        chunk["combined_text"],
                        chunk["display_text"],
                        chunk["retrieval_text"],
                        chunk.get("heading_text", ""),
                        chunk.get("paragraph_span"),
                    )
                    for chunk in chunks
                ],
            )

    def replace_regions(self, document_id, regions):
        with self.connect() as conn:
            conn.execute("DELETE FROM regions WHERE document_id = ?", (document_id,))
            conn.executemany(
                """
                INSERT INTO regions (
                    document_id,
                    page_number,
                    region_index,
                    region_type,
                    bbox,
                    crop_path,
                    prompt,
                    paragraph_index,
                    caption
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        region["document_id"],
                        region["page_number"],
                        region["region_index"],
                        region["region_type"],
                        region["bbox"],
                        region["crop_path"],
                        region["prompt"],
                        region.get("paragraph_index"),
                        region["caption"],
                    )
                    for region in regions
                ],
            )

    def mark_document_ready(self, document_id, page_count, chunk_count):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'ready',
                    page_count = ?,
                    chunk_count = ?,
                    error = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (page_count, chunk_count, document_id),
            )

    def mark_document_failed(self, document_id, error):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET status = 'failed',
                    error = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (str(error), document_id),
            )

    def mark_vector_indexing(self, document_id):
        self._mark_vector_status(document_id, "indexing")

    def mark_vector_indexed(self, document_id):
        self._mark_vector_status(document_id, "indexed")

    def mark_vector_failed(self, document_id, error):
        self._mark_vector_status(document_id, "failed", error)

    def _mark_vector_status(self, document_id, status, error=None):
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE documents
                SET vector_status = ?,
                    vector_error = ?,
                    vector_updated_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                WHERE document_id = ?
                """,
                (status, str(error) if error else None, document_id),
            )

    def delete_document(self, document_id):
        with self.connect() as conn:
            conn.execute("DELETE FROM regions WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            conn.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))

    def purge_all(self):
        with self.connect() as conn:
            conn.execute("DELETE FROM regions")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute(
                "DELETE FROM sqlite_sequence WHERE name IN ('regions', 'chunks')"
            )

    def load_regions(self, document_id):
        if not document_id:
            return []

        query = """
            SELECT *
            FROM regions
            WHERE document_id = ?
            ORDER BY page_number, region_index
        """
        with self.connect() as conn:
            return [
                dict(row)
                for row in conn.execute(query, (document_id,)).fetchall()
            ]

    def load_chunks(self, document_ids):
        if not document_ids:
            return []

        placeholders = ",".join("?" for _ in document_ids)
        query = f"""
            SELECT
                chunks.*,
                documents.file_name
            FROM chunks
            JOIN documents ON documents.document_id = chunks.document_id
            WHERE chunks.document_id IN ({placeholders})
            ORDER BY documents.file_name, chunks.page_number, chunks.chunk_index
        """
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, document_ids).fetchall()]

    def load_chunks_by_refs(self, refs):
        if not refs:
            return []

        conditions = []
        params = []
        for document_id, chunk_index in refs:
            conditions.append("(chunks.document_id = ? AND chunks.chunk_index = ?)")
            params.extend([document_id, chunk_index])

        query = f"""
            SELECT
                chunks.*,
                documents.file_name
            FROM chunks
            JOIN documents ON documents.document_id = chunks.document_id
            WHERE {" OR ".join(conditions)}
        """
        with self.connect() as conn:
            return [dict(row) for row in conn.execute(query, params).fetchall()]
