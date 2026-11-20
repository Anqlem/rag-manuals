"""Postgres + pgvector schema and connection helpers.

One `chunks` table holds every chunking strategy side by side, so the eval can
compare strategies against the same documents. Full-text search uses a
generated tsvector column, so hybrid search needs no second datastore.
"""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from rag.config import get_settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id      serial PRIMARY KEY,
    name    text NOT NULL UNIQUE,
    sha256  text NOT NULL,
    pages   int  NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigserial PRIMARY KEY,
    doc_id      int  NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    strategy    text NOT NULL,
    ord         int  NOT NULL,
    page_start  int  NOT NULL,
    page_end    int  NOT NULL,
    section     text,
    text        text NOT NULL,
    tokens      int  NOT NULL,
    embedding   vector({dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(section, '') || ' ' || text)
                ) STORED
);

CREATE INDEX IF NOT EXISTS chunks_strategy_idx ON chunks (strategy);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


@contextmanager
def connect():
    conn = psycopg.connect(get_settings().database_url, autocommit=False)
    try:
        register_vector(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create the schema. Refuses to mix embedding models with different dimensions."""
    s = get_settings()
    with psycopg.connect(s.database_url, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    with connect() as conn:
        conn.execute(SCHEMA.format(dim=s.embedding_dim))
        row = conn.execute("SELECT value FROM meta WHERE key = 'embedding'").fetchone()
        current = f"{s.embedding_provider}:{s.embedding_model}:{s.embedding_dim}"
        if row is None:
            conn.execute("INSERT INTO meta VALUES ('embedding', %s)", (current,))
        elif row[0] != current:
            raise SystemExit(
                f"Database was built with embeddings '{row[0]}' but config says '{current}'. "
                "Run `python -m rag reset` and ingest again."
            )


def reset_db() -> None:
    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks, documents, meta CASCADE")
