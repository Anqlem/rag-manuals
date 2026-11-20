"""Tests run fully offline: fake embedder, fake LLM, real Postgres + pgvector.

Set DATABASE_URL to a throwaway database. Database tests are skipped if it is unreachable.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("EMBEDDING_PROVIDER", "fake")
os.environ.setdefault("EMBEDDING_DIM", "256")
os.environ.setdefault("LLM_PROVIDER", "fake")
os.environ.setdefault("RERANKER", "none")

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample" / "px200_manual.pdf"


@pytest.fixture(scope="session")
def sample_pdf() -> Path:
    if not SAMPLE.exists():
        from scripts.make_sample_manual import build

        SAMPLE.parent.mkdir(parents=True, exist_ok=True)
        build(str(SAMPLE))
    return SAMPLE


@pytest.fixture(scope="session")
def ingested(sample_pdf):
    import psycopg

    from rag.config import get_settings

    try:
        psycopg.connect(get_settings().database_url, connect_timeout=3).close()
    except psycopg.OperationalError:
        pytest.skip("Postgres not reachable; set DATABASE_URL")
    from rag.db import reset_db
    from rag.ingest import ingest_dir

    reset_db()
    ingest_dir(sample_pdf.parent)
    return sample_pdf
