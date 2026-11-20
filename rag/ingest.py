"""Ingest PDFs: extract -> chunk (every strategy) -> embed -> store.

Idempotent: a document whose hash is unchanged is skipped; a changed document
is re-ingested from scratch.
"""

import hashlib
import logging
from pathlib import Path

from rag.chunking import STRATEGIES, chunk_document
from rag.config import get_settings
from rag.db import connect, init_db
from rag.embeddings import embed_batched
from rag.extract import extract_pdf

log = logging.getLogger(__name__)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def ingest_file(path: Path, strategies=STRATEGIES) -> dict:
    s = get_settings()
    digest = _sha256(path)
    with connect() as conn:
        row = conn.execute("SELECT id, sha256 FROM documents WHERE name = %s", (path.name,)).fetchone()
        if row and row[1] == digest:
            have = {
                r[0]
                for r in conn.execute("SELECT DISTINCT strategy FROM chunks WHERE doc_id = %s", (row[0],))
            }
            if set(strategies) <= have:
                log.info("skip %s (unchanged)", path.name)
                return {"doc": path.name, "skipped": True}
        if row:
            conn.execute("DELETE FROM documents WHERE id = %s", (row[0],))

    doc = extract_pdf(path)
    stats = {"doc": path.name, "pages": len(doc.pages), "chunks": {}}
    if not any(p.lines for p in doc.pages):
        log.warning("%s has no extractable text (scanned PDF?); needs OCR first", path.name)

    with connect() as conn:
        doc_id = conn.execute(
            "INSERT INTO documents (name, sha256, pages) VALUES (%s, %s, %s) RETURNING id",
            (path.name, digest, len(doc.pages)),
        ).fetchone()[0]
        for strategy in strategies:
            chunks = chunk_document(doc, strategy, s)
            vectors = embed_batched([c.embed_text for c in chunks], kind="document")
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO chunks (doc_id, strategy, ord, page_start, page_end,
                                           section, text, tokens, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)""",
                    [
                        (
                            doc_id,
                            strategy,
                            c.ord,
                            c.page_start,
                            c.page_end,
                            c.section,
                            c.text,
                            c.tokens,
                            str(v),
                        )
                        for c, v in zip(chunks, vectors)
                    ],
                )
            stats["chunks"][strategy] = len(chunks)
    log.info("ingested %s: %s", path.name, stats["chunks"])
    return stats


def ingest_dir(directory: str | Path) -> list[dict]:
    init_db()
    files = sorted(Path(directory).glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs in {directory}. Put your manuals there first.")
    return [ingest_file(f) for f in files]
