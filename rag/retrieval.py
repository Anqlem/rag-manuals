"""Three retrieval modes, all over the same chunks table:

- vector:        cosine similarity on pgvector (HNSW index)
- hybrid:        vector + Postgres full-text search, merged with reciprocal rank fusion
- hybrid_rerank: hybrid candidates re-scored by a cross-encoder / Cohere reranker

Keyword search matters a lot for manuals: fault codes ("E-07"), part numbers and
parameter names are exact strings that embeddings blur together.
"""

import re
import time
from dataclasses import dataclass

from rag.config import get_settings
from rag.db import connect
from rag.embeddings import embed_batched
from rag.rerank import get_reranker

MODES = ("vector", "hybrid", "hybrid_rerank")


@dataclass
class Hit:
    chunk_id: int
    doc: str
    page_start: int
    page_end: int
    section: str | None
    text: str
    score: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_COLUMNS = "c.id, d.name, c.page_start, c.page_end, c.section, c.text"


def _vector_search(conn, query_vec, strategy: str, limit: int) -> list[Hit]:
    rows = conn.execute(
        f"""SELECT {_COLUMNS}, 1 - (c.embedding <=> %s::vector) AS score
            FROM chunks c JOIN documents d ON d.id = c.doc_id
            WHERE c.strategy = %s
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s""",
        (str(query_vec), strategy, str(query_vec), limit),
    ).fetchall()
    return [Hit(*r) for r in rows]


def keyword_tsquery(query: str) -> str | None:
    """OR together the query's terms. websearch_to_tsquery ANDs them, which is
    too strict for natural-language questions; ranking still rewards chunks that
    match more terms. Tokens like 'E-07' are kept as phrases."""
    terms = re.findall(r"[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*", query)
    parts = []
    for t in terms:
        pieces = re.split(r"[-_.]", t)
        parts.append(" <-> ".join(pieces) if len(pieces) > 1 else t)
    return " | ".join(f"({p})" for p in parts) or None


def _keyword_search(conn, query: str, strategy: str, limit: int) -> list[Hit]:
    tsq = keyword_tsquery(query)
    if not tsq:
        return []
    rows = conn.execute(
        f"""SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q) AS score
            FROM chunks c JOIN documents d ON d.id = c.doc_id,
                 to_tsquery('english', %s) q
            WHERE c.strategy = %s AND c.tsv @@ q
            ORDER BY score DESC
            LIMIT %s""",
        (tsq, strategy, limit),
    ).fetchall()
    return [Hit(*r) for r in rows]


def rrf(rankings: list[list[Hit]], k: int) -> list[Hit]:
    """Reciprocal rank fusion: score = sum(1 / (k + rank)). Ignores raw scores,
    which are not comparable between cosine similarity and ts_rank."""
    scores: dict[int, float] = {}
    by_id: dict[int, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id.setdefault(hit.chunk_id, hit)
    fused = sorted(by_id.values(), key=lambda h: scores[h.chunk_id], reverse=True)
    for h in fused:
        h.score = scores[h.chunk_id]
    return fused


def search(
    query: str,
    strategy: str = "section",
    mode: str = "hybrid_rerank",
    top_k: int | None = None,
    query_vec=None,
) -> tuple[list[Hit], dict]:
    """Return (hits, timings_ms). Pass `query_vec` to reuse an embedding across modes."""
    s = get_settings()
    top_k = top_k or s.top_k
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    timings = {}

    t = time.perf_counter()
    if query_vec is None:
        query_vec = embed_batched([query], kind="query")[0]
    timings["embed_ms"] = (time.perf_counter() - t) * 1000

    t = time.perf_counter()
    with connect() as conn:
        vec_hits = _vector_search(conn, query_vec, strategy, s.candidates)
        if mode == "vector":
            hits = vec_hits
        else:
            kw_hits = _keyword_search(conn, query, strategy, s.candidates)
            hits = rrf([vec_hits, kw_hits], s.rrf_k)
    timings["search_ms"] = (time.perf_counter() - t) * 1000

    if mode == "hybrid_rerank":
        reranker = get_reranker()
        if reranker is None:
            raise RuntimeError("mode 'hybrid_rerank' needs RERANKER=cohere or RERANKER=local")
        t = time.perf_counter()
        pool = hits[: s.candidates]
        passages = [f"{h.section}\n{h.text}" if h.section else h.text for h in pool]
        for h, score in zip(pool, reranker.scores(query, passages)):
            h.score = score
        hits = sorted(pool, key=lambda h: h.score, reverse=True)
        timings["rerank_ms"] = (time.perf_counter() - t) * 1000

    return hits[:top_k], timings
