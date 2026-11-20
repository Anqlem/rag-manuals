"""Retrieval metrics. Pure functions, unit tested in tests/test_metrics.py.

A retrieved chunk counts as relevant when it comes from a gold document and its
page range overlaps one of that document's gold pages. Page-level gold labels
make the eval independent of the chunking strategy, which is what lets us
compare strategies fairly.
"""

import json
import random
from pathlib import Path


def load_questions(path: str | Path) -> list[dict]:
    questions = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        q = json.loads(line)
        for key in ("id", "question", "answerable"):
            if key not in q:
                raise ValueError(f"{path}:{n} missing '{key}'")
        if q["answerable"] and not q.get("gold"):
            raise ValueError(f"{path}:{n} answerable question {q['id']} needs gold pages")
        questions.append(q)
    return questions


def is_relevant(doc: str, page_start: int, page_end: int, gold: list[dict]) -> bool:
    return any(g["doc"] == doc and any(page_start <= p <= page_end for p in g["pages"]) for g in gold)


def first_relevant_rank(hits: list[dict], gold: list[dict]) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if is_relevant(h["doc"], h["page_start"], h["page_end"], gold):
            return rank
    return None


def page_recall(hits: list[dict], gold: list[dict], k: int) -> float:
    """Share of gold pages covered by the top-k chunks (for multi-page answers)."""
    wanted = {(g["doc"], p) for g in gold for p in g["pages"]}
    covered = {
        (h["doc"], p)
        for h in hits[:k]
        for p in range(h["page_start"], h["page_end"] + 1)
        if (h["doc"], p) in wanted
    }
    return len(covered) / len(wanted) if wanted else 0.0


def per_question(hits: list[dict], gold: list[dict]) -> dict:
    rank = first_relevant_rank(hits, gold)
    return {
        "rank": rank,
        "hit@1": float(rank is not None and rank <= 1),
        "hit@3": float(rank is not None and rank <= 3),
        "hit@5": float(rank is not None and rank <= 5),
        "mrr@10": 1.0 / rank if rank and rank <= 10 else 0.0,
        "page_recall@5": page_recall(hits, gold, 5),
    }


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% bootstrap interval of the mean. With ~50 questions, intervals are wide;
    report them so small differences are not over-read."""
    if not xs:
        return 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(mean([rng.choice(xs) for _ in xs]) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]
