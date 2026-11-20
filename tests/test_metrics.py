from eval.metrics import bootstrap_ci, page_recall, per_question

GOLD = [{"doc": "a.pdf", "pages": [5, 6]}]


def hit(doc, p0, p1=None):
    return {"doc": doc, "page_start": p0, "page_end": p1 or p0}


def test_rank_and_mrr():
    hits = [hit("a.pdf", 1), hit("b.pdf", 5), hit("a.pdf", 4, 5)]
    m = per_question(hits, GOLD)
    assert m["rank"] == 3
    assert m["hit@1"] == 0 and m["hit@3"] == 1
    assert abs(m["mrr@10"] - 1 / 3) < 1e-9


def test_miss():
    m = per_question([hit("a.pdf", 9)] * 10, GOLD)
    assert m["rank"] is None and m["mrr@10"] == 0 and m["hit@5"] == 0


def test_page_recall_counts_page_ranges():
    assert page_recall([hit("a.pdf", 4, 5)], GOLD, 5) == 0.5
    assert page_recall([hit("a.pdf", 4, 7)], GOLD, 5) == 1.0


def test_bootstrap_ci_brackets_mean():
    xs = [1.0] * 30 + [0.0] * 20
    lo, hi = bootstrap_ci(xs)
    assert lo < 0.6 < hi
