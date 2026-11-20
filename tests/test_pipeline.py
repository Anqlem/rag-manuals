from rag.answer import REFUSAL, build_context, is_refusal, parse_citations
from rag.retrieval import Hit, keyword_tsquery, rrf, search


def _hit(i, doc="a.pdf", page=1):
    return Hit(i, doc, page, page, None, f"text {i}", 0.0)


def test_rrf_rewards_agreement():
    a = [_hit(1), _hit(2), _hit(3)]
    b = [_hit(3), _hit(4), _hit(1)]
    fused = rrf([a, b], k=60)
    assert [h.chunk_id for h in fused[:2]] == [1, 3]


def test_tsquery_keeps_codes_as_phrases():
    q = keyword_tsquery("What does E-07 mean on P1.20?")
    assert "(E <-> 07)" in q and "(P1 <-> 20)" in q and " | " in q
    assert keyword_tsquery("???") is None


def test_citations_map_to_pages():
    hits = [_hit(1, page=3), _hit(2, page=8)]
    cites = parse_citations("Use 79 Nm [2]. Also see [1, 2] and [9].", hits)
    assert [(c["id"], c["page_start"]) for c in cites] == [(1, 3), (2, 8)]


def test_refusal_detection():
    assert is_refusal(REFUSAL)
    assert is_refusal("I don’t know based on the provided manuals.")
    assert not is_refusal("The torque is 79 Nm [1].")


def test_context_marks_sources_as_data():
    ctx = build_context([_hit(1, page=4)])
    assert '<source id="1" doc="a.pdf" pages="4">' in ctx


def test_hybrid_search_finds_fault_code(ingested):
    hits, timings = search("E-07", strategy="section", mode="hybrid", top_k=3)
    assert hits[0].page_start == 7
    assert "search_ms" in timings


def test_ingest_is_idempotent(ingested):
    from rag.ingest import ingest_file

    assert ingest_file(ingested)["skipped"] is True


def test_answer_end_to_end(ingested):
    from rag.answer import answer_question

    ans = answer_question("What torque for the mounting bolts?", "section", "hybrid")
    assert not ans.refused and ans.citations and ans.hits


def test_rerank_mode_reorders_by_reranker(ingested, monkeypatch):
    import rag.retrieval as retrieval

    class Stub:  # prefers chunks mentioning "oil"
        def scores(self, query, passages):
            return [float("oil" in p.lower()) for p in passages]

    monkeypatch.setattr(retrieval, "get_reranker", lambda: Stub())
    hits, timings = search("maintenance schedule", strategy="section", mode="hybrid_rerank", top_k=2)
    assert "oil" in hits[0].text.lower() and "rerank_ms" in timings
