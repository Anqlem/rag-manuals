"""Run the evaluation and write markdown reports to eval/results/.

  python -m eval.run_eval retrieval                 # every strategy x mode
  python -m eval.run_eval generation --limit 30     # answers + LLM judge, one config

Both read eval/questions.jsonl unless --questions is given.
"""

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path

from eval.metrics import bootstrap_ci, is_relevant, load_questions, mean, per_question, percentile
from rag.answer import answer_question
from rag.chunking import STRATEGIES
from rag.config import get_settings
from rag.db import connect
from rag.embeddings import Usage, embed_batched
from rag.llm import complete
from rag.retrieval import MODES, search

RESULTS = Path(__file__).parent / "results"
METRICS = ("hit@1", "hit@3", "hit@5", "mrr@10", "page_recall@5")


def _stamp() -> str:
    s = get_settings()
    return (
        f"_Run {datetime.now(UTC):%Y-%m-%d %H:%M} UTC · embeddings "
        f"`{s.embedding_provider}:{s.embedding_model}` · reranker `{s.reranker}` · "
        f"chunk {s.chunk_tokens}/{s.chunk_overlap_tokens} tokens · candidates {s.candidates}_"
    )


def _check_docs(questions: list[dict]) -> None:
    with connect() as conn:
        have = {r[0] for r in conn.execute("SELECT name FROM documents")}
    missing = {g["doc"] for q in questions for g in q.get("gold", [])} - have
    if missing:
        raise SystemExit(f"Gold labels reference documents that are not ingested: {sorted(missing)}")


# ---------------------------------------------------------------- retrieval


def run_retrieval(questions: list[dict], strategies, modes) -> None:
    answerable = [q for q in questions if q["answerable"]]
    _check_docs(answerable)
    print(f"{len(answerable)} answerable questions; embedding queries once")
    vectors = embed_batched([q["question"] for q in answerable], kind="query")

    rows, per_q = [], []
    for strategy in strategies:
        for mode in modes:
            results, latencies = [], []
            for q, vec in zip(answerable, vectors):
                t = time.perf_counter()
                hits, _ = search(q["question"], strategy, mode, top_k=10, query_vec=vec)
                latencies.append((time.perf_counter() - t) * 1000)
                hit_dicts = [h.to_dict() for h in hits]
                m = per_question(hit_dicts, q["gold"])
                results.append(m)
                per_q.append(
                    {
                        "id": q["id"],
                        "strategy": strategy,
                        "mode": mode,
                        **m,
                        "top": [(h["doc"], h["page_start"], h["page_end"]) for h in hit_dicts[:5]],
                    }
                )
            row = {
                "strategy": strategy,
                "mode": mode,
                **{k: mean([r[k] for r in results]) for k in METRICS},
                "hit@5_ci": bootstrap_ci([r["hit@5"] for r in results]),
                "p50_ms": percentile(latencies, 50),
                "p95_ms": percentile(latencies, 95),
            }
            rows.append(row)
            print(f"  {strategy:8s} {mode:14s} hit@5={row['hit@5']:.2f} mrr={row['mrr@10']:.2f}")

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "retrieval_per_question.jsonl").open("w") as f:
        for r in per_q:
            f.write(json.dumps(r) + "\n")
    (RESULTS / "retrieval.md").write_text(_retrieval_report(rows, per_q, answerable))
    print(f"wrote {RESULTS / 'retrieval.md'}")


def _retrieval_report(rows, per_q, questions) -> str:
    best = max(rows, key=lambda r: (r["mrr@10"], r["hit@5"]))
    base = rows[0]
    lines = [
        "# Retrieval results",
        "",
        _stamp(),
        "",
        f"{len(questions)} answerable questions. A hit means a retrieved chunk overlaps a gold page.",
        "",
        "| Chunking | Retrieval | hit@1 | hit@3 | hit@5 (95% CI) | MRR@10 | page recall@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lo, hi = r["hit@5_ci"]
        mark = " **(best)**" if r is best else ""
        lines.append(
            f"| {r['strategy']} | {r['mode']}{mark} | {r['hit@1']:.2f} | {r['hit@3']:.2f} | "
            f"{r['hit@5']:.2f} ({lo:.2f}–{hi:.2f}) | {r['mrr@10']:.2f} | {r['page_recall@5']:.2f} | "
            f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} |"
        )

    # paired comparison: which questions did the best config fix or break vs the baseline?
    def ranks(cfg):
        return {p["id"]: p for p in per_q if p["strategy"] == cfg["strategy"] and p["mode"] == cfg["mode"]}

    b, w = ranks(base), ranks(best)
    fixed = [i for i in b if not b[i]["hit@5"] and w[i]["hit@5"]]
    broke = [i for i in b if b[i]["hit@5"] and not w[i]["hit@5"]]
    lines += [
        "",
        f"## Baseline ({base['strategy']}/{base['mode']}) vs best ({best['strategy']}/{best['mode']})",
        "",
        f"- Fixed at @5: {len(fixed)} {fixed}",
        f"- Broken at @5: {len(broke)} {broke}",
    ]

    misses = [q for q in questions if not w[q["id"]]["hit@5"]]
    lines += [
        "",
        f"## Misses for the best config ({len(misses)})",
        "",
        "Read these before tuning anything. Each one is a failure-analysis candidate.",
        "",
    ]
    for q in misses:
        gold = "; ".join(f"{g['doc']} p.{g['pages']}" for g in q["gold"])
        got = ", ".join(f"{d} p.{a}" + (f"-{z}" if z != a else "") for d, a, z in w[q["id"]]["top"][:3])
        lines.append(f"- **{q['id']}** {q['question']}  \n  gold: {gold} · got: {got}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- generation

JUDGE_SYSTEM = """You are a strict JUDGE grading a RAG system's answer.
Return only JSON: {"faithful": bool, "correct": bool | null, "reason": "<one sentence>"}
- faithful: every factual claim in the ANSWER is supported by the SOURCES. Claims not in the sources make it false.
- correct: the ANSWER agrees with the REFERENCE and answers the question. Use null if no REFERENCE is given."""


def _judge(question: str, answer: str, sources: str, reference: str | None) -> dict:
    s = get_settings()
    user = (
        f"QUESTION:\n{question}\n\nSOURCES:\n{sources}\n\nANSWER:\n{answer}\n\n"
        f"REFERENCE:\n{reference or '(none)'}"
    )
    out = complete(JUDGE_SYSTEM, user, model=s.judge_model or s.llm_model, max_tokens=300)
    text = out.text.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    try:
        verdict = json.loads(text)
    except json.JSONDecodeError:
        verdict = {"faithful": None, "correct": None, "reason": f"unparseable judge output: {text[:120]}"}
    verdict["judge_tokens"] = out.input_tokens + out.output_tokens
    return verdict


def run_generation(questions: list[dict], strategy: str, mode: str, limit: int | None) -> None:
    from rag.answer import build_context

    _check_docs(questions)
    if limit:
        # keep every unanswerable question; they are the cheapest test of hallucination
        unans = [q for q in questions if not q["answerable"]]
        ans = [q for q in questions if q["answerable"]][: max(0, limit - len(unans))]
        questions = ans + unans
    s = get_settings()
    emb_before = Usage.embedding_tokens
    records = []
    for q in questions:
        a = answer_question(q["question"], strategy, mode)
        rec = {
            "id": q["id"],
            "answerable": q["answerable"],
            "question": q["question"],
            "answer": a.text,
            "refused": a.refused,
            "citations": a.citations,
            "usage": a.usage,
            "timings": a.timings,
        }
        if q["answerable"] and not a.refused:
            rec["citation_on_gold"] = any(
                is_relevant(c["doc"], c["page_start"], c["page_end"], q["gold"]) for c in a.citations
            )
            rec["judge"] = _judge(q["question"], a.text, build_context(a.hits), q.get("reference_answer"))
        records.append(rec)
        print(f"  {q['id']}: {'REFUSED' if a.refused else 'answered'}")

    RESULTS.mkdir(exist_ok=True)
    with (RESULTS / "generation_per_question.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    emb_tokens = Usage.embedding_tokens - emb_before
    report = _generation_report(records, strategy, mode, emb_tokens, s)
    (RESULTS / "generation.md").write_text(report)
    print(f"wrote {RESULTS / 'generation.md'}")


def _rate(xs) -> str:
    xs = [x for x in xs if x is not None]
    return f"{mean([float(x) for x in xs]):.2f} (n={len(xs)})" if xs else "n/a"


def _generation_report(records, strategy, mode, emb_tokens, s) -> str:
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    answered = [r for r in ans if not r["refused"]]
    judged = [r["judge"] for r in answered if "judge" in r]
    tokens = [r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in records]
    costs = [r["usage"]["cost_usd"] for r in records]
    total_ms = [sum(r["timings"].values()) for r in records]
    emb_cost = emb_tokens * s.price_embedding / 1e6

    lines = [
        "# Generation results",
        "",
        _stamp(),
        "",
        f"Config: `{strategy}` chunks, `{mode}` retrieval, model `{s.llm_provider}:{s.llm_model}`, "
        f"judge `{s.judge_model or s.llm_model}`, {len(records)} questions.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Answerable questions answered (not refused) | {_rate([not r['refused'] for r in ans])} |",
        f"| Faithfulness (claims supported by sources, judge) | {_rate([j.get('faithful') for j in judged])} |",
        f"| Correctness vs reference (judge) | {_rate([j.get('correct') for j in judged])} |",
        f"| Cites a gold page | {_rate([r.get('citation_on_gold') for r in answered])} |",
        f"| Unanswerable questions correctly refused | {_rate([r['refused'] for r in unans])} |",
        f"| Mean LLM tokens per query | {statistics.mean(tokens):.0f} |" if tokens else "",
        f"| Mean cost per query (LLM + embedding) | ${(sum(costs) + emb_cost) / max(1, len(records)):.5f} |",
        f"| Latency p50 / p95 (ms, end to end) | {percentile(total_ms, 50):.0f} / {percentile(total_ms, 95):.0f} |",
        "",
    ]
    if not s.price_llm_input:
        lines.append(
            "_Costs are 0 because PRICE_* settings are unset. Fill them in from your provider's price page._\n"
        )

    problems = [
        r
        for r in answered
        if r.get("judge", {}).get("faithful") is False or r.get("judge", {}).get("correct") is False
    ]
    problems += [r for r in unans if not r["refused"]]
    problems += [r for r in ans if r["refused"]]
    lines += [f"## Cases to read ({len(problems)})", ""]
    for r in problems:
        why = (
            "answered an unanswerable question"
            if not r["answerable"]
            else "refused an answerable question"
            if r["refused"]
            else r["judge"].get("reason", "")
        )
        lines.append(f"- **{r['id']}** {r['question']}  \n  {why}  \n  > {r['answer'][:300]}")
    return "\n".join(line for line in lines if line is not None) + "\n"


# ---------------------------------------------------------------- cli


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=["retrieval", "generation"])
    p.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--modes", nargs="+", choices=MODES)
    p.add_argument("--strategy", default="section", choices=STRATEGIES)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    questions = load_questions(args.questions)
    has_reranker = get_settings().reranker != "none"
    if args.what == "retrieval":
        modes = args.modes or [m for m in MODES if has_reranker or m != "hybrid_rerank"]
        run_retrieval(questions, args.strategies, modes)
    else:
        mode = args.mode or ("hybrid_rerank" if has_reranker else "hybrid")
        run_generation(questions, args.strategy, mode, args.limit)


if __name__ == "__main__":
    main()
