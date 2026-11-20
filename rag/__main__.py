"""Command line entry point.

python -m rag ingest [dir]          ingest PDFs (default data/corpus)
python -m rag ask "question"        answer with citations
python -m rag search "question"     show retrieved chunks only
python -m rag reset                 drop all tables
"""

import argparse
import json
import logging

from rag.chunking import STRATEGIES
from rag.retrieval import MODES


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)

    i = sub.add_parser("ingest")
    i.add_argument("directory", nargs="?", default="data/corpus")

    for name in ("ask", "search"):
        a = sub.add_parser(name)
        a.add_argument("question")
        a.add_argument("--strategy", choices=STRATEGIES, default="section")
        a.add_argument("--mode", choices=MODES, default="hybrid")
        a.add_argument("--json", action="store_true")

    sub.add_parser("reset")
    args = p.parse_args()

    if args.cmd == "ingest":
        from rag.ingest import ingest_dir

        for stats in ingest_dir(args.directory):
            print(stats)
    elif args.cmd == "reset":
        from rag.db import reset_db

        reset_db()
        print("dropped all tables")
    elif args.cmd == "search":
        from rag.retrieval import search

        hits, timings = search(args.question, args.strategy, args.mode)
        for n, h in enumerate(hits, 1):
            print(f"[{n}] {h.doc} p.{h.page_start}-{h.page_end} score={h.score:.4f} {h.section or ''}")
            print("    " + h.text[:240].replace("\n", " "))
        print(timings)
    else:
        from rag.answer import answer_question

        ans = answer_question(args.question, args.strategy, args.mode)
        if args.json:
            print(json.dumps(ans.to_dict(), indent=2, default=str))
        else:
            print(ans.text, "\n")
            for c in ans.citations:
                print(f"  [{c['id']}] {c['doc']} p.{c['page_start']}-{c['page_end']}")
            print(f"\n  {ans.usage} {ans.timings}")


if __name__ == "__main__":
    main()
