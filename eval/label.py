"""Helpers for writing eval questions by hand.

  python -m eval.label find "torque bolt"      keyword search over raw pages, to find gold pages
  python -m eval.label page px200.pdf 12       print one page so you can write a question from it
  python -m eval.label check                   validate eval/questions.jsonl against the corpus

Write the question first, in the words a technician would use, THEN find the page.
Questions copied from the page's own wording make retrieval look better than it is.
"""

import argparse
import sys
from pathlib import Path

from eval.metrics import load_questions
from rag.extract import extract_pdf

CORPUS = Path("data/corpus")


def _docs():
    return {p.name: p for p in sorted(CORPUS.glob("*.pdf"))}


def find(terms: list[str]) -> None:
    terms = [t.lower() for t in terms]
    for name, path in _docs().items():
        for page in extract_pdf(path).pages:
            text = page.text.lower()
            if all(t in text for t in terms):
                i = text.find(terms[0])
                snippet = page.text[max(0, i - 80) : i + 160].replace("\n", " ")
                print(f"{name} p.{page.number}: …{snippet}…")


def show_page(doc: str, number: int) -> None:
    page = extract_pdf(_docs()[doc]).pages[number - 1]
    for line in page.lines:
        print(("## " if line.is_heading else "") + line.text)


def check(path: str) -> None:
    docs = _docs()
    pages = {name: len(extract_pdf(p).pages) for name, p in docs.items()}
    qs = load_questions(path)
    errors = 0
    for q in qs:
        for g in q.get("gold", []):
            if g["doc"] not in pages:
                print(f"{q['id']}: unknown doc {g['doc']}")
                errors += 1
            elif any(not 1 <= p <= pages[g["doc"]] for p in g["pages"]):
                print(f"{q['id']}: page out of range in {g['doc']}")
                errors += 1
    ids = [q["id"] for q in qs]
    if len(ids) != len(set(ids)):
        print("duplicate ids")
        errors += 1
    n_un = sum(not q["answerable"] for q in qs)
    n_h = sum(q.get("origin") == "human" for q in qs)
    print(f"{len(qs)} questions, {n_un} unanswerable, {n_h} written by hand, {errors} errors")
    sys.exit(1 if errors else 0)


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("terms", nargs="+")
    pg = sub.add_parser("page")
    pg.add_argument("doc")
    pg.add_argument("number", type=int)
    c = sub.add_parser("check")
    c.add_argument("--questions", default="eval/questions.jsonl")
    p.add_argument("--corpus", default="data/corpus")
    args = p.parse_args()
    global CORPUS
    CORPUS = Path(args.corpus)
    if args.cmd == "find":
        find(args.terms)
    elif args.cmd == "page":
        show_page(args.doc, args.number)
    else:
        check(args.questions)


if __name__ == "__main__":
    main()
