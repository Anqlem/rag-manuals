"""Draft extra eval questions with an LLM from random pages.

  python -m eval.draft_questions --n 20 > eval/drafts.jsonl

Every draft is tagged origin="llm". Review and edit each one before moving it into
questions.jsonl, and keep at least half the set human-written: LLM questions reuse
the page's wording, which flatters keyword and vector search alike.
"""

import argparse
import json
import random
from pathlib import Path

from rag.extract import extract_pdf
from rag.llm import complete

PROMPT = """Below is one page of an equipment manual. Write ONE question a field technician
might ask that this page answers, phrased the way a technician would say it (not copying
the page's wording), plus a one-sentence reference answer.
Return only JSON: {"question": "...", "reference_answer": "..."}.
If the page has no useful content (table of contents, legal text), return {"skip": true}."""


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--corpus", default="data/corpus")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    rng = random.Random(args.seed)
    pages = [
        (path.name, page)
        for path in sorted(Path(args.corpus).glob("*.pdf"))
        for page in extract_pdf(path).pages
        if len(page.text) > 400
    ]
    made = 0
    for doc, page in rng.sample(pages, min(len(pages), args.n * 2)):
        if made >= args.n:
            break
        out = complete(PROMPT, page.text[:6000], max_tokens=300).text.strip()
        try:
            data = json.loads(out.removeprefix("```json").removesuffix("```"))
        except json.JSONDecodeError:
            continue
        if data.get("skip"):
            continue
        made += 1
        print(
            json.dumps(
                {
                    "id": f"llm{made:03d}",
                    "question": data["question"],
                    "answerable": True,
                    "gold": [{"doc": doc, "pages": [page.number]}],
                    "reference_answer": data.get("reference_answer"),
                    "origin": "llm",
                }
            )
        )


if __name__ == "__main__":
    main()
