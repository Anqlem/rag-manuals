"""PDF text extraction with layout hints.

Keeps per-line font size and boldness so the section-aware chunker can find
headings, and strips running headers/footers (which otherwise show up in every
chunk and pollute both keyword and vector search).
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz

NUMBERED_HEADING = re.compile(r"^(\d+(\.\d+){0,3}|[A-Z]\.\d+|Appendix [A-Z])\s+\S")


@dataclass
class Line:
    text: str
    size: float
    bold: bool
    is_heading: bool = False
    level: int = 0


@dataclass
class Page:
    number: int  # 1-based, matches what a human sees in a PDF viewer
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


@dataclass
class Document:
    name: str
    pages: list[Page]


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", text.strip().lower())


def extract_pdf(path: str | Path) -> Document:
    path = Path(path)
    pages: list[Page] = []
    with fitz.open(path) as pdf:
        for i, pg in enumerate(pdf, start=1):
            page = Page(number=i)
            for block in pg.get_text("dict")["blocks"]:
                for ln in block.get("lines", []):
                    spans = [s for s in ln["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    size = max(s["size"] for s in spans)
                    bold = all(s["flags"] & 16 or "Bold" in s["font"] for s in spans)
                    page.lines.append(Line(text=text, size=round(size, 1), bold=bold))
            pages.append(page)

    _strip_running_lines(pages)
    _mark_headings(pages)
    return Document(name=path.name, pages=pages)


def _strip_running_lines(pages: list[Page]) -> None:
    """Drop short lines repeated on at least half of the pages (headers, footers, page numbers)."""
    if len(pages) < 4:
        return
    counts = Counter()
    for page in pages:
        counts.update({_normalize(line.text) for line in page.lines if len(line.text) < 100})
    threshold = len(pages) / 2
    repeated = {text for text, n in counts.items() if n >= threshold}
    for page in pages:
        page.lines = [line for line in page.lines if _normalize(line.text) not in repeated]


def _mark_headings(pages: list[Page]) -> None:
    sizes = Counter()
    for page in pages:
        for line in page.lines:
            sizes[line.size] += len(line.text)
    if not sizes:
        return
    body = sizes.most_common(1)[0][0]
    heading_sizes = sorted({s for s in sizes if s >= body * 1.15}, reverse=True)
    level_of = {size: i + 1 for i, size in enumerate(heading_sizes)}

    for page in pages:
        for line in page.lines:
            short = 2 <= len(line.text) <= 120 and any(c.isalpha() for c in line.text)
            if not short or line.text.endswith((".", ",", ";")):
                continue
            if line.size in level_of:
                line.is_heading, line.level = True, level_of[line.size]
            elif line.bold and NUMBERED_HEADING.match(line.text):
                line.is_heading, line.level = True, len(heading_sizes) + 1
