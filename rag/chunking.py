"""Two chunking strategies to compare in the eval.

- fixed:   sliding window of ~N tokens with overlap, blind to document structure.
- section: split at detected headings, keep the heading breadcrumb with each
           chunk, split long sections with the same sliding window, and merge
           tiny sections forward so a lone heading never becomes a chunk.

Tokens are approximated as characters / 4. That is close enough for sizing
chunks and avoids a tokenizer download; swap in tiktoken if you need exact counts.
"""

from dataclasses import dataclass

from rag.extract import Document

STRATEGIES = ("fixed", "section")


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class Chunk:
    ord: int
    text: str
    page_start: int
    page_end: int
    section: str | None
    tokens: int

    @property
    def embed_text(self) -> str:
        """What gets embedded. Section chunks carry their heading path as context."""
        return f"{self.section}\n{self.text}" if self.section else self.text


def _windows(words: list[tuple[str, int]], size_tokens: int, overlap_tokens: int):
    """Yield (text, first_page, last_page) windows over (word, page) pairs."""
    size_chars, overlap_chars = size_tokens * 4, overlap_tokens * 4
    start = 0
    while start < len(words):
        end, chars = start, 0
        while end < len(words) and (chars < size_chars or end == start):
            chars += len(words[end][0]) + 1
            end += 1
        window = words[start:end]
        yield " ".join(w for w, _ in window), window[0][1], window[-1][1]
        if end >= len(words):
            break
        # step back far enough to keep ~overlap_chars of context, but always advance
        back, kept = end, 0
        while back > start + 1 and kept < overlap_chars:
            back -= 1
            kept += len(words[back][0]) + 1
        start = back


def chunk_fixed(doc: Document, size_tokens: int, overlap_tokens: int) -> list[Chunk]:
    words = [(w, page.number) for page in doc.pages for w in page.text.split()]
    return [
        Chunk(i, text, p0, p1, None, approx_tokens(text))
        for i, (text, p0, p1) in enumerate(_windows(words, size_tokens, overlap_tokens))
    ]


def chunk_sections(doc: Document, size_tokens: int, overlap_tokens: int, min_tokens: int) -> list[Chunk]:
    sections: list[tuple[str, list[tuple[str, int]], bool]] = []  # (path, words, has_body)
    stack: list[tuple[int, str]] = []  # (level, heading)
    current: list[tuple[str, int]] = []
    has_body = False
    current_path = "(front matter)"

    def flush():
        if current:
            sections.append((current_path, list(current), has_body))

    for page in doc.pages:
        for line in page.lines:
            if line.is_heading:
                flush()
                current.clear()
                has_body = False
                while stack and stack[-1][0] >= line.level:
                    stack.pop()
                stack.append((line.level, line.text))
                current_path = " > ".join(h for _, h in stack)
            else:
                has_body = True
            # headings stay in the text too, so merged sections keep their titles
            current.extend((w, page.number) for w in line.text.split())
    flush()

    def size(words):
        return approx_tokens(" ".join(w for w, _ in words))

    # A heading with no body (e.g. a chapter title right before its first
    # subsection) is carried into the next section. A tiny section with a body
    # is merged into the previous one so it keeps some context.
    merged: list[tuple[str, list[tuple[str, int]]]] = []
    carry: list[tuple[str, int]] = []
    for path, words, body in sections:
        words = carry + words
        carry = []
        if not body:
            carry = words
        elif size(words) >= min_tokens:
            merged.append((path, words))
        elif merged:
            prev_path, prev_words = merged[-1]
            merged[-1] = (prev_path, prev_words + words)
        else:
            carry = words
    if carry:
        merged.append(("(front matter)", carry))

    chunks: list[Chunk] = []
    for path, words in merged:
        for text, p0, p1 in _windows(words, size_tokens, overlap_tokens):
            chunks.append(Chunk(len(chunks), text, p0, p1, path, approx_tokens(text)))
    return chunks


def chunk_document(doc: Document, strategy: str, settings) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed(doc, settings.chunk_tokens, settings.chunk_overlap_tokens)
    if strategy == "section":
        return chunk_sections(
            doc, settings.chunk_tokens, settings.chunk_overlap_tokens, settings.section_min_tokens
        )
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
