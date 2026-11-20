from rag.chunking import chunk_fixed, chunk_sections
from rag.extract import extract_pdf


def test_running_header_and_footer_removed(sample_pdf):
    doc = extract_pdf(sample_pdf)
    text = "\n".join(p.text for p in doc.pages)
    assert "Rev C" not in text
    assert "Page 3 of" not in text


def test_headings_detected(sample_pdf):
    doc = extract_pdf(sample_pdf)
    headings = [line.text for p in doc.pages for line in p.lines if line.is_heading]
    assert "5 Fault codes" in headings
    assert "6.2 Gearbox oil" in headings


def test_fixed_chunks_cover_all_words_with_overlap(sample_pdf):
    doc = extract_pdf(sample_pdf)
    chunks = chunk_fixed(doc, size_tokens=60, overlap_tokens=10)
    all_words = [w for p in doc.pages for w in p.text.split()]
    joined = " ".join(c.text for c in chunks).split()
    assert set(all_words) <= set(joined)
    assert len(joined) > len(all_words)  # overlap repeats some words
    assert all(c.page_start <= c.page_end for c in chunks)


def test_section_chunks_keep_breadcrumb_and_pages(sample_pdf):
    chunks = chunk_sections(extract_pdf(sample_pdf), 500, 50, 40)
    fault = [c for c in chunks if c.section == "5 Fault codes"]
    assert fault and fault[0].page_start == 7 and "E-07" in fault[0].text
    assert all(c.section for c in chunks)
    # a body-less chapter heading is carried forward, not glued to the previous chapter
    maint = next(c for c in chunks if c.section.endswith("6.1 Schedule"))
    assert maint.text.startswith("6 Maintenance")


def test_long_section_is_split():
    from rag.extract import Document, Line, Page

    body = [Line(f"word{i} " * 20, 10, False) for i in range(30)]
    doc = Document("x.pdf", [Page(1, [Line("1 Big", 16, True, True, 1), *body])])
    chunks = chunk_sections(doc, size_tokens=100, overlap_tokens=10, min_tokens=10)
    assert len(chunks) > 3
    assert {c.section for c in chunks} == {"1 Big"}
