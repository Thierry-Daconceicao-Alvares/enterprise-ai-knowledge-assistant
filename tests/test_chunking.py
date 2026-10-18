import itertools

import pytest
from conftest import SAMPLE_DIR

from knowledge_assistant.chunking import chunk_document, embedding_text, split_sections
from knowledge_assistant.extraction import extract_document

ALL_DOCUMENTS = sorted(path.name for path in SAMPLE_DIR.iterdir())


def _document(name: str):
    path = SAMPLE_DIR / name
    return extract_document(path.name, path.read_bytes())


def _normalised(text: str) -> str:
    return " ".join(text.split())


@pytest.mark.parametrize("name", ALL_DOCUMENTS)
def test_chunks_respect_the_size_limit_and_have_stable_ids(name):
    document = _document(name)
    chunks = chunk_document(document, chunk_size=600, chunk_overlap=100)
    assert chunks
    assert all(len(chunk.text) <= 600 for chunk in chunks)
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.chunk_id == f"{document.doc_key}:{chunk.index:04d}" for chunk in chunks)
    assert chunk_document(document, 600, 100) == chunks  # deterministic


@pytest.mark.parametrize("name", [n for n in ALL_DOCUMENTS if n.endswith(".pdf")])
def test_page_numbers_point_to_the_pages_that_contain_the_text(name):
    """Each chunk's first line is found on its first page and its last line on its last page."""
    document = _document(name)
    pages = {page.number: _normalised(page.text) for page in document.pages}
    for chunk in chunk_document(document, chunk_size=500, chunk_overlap=80):
        lines = chunk.text.split("\n")
        assert chunk.page_start is not None and chunk.page_end is not None
        assert _normalised(lines[0])[:40] in pages[chunk.page_start]
        assert _normalised(lines[-1])[-40:] in pages[chunk.page_end]


def test_a_chunk_spanning_two_pages_has_a_page_range():
    document = _document("NX-PUR-002_delegation-of-authority_v2.0.pdf")
    by_section = {chunk.section: chunk for chunk in chunk_document(document)}
    rules = by_section["5. Rules for approvers"]
    assert (rules.page_start, rules.page_end) == (1, 2)
    assert rules.pages_label == "pp. 1-2"
    assert by_section["2. Approval matrix"].pages_label == "p. 1"


def test_sections_follow_the_heading_numbers():
    document = _document("NX-PUR-008_supplier-compliance-documents.docx")
    sections = [chunk.section for chunk in chunk_document(document)]
    # The numbered list of 15 documents in section 2 is not mistaken for headings.
    assert sections == [
        None,
        "1. Purpose",
        "2. Required documents",
        "3. Expiry alerts",
        "4. Expired mandatory documents",
        "5. Compliance rate",
        "6. Responsibilities",
    ]


def test_long_sections_are_split_with_overlap():
    document = _document("NX-FIN-002_travel-and-expense-policy.pdf")
    chunks = [c for c in chunk_document(document, 300, 120) if c.section == "7. Expense reports"]
    assert len(chunks) >= 3
    for previous, current in itertools.pairwise(chunks):
        # The beginning of each chunk repeats the end of the previous one.
        assert current.text.split("\n")[0] in previous.text


def test_overlap_must_be_smaller_than_half_the_chunk_size():
    with pytest.raises(ValueError):
        chunk_document(_document("NX-HR-004_new-employee-onboarding-checklist.txt"), 400, 300)


def test_wrapped_lines_are_merged_but_list_items_are_not():
    sections = split_sections(
        "1. Rules\nThe first rule is long and\ncontinues on a second line.\n- item one\n- item two"
    )
    units = [unit.text for unit in sections[0].units]
    assert units == [
        "1. Rules",
        "The first rule is long and continues on a second line.",
        "- item one",
        "- item two",
    ]


def test_embedding_text_adds_document_and_section_context():
    document = _document("NX-PUR-004_purchasing-card-policy.docx")
    chunk = next(c for c in chunk_document(document) if c.section == "3. Card limits")
    text = embedding_text(chunk, "Purchasing Card and Low-Value Purchases", "NX-PUR-004", "1.3")
    assert text.startswith(
        "Document: Purchasing Card and Low-Value Purchases (NX-PUR-004 v1.3)\n"
        "Section: 3. Card limits\n\n"
    )
