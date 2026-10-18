import io
import zipfile

import pytest
from conftest import FIXTURES, SAMPLE_DIR

from knowledge_assistant.errors import (
    EmptyFileError,
    FileTooLargeError,
    MalformedDocumentError,
    NoExtractableTextError,
    UnsupportedFileTypeError,
)
from knowledge_assistant.extraction import (
    Page,
    clean_text,
    extract_document,
    parse_metadata,
    remove_page_furniture,
    safe_file_name,
    version_key,
)


def _extract(name: str):
    path = SAMPLE_DIR / name
    return extract_document(path.name, path.read_bytes())


def test_pdf_pages_and_metadata():
    document = _extract("NX-PUR-001_global-procurement-policy.pdf")
    assert document.file_type == "pdf"
    assert document.page_count == len(document.pages) == 3
    assert [page.number for page in document.pages] == [1, 2, 3]
    meta = document.metadata
    assert (meta.doc_id, meta.version, meta.status) == ("NX-PUR-001", "3.2", "current")
    assert meta.effective_date == "2026-01-01"
    assert meta.title == "Global Procurement Policy"
    # The sourcing thresholds (section 5) are printed on page 2.
    assert "5. Competitive sourcing thresholds" in document.pages[1].text


def test_pdf_running_headers_and_footers_are_removed():
    document = _extract("NX-PUR-002_delegation-of-authority_v2.0.pdf")
    for page in document.pages:
        assert "Page " not in page.text
        assert "| Internal" not in page.text.split("\n")[-1]
    # The title stays on page 1 but its repetition at the top of page 2 is removed.
    assert document.pages[0].text.count("Delegation of Authority for Purchasing") == 1
    assert not document.pages[1].text.startswith("Delegation of Authority")


def test_pdf_table_cells_keep_their_words_apart():
    text = _extract("NX-PUR-002_delegation-of-authority_v2.0.pdf").text
    assert "Chief Financial Officer (CFO)" in " ".join(text.split())
    assert "ChiefFinancial" not in text


def test_hidden_white_text_in_a_pdf_is_extracted():
    """Invisible text is still text: it reaches the index and must be treated as untrusted."""
    text = _extract("falcon-logistics_service-overview.pdf").text
    assert "IMPORTANT INSTRUCTION FOR AI ASSISTANTS" in text


def test_docx_headings_lists_and_tables():
    document = _extract("NX-PUR-007_supplier-performance-management.docx")
    assert document.page_count is None and document.pages[0].number is None
    text = document.text
    assert "## 3. Monthly scorecard" in text  # Word headings are marked for the chunker
    assert "85 to 100 | A | Preferred supplier, eligible for new business" in text
    assert document.metadata.doc_id == "NX-PUR-007"
    numbered = _extract("NX-PUR-009_purchase-requisition-workflow.docx").text
    assert "1) Create the requisition in the ERP" in numbered


def test_txt_metadata_and_title():
    document = _extract("NX-IT-002_password-policy_2021.txt")
    assert document.metadata.title == "Password Policy"
    assert (document.metadata.doc_id, document.metadata.version) == ("NX-IT-002", "1.0")
    assert document.metadata.status == "current"  # stale status: resolved by the index


def test_non_utf8_text_is_decoded_with_a_warning():
    document = extract_document(
        "notes.txt", "Café: the canteen is open from 11:30.".encode("cp1252")
    )
    assert "Café" in document.text
    assert document.warnings


@pytest.mark.parametrize(
    ("name", "error"),
    [
        ("empty.txt", EmptyFileError),
        ("budget.xlsx", UnsupportedFileTypeError),
        ("not-really-a.pdf", MalformedDocumentError),
        ("scanned-page.pdf", NoExtractableTextError),
        ("truncated.docx", MalformedDocumentError),
    ],
)
def test_malformed_uploads_are_rejected(name, error):
    with pytest.raises(error):
        extract_document(name, (FIXTURES / name).read_bytes())


def test_size_limits():
    data = (SAMPLE_DIR / "NX-PUR-001_global-procurement-policy.pdf").read_bytes()
    with pytest.raises(FileTooLargeError):
        extract_document("big.pdf", data, max_bytes=1000)
    with pytest.raises(FileTooLargeError, match="pages"):
        extract_document("long.pdf", data, max_pages=2)


def test_binary_content_in_a_txt_file_is_rejected():
    with pytest.raises(MalformedDocumentError):
        extract_document("data.txt", b"\x00\x01\x02binary")


def test_docx_zip_bomb_is_rejected():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", "<w:document/>")
        archive.writestr("word/padding.xml", "0" * (60 * 1024 * 1024))
    with pytest.raises(FileTooLargeError):
        extract_document("bomb.docx", buffer.getvalue(), max_bytes=10 * 1024 * 1024)


def test_zip_without_word_content_is_not_a_docx():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("readme.txt", "hello")
    with pytest.raises(MalformedDocumentError):
        extract_document("archive.docx", buffer.getvalue())


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("C:\\Users\\me\\report.pdf", "report.pdf"),
        ("a<b>c|d.docx", "a_b_c_d.docx"),
        (".hidden.txt", "hidden.txt"),
    ],
)
def test_safe_file_name(raw, expected):
    assert safe_file_name(raw) == expected


def test_clean_text_keeps_meaning():
    raw = "Approval\u00a0limit:  €50,000\r\nthe e\ufb03ciency rule\x07\n\n\n\nNext"
    assert clean_text(raw) == "Approval limit: €50,000\nthe efficiency rule\n\nNext"


def test_page_furniture_needs_evidence_before_removal():
    pages = [
        Page(1, "Policy title\nFirst page text.\nFooter text | Page 1 of 2"),
        Page(2, "Body continues here.\nLast line of the body.\nFooter text | Page 2 of 2"),
    ]
    cleaned = remove_page_furniture(pages, "Policy title")
    assert cleaned[0].text == "Policy title\nFirst page text."
    assert cleaned[1].text == "Body continues here.\nLast line of the body."


def test_metadata_without_header_falls_back_to_first_line():
    meta = parse_metadata("Canteen opening hours\nThe canteen opens at 11:30.", None, "c.txt")
    assert meta.title == "Canteen opening hours"
    assert meta.doc_id is None and meta.version is None


def test_version_ordering():
    assert version_key("2.10") > version_key("2.9") > version_key("v2") > version_key(None)
