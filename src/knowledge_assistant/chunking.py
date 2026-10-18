"""Split an extracted document into sections, then into chunks with overlap.

Design choices:

- Sections first. Numbered headings ("5. Approval thresholds") and Word
  headings start a new section, and a chunk never spans two sections, so every
  citation points to one section of one document.
- Chunks of at most `chunk_size` characters, built from whole lines or
  sentences. A long section is split into several chunks, and each chunk
  repeats the end of the previous one (`chunk_overlap` characters, whole
  sentences when possible) so that a rule cut in two is still readable.
- Page numbers are tracked through character offsets: a chunk that starts on
  page 2 and ends on page 3 is cited as "pages 2-3".
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass

from .extraction import ExtractedDocument

_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(\S.*)$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2})\.\s+([A-Z][^\n]{0,90})$")
_LIST_ITEM = re.compile(r"^(•|-|\*|\d{1,2}[.)])\s+")
_SENTENCE_END = re.compile(r"[.!?:;]$")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9€\"(])")


@dataclass(frozen=True)
class Chunk:
    """A passage of a document, the unit that is embedded and retrieved."""

    chunk_id: str
    doc_key: str
    index: int
    text: str
    section: str | None
    page_start: int | None
    page_end: int | None

    @property
    def pages_label(self) -> str | None:
        if self.page_start is None:
            return None
        if self.page_end is None or self.page_end == self.page_start:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"


@dataclass(frozen=True)
class _Unit:
    """A line, a merged paragraph or a sentence, with its position in the text."""

    text: str
    start: int
    end: int


@dataclass
class _Section:
    title: str | None
    units: list[_Unit]


def _lines_with_offsets(text: str) -> list[tuple[str, int]]:
    lines, offset = [], 0
    for line in text.split("\n"):
        lines.append((line, offset))
        offset += len(line) + 1
    return lines


def _heading_title(line: str, last_number: int) -> tuple[str, int] | None:
    """The section title if the line is a heading, with the new heading number.

    A numbered line is a heading only if its number follows the previous
    heading (1, 2, 3...), so that numbered list items are not mistaken for
    headings."""
    markdown = _MARKDOWN_HEADING.match(line)
    if markdown:
        title = markdown.group(2).strip()
        numbered = _NUMBERED_HEADING.match(title)
        return title, int(numbered.group(1)) if numbered else last_number
    numbered = _NUMBERED_HEADING.match(line)
    if numbered and not _SENTENCE_END.search(line):
        number = int(numbered.group(1))
        if number == last_number + 1:
            return line.strip(), number
    return None


def split_sections(text: str) -> list[_Section]:
    """Group the lines of a document into sections and merge wrapped lines
    into paragraphs (units).

    A line continues the previous unit unless the previous unit ends a
    sentence, the line is blank, or the line starts a list item."""
    sections: list[_Section] = [_Section(title=None, units=[])]
    last_number = 0
    current: _Unit | None = None

    def flush() -> None:
        nonlocal current
        if current is not None:
            sections[-1].units.append(current)
            current = None

    for line, offset in _lines_with_offsets(text):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        heading = _heading_title(stripped, last_number)
        if heading:
            flush()
            title, last_number = heading
            sections.append(_Section(title=title, units=[]))
            start = offset + line.index(stripped)
            sections[-1].units.append(_Unit(title, start, start + len(stripped)))
            continue
        start = offset + line.index(stripped)
        if (
            current is not None
            and not _SENTENCE_END.search(current.text)
            and not _LIST_ITEM.match(stripped)
            and " | " not in stripped  # table rows stay on their own line
            and " | " not in current.text
        ):
            current = _Unit(f"{current.text} {stripped}", current.start, start + len(stripped))
        else:
            flush()
            current = _Unit(stripped, start, start + len(stripped))
    flush()
    return [section for section in sections if section.units]


def _split_long_unit(unit: _Unit, limit: int) -> list[_Unit]:
    """Split a unit longer than the chunk size into sentences, and a sentence
    that is still too long into pieces at word boundaries."""
    pieces: list[_Unit] = []
    position = 0
    for sentence in _SENTENCE_SPLIT.split(unit.text):
        offset = unit.text.index(sentence, position)
        position = offset + len(sentence)
        while len(sentence) > limit:
            cut = sentence.rfind(" ", 0, limit)
            cut = cut if cut > limit // 2 else limit
            pieces.append(_Unit(sentence[:cut], unit.start + offset, unit.start + offset + cut))
            offset += cut
            sentence = sentence[cut:].lstrip()
            offset = unit.text.index(sentence, offset) if sentence else offset
        if sentence:
            pieces.append(_Unit(sentence, unit.start + offset, unit.start + offset + len(sentence)))
    return pieces


def _overlap_units(units: list[_Unit], overlap: int) -> list[_Unit]:
    """The last units of a chunk whose total length fits in the overlap. When
    even the last unit is too long, its last words are used instead."""
    if overlap <= 0 or not units:
        return []
    selected: list[_Unit] = []
    total = 0
    for unit in reversed(units):
        if total + len(unit.text) + 1 > overlap:
            break
        selected.insert(0, unit)
        total += len(unit.text) + 1
    if selected:
        return selected
    last = units[-1]
    cut = last.text.find(" ", len(last.text) - overlap)
    if cut <= 0:
        return []
    tail = last.text[cut + 1 :]
    return [_Unit(tail, last.end - len(tail), last.end)]


def chunk_document(
    document: ExtractedDocument, chunk_size: int = 1000, chunk_overlap: int = 150
) -> list[Chunk]:
    """Split a document into chunks of at most `chunk_size` characters."""
    if chunk_overlap * 2 > chunk_size:
        raise ValueError("chunk_overlap must be at most half of chunk_size")

    page_starts: list[int] = []
    page_numbers: list[int | None] = []
    parts: list[str] = []
    offset = 0
    for page in document.pages:
        if not page.text:
            continue
        page_starts.append(offset)
        page_numbers.append(page.number)
        parts.append(page.text)
        offset += len(page.text) + 2  # the pages are joined with "\n\n"
    text = "\n\n".join(parts)

    def page_at(position: int) -> int | None:
        if not page_starts:
            return None
        return page_numbers[max(0, bisect.bisect_right(page_starts, position) - 1)]

    chunks: list[Chunk] = []

    def emit(units: list[_Unit], section_title: str | None) -> None:
        index = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.doc_key}:{index:04d}",
                doc_key=document.doc_key,
                index=index,
                text="\n".join(unit.text for unit in units),
                section=section_title,
                page_start=page_at(units[0].start),
                page_end=page_at(max(unit.end for unit in units) - 1),
            )
        )

    for section in split_sections(text):
        units: list[_Unit] = []
        for unit in section.units:
            units.extend(
                _split_long_unit(unit, chunk_size) if len(unit.text) > chunk_size else [unit]
            )

        current: list[_Unit] = []
        current_length = 0
        has_new_content = False
        for unit in units:
            added = len(unit.text) + (1 if current else 0)
            if current and has_new_content and current_length + added > chunk_size:
                emit(current, section.title)
                current = _overlap_units(current, chunk_overlap)
                current_length = sum(len(u.text) for u in current) + max(0, len(current) - 1)
                has_new_content = False
                added = len(unit.text) + (1 if current else 0)
                while current and current_length + added > chunk_size:
                    current_length -= len(current.pop(0).text) + 1
            current.append(unit)
            current_length += added
            has_new_content = True
        if current and has_new_content:
            emit(current, section.title)
    return chunks


def embedding_text(chunk: Chunk, title: str, doc_id: str | None, version: str | None) -> str:
    """The text that is embedded: the passage preceded by its document and
    section, so that a passage such as "The limit is €1,500" keeps its context."""
    reference = f" ({doc_id}" + (f" v{version}" if version else "") + ")" if doc_id else ""
    header = f"Document: {title}{reference}"
    if chunk.section:
        header += f"\nSection: {chunk.section}"
    return f"{header}\n\n{chunk.text}"
