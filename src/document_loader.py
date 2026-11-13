"""Load PDF and DOCX files into LangChain ``Document`` objects.

PDF files yield one ``Document`` per page (so page numbers are preserved).
DOCX files yield a single ``Document`` because the format has no fixed pages.
CSV files (e.g. an employee table) yield one ``Document`` per row, written as
"column: value" text so that it can be embedded and retrieved.

Metadata attached to every document:
    source:      file name (e.g. ``paper.pdf``)
    doc_type:    ``"pdf"``, ``"docx"`` or ``"csv"``
    title:       document title, only when the file declares one
    page:        1-based page number (PDF only)
    total_pages: number of pages in the PDF (PDF only)
    record_id:   value of the first column of the row (CSV only)
    row:         1-based data row number (CSV only)
"""

from __future__ import annotations

import csv
import logging
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import docx2txt
import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx", ".csv"})
_IGNORED_STEMS: frozenset[str] = frozenset({"readme", "license", "licence"})


class DocumentLoadError(Exception):
    """Raised when a single file cannot be loaded."""


@dataclass(frozen=True)
class FailedFile:
    """A file that was skipped, with the reason."""

    path: Path
    reason: str


@dataclass
class LoadResult:
    """Outcome of loading a directory."""

    documents: list[Document] = field(default_factory=list)
    loaded_files: list[Path] = field(default_factory=list)
    failed_files: list[FailedFile] = field(default_factory=list)


_PLACEHOLDER_TITLES: frozenset[str] = frozenset({"", "(anonymous)", "anonymous", "untitled", "none"})
_CONTINUATION_TABLE_MAX_Y = 100.0  # tables starting this close to the page top have no header row


def _table_to_paragraphs(page: pymupdf.Page, table: "pymupdf.table.Table") -> list[str]:
    """Render a table as one paragraph per row: ``Header: value | Header: value``.

    Keeping the column name next to every value stops small language models from
    mixing up columns (for example "Office Days" and "Remote Days"). A table that
    starts at the very top of a page is a continuation of a table from the previous
    page, so its first row is data, not a header: such rows are joined without labels.
    """
    grid: list[list[str]] = []
    for row in table.rows:
        cells = [
            " ".join(page.get_text("text", clip=pymupdf.Rect(cell)).split()) if cell else ""
            for cell in row.cells
        ]
        if any(cells):
            grid.append(cells)
    if not grid:
        return []
    has_header = table.bbox[1] > _CONTINUATION_TABLE_MAX_Y and len(grid) > 1
    if not has_header:
        return [" | ".join(cell for cell in row if cell) for row in grid]
    header, body = grid[0], grid[1:]
    return [
        " | ".join(f"{name}: {value}" if name else value for name, value in zip(header, row) if value)
        for row in body
    ]


def _extract_pdf_page_text(page: pymupdf.Page) -> str:
    """Extract page text: one paragraph per text block, one paragraph per table row.

    Falls back to plain text blocks if table detection fails.
    """
    try:
        tables = page.find_tables().tables
    except Exception as exc:  # table detection is best-effort
        logger.debug("table detection failed: %s", exc)
        tables = []

    items: list[tuple[float, str]] = []
    for table in tables:
        rows = _table_to_paragraphs(page, table)
        if rows:
            items.append((table.bbox[1], "\n\n".join(rows)))
    table_rects = [pymupdf.Rect(table.bbox) for table in tables]

    for block in page.get_text("blocks", sort=True):
        if len(block) < 7 or block[6] != 0 or not block[4].strip():
            continue
        center = pymupdf.Point((block[0] + block[2]) / 2, (block[1] + block[3]) / 2)
        if any(center in rect for rect in table_rects):
            continue  # already rendered as part of a table
        items.append((block[1], block[4].strip()))

    items.sort(key=lambda item: item[0])
    return "\n\n".join(text for _, text in items)


def load_pdf(path: Path) -> list[Document]:
    """Load a PDF with PyMuPDF, one ``Document`` per non-empty page.

    Raises:
        DocumentLoadError: If the file is missing, corrupted, encrypted, or has
            no extractable text (for example a scanned PDF without OCR).
    """
    if not path.is_file():
        raise DocumentLoadError(f"file not found: {path}")
    try:
        pdf = pymupdf.open(path)
    except Exception as exc:  # PyMuPDF raises several unrelated exception types
        raise DocumentLoadError(f"cannot open PDF (corrupted or empty?): {exc}") from exc

    documents: list[Document] = []
    with pdf:
        if pdf.needs_pass:
            raise DocumentLoadError("PDF is password protected")
        title = ((pdf.metadata or {}).get("title") or "").strip()
        if title.lower() in _PLACEHOLDER_TITLES:
            title = ""
        total_pages = pdf.page_count
        for page_index in range(total_pages):
            try:
                text = _extract_pdf_page_text(pdf[page_index])
            except Exception as exc:
                logger.warning("%s: page %d could not be read (%s)", path.name, page_index + 1, exc)
                continue
            if not text.strip():
                logger.debug("%s: page %d has no text", path.name, page_index + 1)
                continue
            metadata: dict[str, str | int] = {
                "source": path.name,
                "doc_type": "pdf",
                "page": page_index + 1,
                "total_pages": total_pages,
            }
            if title:
                metadata["title"] = title
            documents.append(Document(page_content=text, metadata=metadata))

    if not documents:
        raise DocumentLoadError(
            "no extractable text (empty or scanned PDF; OCR is not supported)"
        )
    return documents


def _docx_core_title(path: Path) -> str | None:
    """Read the title from ``docProps/core.xml`` inside a DOCX, if present."""
    try:
        with zipfile.ZipFile(path) as archive:
            with archive.open("docProps/core.xml") as core_xml:
                root = ET.parse(core_xml).getroot()
    except (KeyError, zipfile.BadZipFile, ET.ParseError, OSError):
        return None
    for element in root.iter():
        if element.tag.endswith("}title") and element.text and element.text.strip():
            return element.text.strip()
    return None


def load_docx(path: Path) -> list[Document]:
    """Load a DOCX with docx2txt as a single ``Document`` (no page numbers).

    Raises:
        DocumentLoadError: If the file is missing, corrupted, or empty.
    """
    if not path.is_file():
        raise DocumentLoadError(f"file not found: {path}")
    try:
        text = docx2txt.process(str(path))
    except Exception as exc:  # docx2txt surfaces zipfile / XML errors
        raise DocumentLoadError(f"cannot read DOCX (corrupted?): {exc}") from exc
    if not text or not text.strip():
        raise DocumentLoadError("document contains no text")

    metadata: dict[str, str | int] = {"source": path.name, "doc_type": "docx"}
    title = _docx_core_title(path)
    if title:
        metadata["title"] = title
    return [Document(page_content=text, metadata=metadata)]


def load_csv(path: Path) -> list[Document]:
    """Load a CSV as one ``Document`` per data row.

    The first column is used as the record id (for example ``employee_id``) and
    appears in citations as ``file.csv, record EMP001``.

    Raises:
        DocumentLoadError: If the file is missing, undecodable, or has no data rows.
    """
    if not path.is_file():
        raise DocumentLoadError(f"file not found: {path}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            headers = [h.strip() for h in (reader.fieldnames or []) if h and h.strip()]
            rows = list(reader)
    except (UnicodeDecodeError, csv.Error, OSError) as exc:
        raise DocumentLoadError(f"cannot read CSV (is it UTF-8?): {exc}") from exc
    if not headers:
        raise DocumentLoadError("CSV has no header row")

    kind = path.stem.replace("_", " ")
    documents: list[Document] = []
    for row_number, row in enumerate(rows, start=1):
        fields = [
            (header.replace("_", " "), (row.get(header) or "").strip()) for header in headers
        ]
        fields = [(label, value) for label, value in fields if value]
        if not fields:
            continue
        record_id = fields[0][1]
        text = f"{kind} record {record_id}. " + "; ".join(f"{label}: {value}" for label, value in fields)
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "doc_type": "csv",
                    "record_id": record_id,
                    "row": row_number,
                },
            )
        )
    if not documents:
        raise DocumentLoadError("CSV contains no data rows")
    documents.extend(_csv_summary_documents(path, kind, headers, rows, len(documents)))
    return documents


_SUMMARY_MAX_DISTINCT = 8


def _csv_summary_documents(
    path: Path, kind: str, headers: list[str], rows: list[dict[str, str]], record_count: int
) -> list[Document]:
    """Build one short overview document per low-variety column.

    Retrieval returns only a few rows, so questions such as "list all managers" or
    "how many employees are on probation" cannot be answered from rows alone. For
    every column with few distinct values (e.g. manager, department, status) this
    adds a focused document listing each value and how many records have it. One
    document per column keeps each embedding about a single topic.
    """
    summaries: list[Document] = []
    for header in headers[1:]:
        values = [(row.get(header) or "").strip() for row in rows]
        values = [value for value in values if value]
        counts: dict[str, int] = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        if not 1 < len(counts) <= _SUMMARY_MAX_DISTINCT or len(counts) >= len(values):
            continue
        label = header.replace("_", " ")
        listing = ", ".join(f"{value} ({count})" for value, count in sorted(counts.items()))
        text = (
            f"{kind} summary of {label}. Total records: {record_count}. "
            f"All distinct values of {label} ({len(counts)}), with the number of records: {listing}."
        )
        summaries.append(
            Document(
                page_content=text,
                metadata={
                    "source": path.name,
                    "doc_type": "csv",
                    "record_id": f"summary-{header.strip().lower().replace(' ', '_')}",
                    "row": 0,
                },
            )
        )
    return summaries


def load_file(path: Path) -> list[Document]:
    """Load one supported file, dispatching on its extension.

    Raises:
        DocumentLoadError: If the extension is unsupported or loading fails.
    """
    extension = path.suffix.lower()
    if extension == ".pdf":
        return load_pdf(path)
    if extension == ".docx":
        return load_docx(path)
    if extension == ".csv":
        return load_csv(path)
    raise DocumentLoadError(
        f"unsupported file type '{extension or 'none'}' "
        f"(supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))})"
    )


def _is_ignored(path: Path) -> bool:
    """Hidden files, Office lock files and README/LICENSE notes are not documents."""
    name = path.name
    return name.startswith((".", "~$")) or path.stem.lower() in _IGNORED_STEMS


is_ignored_file = _is_ignored  # public alias for other modules (the web API)


def load_directory(directory: Path) -> LoadResult:
    """Load every supported file in ``directory`` (non-recursive).

    Bad files never abort the run: they are recorded in
    ``LoadResult.failed_files`` and logged. A missing directory yields an empty
    result.
    """
    result = LoadResult()
    if not directory.is_dir():
        logger.warning("Document directory does not exist: %s", directory)
        return result

    for path in sorted(p for p in directory.iterdir() if p.is_file() and not _is_ignored(p)):
        try:
            documents = load_file(path)
        except DocumentLoadError as exc:
            logger.warning("Skipping %s: %s", path.name, exc)
            result.failed_files.append(FailedFile(path=path, reason=str(exc)))
            continue
        result.documents.extend(documents)
        result.loaded_files.append(path)
        logger.info("Loaded %s (%d section(s))", path.name, len(documents))
    return result
