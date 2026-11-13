"""Text cleaning and chunking.

Chunks never cross page boundaries: each PDF page is split on its own, so every
chunk carries an exact page number. The trade-off is that a sentence spanning
two pages is cut at the page break.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

# Prefer paragraph, then line, then sentence boundaries before falling back to words.
_SEPARATORS: list[str] = ["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""]

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHENATED_BREAK_RE = re.compile(r"([a-z])-\n([a-z])")
_HORIZONTAL_SPACE_RE = re.compile(r"[ \t ]+")
_SPACE_AROUND_NEWLINE_RE = re.compile(r" ?\n ?")
_SINGLE_NEWLINE_RE = re.compile(r"(?<!\n)\n(?!\n)")
_MANY_NEWLINES_RE = re.compile(r"\n{3,}")


def clean_text(text: str) -> str:
    """Normalise extracted text.

    - Unicode NFKC normalisation (turns ligatures such as "ﬁ" into "fi").
    - Strips control characters.
    - Re-joins words hyphenated across a line break (only lowercase-to-lowercase,
      so genuine hyphens like "State-\\nOf" are less likely to be damaged).
    - Turns single line breaks into spaces but keeps paragraph breaks.
    - Collapses repeated spaces and blank lines.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _CONTROL_CHARS_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHENATED_BREAK_RE.sub(r"\1\2", text)
    text = _HORIZONTAL_SPACE_RE.sub(" ", text)
    text = _SPACE_AROUND_NEWLINE_RE.sub("\n", text)
    text = _SINGLE_NEWLINE_RE.sub(" ", text)
    text = _MANY_NEWLINES_RE.sub("\n\n", text)
    return text.strip()


def clean_documents(documents: list[Document]) -> list[Document]:
    """Clean every document and drop those that become empty."""
    cleaned: list[Document] = []
    for document in documents:
        text = clean_text(document.page_content)
        if text:
            cleaned.append(Document(page_content=text, metadata=dict(document.metadata)))
        else:
            logger.debug("Dropping empty document from %s", document.metadata.get("source"))
    return cleaned


def build_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    """Create the configured ``RecursiveCharacterTextSplitter``.

    Raises:
        ValueError: If the sizes are inconsistent.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
        keep_separator="end",
        length_function=len,
        add_start_index=True,
    )


def make_chunk_id(source: str, page: int | None, chunk_index: int, text: str) -> str:
    """Deterministic chunk id: identical content always maps to the same id."""
    payload = f"{source}|{page if page is not None else ''}|{chunk_index}|{text}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def chunk_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Split documents into chunks, preserving and extending their metadata.

    Each chunk keeps all parent metadata (source, title, page, doc_type, ...) and
    gains ``chunk_index`` (per source file, in reading order), ``start_index``
    (character offset within the page) and a deterministic ``chunk_id``.
    """
    splitter = build_text_splitter(chunk_size, chunk_overlap)
    counters: dict[str, int] = defaultdict(int)
    chunks: list[Document] = []
    for document in documents:
        source = str(document.metadata.get("source", "unknown"))
        page = document.metadata.get("page")
        for piece in splitter.split_documents([document]):
            text = piece.page_content.strip()
            if not text:
                continue
            metadata = dict(piece.metadata)
            metadata["chunk_index"] = counters[source]
            metadata["chunk_id"] = make_chunk_id(source, page, counters[source], text)
            counters[source] += 1
            chunks.append(Document(page_content=text, metadata=metadata))
    return chunks


def process_documents(
    documents: list[Document], chunk_size: int = 1000, chunk_overlap: int = 200
) -> list[Document]:
    """Clean and chunk documents (the full preprocessing step)."""
    chunks = chunk_documents(clean_documents(documents), chunk_size, chunk_overlap)
    logger.info("Created %d chunk(s) from %d document section(s)", len(chunks), len(documents))
    return chunks


def save_processed_chunks(chunks: list[Document], output_path: Path) -> Path:
    """Write chunks as JSON Lines so the processed corpus can be inspected."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {"text": chunk.page_content, "metadata": chunk.metadata}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path
