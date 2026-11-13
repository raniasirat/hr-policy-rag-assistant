"""Verify answer citations against the chunks retrieved for the same query."""

from __future__ import annotations

import re
import os
from dataclasses import dataclass
from typing import Literal

from src.retriever import RetrievedChunk

_CITATION_RE = re.compile(
    r"\[(?P<source>[^\[\]]+?),\s*(?:"
    r"p\.\s*(?P<page>[^\]\r\n]*)|"
    r"record\s+(?P<record_id>[^\]\r\n]*))\]",
    re.IGNORECASE,
)


def normalize_filename(filename: str) -> str:
    """Normalize paths and casing before comparing cited and retrieved sources."""
    return os.path.basename(filename).strip().casefold()


@dataclass(frozen=True)
class CitationVerification:
    citation: str
    source: str
    page: int | None
    record_id: str | None
    status: Literal["verified", "unverified"]


def verify_citations(answer: str, chunks: list[RetrievedChunk]) -> list[CitationVerification]:
    """Extract page/record citations and verify their metadata against retrieved chunks.

    Malformed locations and chunks without the required metadata remain present as
    ``unverified`` citations. An answer with no recognizable citations returns an
    empty list.
    """
    results: list[CitationVerification] = []
    for match in _CITATION_RE.finditer(answer):
        source = match.group("source").strip()
        page_text = match.group("page")
        record_text = match.group("record_id")
        page: int | None = None
        record_id = record_text.strip() if record_text is not None else None
        valid_location = False

        if page_text is not None:
            page_text = page_text.strip()
            if re.fullmatch(r"\d+", page_text):
                page = int(page_text)
                valid_location = page > 0
        elif record_id:
            valid_location = True

        verified = bool(source and source.lower() != "unknown" and valid_location)
        if verified and page is not None:
            verified = any(
                normalize_filename(chunk.source) == normalize_filename(source)
                and chunk.page is not None
                and chunk.page > 0
                and chunk.page == page
                for chunk in chunks
            )
        elif verified and record_id is not None:
            verified = any(
                normalize_filename(chunk.source) == normalize_filename(source)
                and chunk.record_id is not None
                and chunk.record_id == record_id
                for chunk in chunks
            )

        results.append(
            CitationVerification(
                citation=match.group(0),
                source=source,
                page=page,
                record_id=record_id,
                status="verified" if verified else "unverified",
            )
        )
    return results
