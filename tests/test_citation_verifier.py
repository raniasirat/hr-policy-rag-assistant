from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.citation_verifier import verify_citations  # noqa: E402
from src.retriever import RetrievedChunk  # noqa: E402


def test_verifies_citations_only_for_retrieved_source_and_page() -> None:
    chunks = [RetrievedChunk("policy text", "Employee_Handbook.pdf", 12, None, "pdf", 0.9)]

    citations = verify_citations(
        "[EMPLOYEE_handbook.PDF, p. 12] [Employee_Handbook.pdf, p. 13] [other.pdf, p. 12]",
        chunks,
    )

    assert [citation.status for citation in citations] == ["verified", "unverified", "unverified"]
    assert citations[0].page == 12


def test_invalid_or_missing_page_metadata_is_unverified() -> None:
    chunks = [RetrievedChunk("no page metadata", "handbook.pdf", None, None, "pdf", 0.8)]

    citations = verify_citations("[handbook.pdf, p. 0] [handbook.pdf, p. nope] [handbook.pdf, p. 4]", chunks)

    assert [citation.status for citation in citations] == ["unverified", "unverified", "unverified"]
    assert [citation.page for citation in citations] == [0, None, 4]


def test_verifies_record_citations_and_requires_record_metadata() -> None:
    chunks = [RetrievedChunk("employee row", "employees.csv", None, None, "csv", 0.8, "EMP001")]

    citations = verify_citations(
        "[employees.csv, record EMP001] [employees.csv, record EMP002]",
        chunks,
    )

    assert [citation.status for citation in citations] == ["verified", "unverified"]
    assert citations[0].record_id == "EMP001"


def test_missing_citations_returns_empty_results() -> None:
    assert verify_citations("An answer without a citation.", []) == []


def test_missing_source_metadata_cannot_verify_a_citation() -> None:
    chunks = [RetrievedChunk("text", "unknown", 1, None, "pdf", 0.8)]

    result = verify_citations("[unknown, p. 1]", chunks)

    assert len(result) == 1 and result[0].status == "unverified"
