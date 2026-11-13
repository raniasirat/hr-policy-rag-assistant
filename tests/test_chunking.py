"""Tests: document loading, text cleaning, chunking, metadata and configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pymupdf
import pytest
from docx import Document as DocxDocument
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import ConfigurationError, Settings, load_settings  # noqa: E402
from src.document_loader import (  # noqa: E402
    DocumentLoadError,
    load_directory,
    load_docx,
    load_file,
    load_pdf,
)
from src.document_processor import (  # noqa: E402
    chunk_documents,
    clean_text,
    make_chunk_id,
    process_documents,
    save_processed_chunks,
)

LONG_TEXT = " ".join(
    f"Sentence number {i} explains how retrieval augmented generation grounds answers in documents."
    for i in range(60)
)


def make_pdf(path: Path, pages: list[str], title: str | None = None) -> Path:
    doc = pymupdf.open()
    if title:
        doc.set_metadata({"title": title})
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    doc.save(path)
    doc.close()
    return path


def make_docx(path: Path, paragraphs: list[str], title: str | None = None) -> Path:
    document = DocxDocument()
    if title:
        document.core_properties.title = title
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)
    document.save(path)
    return path


# ------------------------------------------------------------------ loading
def test_load_pdf_keeps_pages_and_metadata(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "paper.pdf", ["First page text.", "Second page text."], title="My Paper")
    docs = load_pdf(pdf)
    assert [d.metadata["page"] for d in docs] == [1, 2]
    assert all(d.metadata["source"] == "paper.pdf" for d in docs)
    assert all(d.metadata["doc_type"] == "pdf" for d in docs)
    assert all(d.metadata["title"] == "My Paper" for d in docs)
    assert all(d.metadata["total_pages"] == 2 for d in docs)
    assert "First page" in docs[0].page_content


def test_load_pdf_skips_blank_pages_but_keeps_true_page_numbers(tmp_path: Path) -> None:
    pdf = make_pdf(tmp_path / "gap.pdf", ["Page one.", "", "Page three."])
    docs = load_pdf(pdf)
    assert [d.metadata["page"] for d in docs] == [1, 3]


def test_load_pdf_without_title_has_no_title_key(tmp_path: Path) -> None:
    docs = load_pdf(make_pdf(tmp_path / "t.pdf", ["Some text."]))
    assert "title" not in docs[0].metadata


def test_load_docx_metadata(tmp_path: Path) -> None:
    path = make_docx(tmp_path / "notes.docx", ["Hello world.", "Second paragraph."], title="Notes Title")
    docs = load_docx(path)
    assert len(docs) == 1
    assert docs[0].metadata["source"] == "notes.docx"
    assert docs[0].metadata["doc_type"] == "docx"
    assert docs[0].metadata["title"] == "Notes Title"
    assert "page" not in docs[0].metadata
    assert "Second paragraph." in docs[0].page_content


def test_blank_pdf_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="no extractable text"):
        load_pdf(make_pdf(tmp_path / "blank.pdf", [""]))


def test_empty_docx_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError):
        load_docx(make_docx(tmp_path / "empty.docx", []))


def test_corrupted_and_zero_byte_files_are_rejected(tmp_path: Path) -> None:
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"this is not a pdf")
    zero_pdf = tmp_path / "zero.pdf"
    zero_pdf.write_bytes(b"")
    bad_docx = tmp_path / "bad.docx"
    bad_docx.write_bytes(b"this is not a zip")
    for path in (bad_pdf, zero_pdf, bad_docx):
        with pytest.raises(DocumentLoadError):
            load_file(path)


def test_missing_file_and_unsupported_extension(tmp_path: Path) -> None:
    with pytest.raises(DocumentLoadError, match="not found"):
        load_pdf(tmp_path / "missing.pdf")
    txt = tmp_path / "notes.txt"
    txt.write_text("hello")
    with pytest.raises(DocumentLoadError, match="unsupported"):
        load_file(txt)


def test_load_directory_is_graceful(tmp_path: Path) -> None:
    make_pdf(tmp_path / "good.pdf", ["Good PDF text."])
    make_docx(tmp_path / "good.docx", ["Good DOCX text."])
    (tmp_path / "broken.pdf").write_bytes(b"garbage")
    (tmp_path / "notes.txt").write_text("unsupported")
    (tmp_path / "README.md").write_text("ignored silently")
    (tmp_path / ".hidden.pdf").write_bytes(b"ignored")

    result = load_directory(tmp_path)

    assert sorted(p.name for p in result.loaded_files) == ["good.docx", "good.pdf"]
    assert sorted(f.path.name for f in result.failed_files) == ["broken.pdf", "notes.txt"]
    assert len(result.documents) == 2


def test_load_directory_missing_or_empty_folder(tmp_path: Path) -> None:
    assert load_directory(tmp_path / "does_not_exist").documents == []
    empty = tmp_path / "empty"
    empty.mkdir()
    result = load_directory(empty)
    assert result.documents == [] and result.failed_files == []


# ----------------------------------------------------------------- cleaning
def test_clean_text_normalises_whitespace_and_hyphenation() -> None:
    raw = "Retrie-\nval   augmented\ngeneration\t works.\n\n\n\nNew   paragraph.\x00"
    assert clean_text(raw) == "Retrieval augmented generation works.\n\nNew paragraph."


def test_clean_text_expands_ligatures_and_handles_empty() -> None:
    assert clean_text("eﬃcient ﬁne") == "efficient fine"
    assert clean_text("   \n\t ") == ""


# ----------------------------------------------------------------- chunking
def test_chunks_respect_size_and_overlap() -> None:
    doc = Document(page_content=LONG_TEXT, metadata={"source": "a.pdf", "page": 1})
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 3
    assert all(len(c.page_content) <= 300 for c in chunks)
    # consecutive chunks share some text because of the overlap
    assert any(
        set(a.page_content.split()[-3:]) & set(b.page_content.split()[:12])
        for a, b in zip(chunks, chunks[1:])
    )


def test_chunk_metadata_is_preserved_and_extended() -> None:
    meta = {"source": "a.pdf", "page": 4, "title": "T", "doc_type": "pdf", "total_pages": 9}
    chunks = chunk_documents([Document(page_content=LONG_TEXT, metadata=meta)], 300, 60)
    for index, chunk in enumerate(chunks):
        for key, value in meta.items():
            assert chunk.metadata[key] == value
        assert chunk.metadata["chunk_index"] == index
        assert "chunk_id" in chunk.metadata and "start_index" in chunk.metadata
    assert len({c.metadata["chunk_id"] for c in chunks}) == len(chunks)


def test_chunk_ids_are_deterministic() -> None:
    doc = Document(page_content=LONG_TEXT, metadata={"source": "a.pdf", "page": 1})
    first = [c.metadata["chunk_id"] for c in chunk_documents([doc], 300, 60)]
    second = [c.metadata["chunk_id"] for c in chunk_documents([doc], 300, 60)]
    assert first == second
    assert make_chunk_id("a.pdf", 1, 0, "x") != make_chunk_id("a.pdf", 2, 0, "x")


def test_chunks_never_cross_pages() -> None:
    docs = [
        Document(page_content="Alpha content. " * 40, metadata={"source": "a.pdf", "page": 1}),
        Document(page_content="Beta content. " * 40, metadata={"source": "a.pdf", "page": 2}),
    ]
    for chunk in chunk_documents(docs, 200, 40):
        text = chunk.page_content
        assert not ("Alpha" in text and "Beta" in text)
        assert ("Alpha" in text) == (chunk.metadata["page"] == 1)


def test_invalid_chunk_configuration_is_rejected() -> None:
    doc = Document(page_content="text", metadata={"source": "a.pdf"})
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        chunk_documents([doc], chunk_size=0, chunk_overlap=0)


def test_process_documents_drops_empty_and_saves_jsonl(tmp_path: Path) -> None:
    docs = [
        Document(page_content="   \n ", metadata={"source": "empty.pdf", "page": 1}),
        Document(page_content=LONG_TEXT, metadata={"source": "full.pdf", "page": 1}),
    ]
    chunks = process_documents(docs, 300, 60)
    assert chunks and all(c.metadata["source"] == "full.pdf" for c in chunks)
    out = save_processed_chunks(chunks, tmp_path / "processed" / "chunks.jsonl")
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(chunks)
    assert json.loads(lines[0])["metadata"]["source"] == "full.pdf"


# ------------------------------------------------------------ configuration
def test_settings_defaults_match_env_example() -> None:
    settings = Settings()
    assert (settings.embedding_model, settings.groq_model) == ("BAAI/bge-m3", "openai/gpt-oss-120b")
    assert (settings.chunk_size, settings.chunk_overlap, settings.top_k) == (1000, 200, 5)
    assert settings.chroma_db_path.is_absolute()


def test_settings_reject_overlap_not_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        Settings(chunk_size=500, chunk_overlap=500)


def test_load_settings_reads_env_file_and_reports_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "GROQ_MODEL"):
        monkeypatch.delenv(name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("CHUNK_SIZE=800\nCHUNK_OVERLAP=100\nTOP_K=3\nGROQ_MODEL=openai/gpt-oss-120b\n")
    settings = load_settings(env_file)
    assert (settings.chunk_size, settings.chunk_overlap, settings.top_k) == (800, 100, 3)

    monkeypatch.setenv("CHUNK_OVERLAP", "900")
    with pytest.raises(ConfigurationError):
        load_settings(env_file)


# ---------------------------------------------------------------------- CSV
def test_load_csv_one_document_per_row(tmp_path: Path) -> None:
    from src.document_loader import load_csv

    path = tmp_path / "employees.csv"
    path.write_text(
        "employee_id,full_name,leave_balance\nEMP001,Sara Al-Rashidi,18\nEMP002,Omar Benali,\n,,\n",
        encoding="utf-8",
    )
    docs = load_csv(path)
    assert len(docs) == 2  # the fully empty row is skipped
    assert docs[0].metadata["record_id"] == "EMP001" and docs[0].metadata["doc_type"] == "csv"
    assert "full name: Sara Al-Rashidi" in docs[0].page_content
    assert "leave balance: 18" in docs[0].page_content
    assert "leave balance" not in docs[1].page_content  # empty values are omitted


def test_load_csv_rejects_empty_and_header_only(tmp_path: Path) -> None:
    from src.document_loader import load_csv

    (tmp_path / "empty.csv").write_text("")
    (tmp_path / "header.csv").write_text("a,b\n")
    for name in ("empty.csv", "header.csv"):
        with pytest.raises(DocumentLoadError):
            load_csv(tmp_path / name)


# ------------------------------------------------ tables, titles, CSV summary
def test_pdf_table_rows_keep_column_labels() -> None:
    raw_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    page_text = load_pdf(raw_dir / "policy_02_remote_work.pdf")[0].page_content
    assert "Model: Office First | Office Days: 5 days/week | Remote Days: 0" in page_text
    assert "Model: Hybrid Flexible | Office Days: 2 days/week | Remote Days: 3 days/week" in page_text


def test_placeholder_pdf_title_is_ignored(tmp_path: Path) -> None:
    docs = load_pdf(make_pdf(tmp_path / "anon.pdf", ["Some text."], title="(anonymous)"))
    assert "title" not in docs[0].metadata


def test_csv_summary_has_one_focused_document_per_low_variety_column(tmp_path: Path) -> None:
    from src.document_loader import load_csv

    rows = ["id,name,manager,status"] + [
        f"E{i},Person {i},{'Ann' if i % 2 else 'Bob'},{'Active' if i < 5 else 'Probation'}" for i in range(6)
    ]
    path = tmp_path / "staff.csv"
    path.write_text("\n".join(rows), encoding="utf-8")
    docs = load_csv(path)
    summaries = {d.metadata["record_id"]: d.page_content for d in docs if d.metadata["record_id"].startswith("summary-")}
    assert len(docs) == 6 + 2 and set(summaries) == {"summary-manager", "summary-status"}  # 'name' is unique -> skipped
    assert "Ann (3), Bob (3)" in summaries["summary-manager"]
    assert "Active (5), Probation (1)" in summaries["summary-status"]
