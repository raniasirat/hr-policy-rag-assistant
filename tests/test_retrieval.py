"""Tests: vector indexing, retrieval and the RAG pipeline (offline, no Groq needed).

A deterministic ``HashingEmbeddings`` model and a fake chat model replace the providers,
so these tests exercise the real ChromaDB persistence and the real retrieval
logic but say nothing about BGE-M3 or Groq model quality.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_processor import chunk_documents  # noqa: E402
from src.embeddings import HashingEmbeddings  # noqa: E402
from src.prompts import INSUFFICIENT_CONTEXT_MESSAGE, format_context  # noqa: E402
from src.rag_pipeline import RAGPipeline, RAGPipelineError  # noqa: E402
from src.retriever import RetrievedChunk, Retriever  # noqa: E402
from src.vector_store import VectorStoreManager  # noqa: E402

CORPUS: dict[str, list[str]] = {
    "rag.pdf": [
        "Retrieval augmented generation combines a retriever with a language model to ground answers in documents.",
        "Reranking with a cross encoder improves the ordering of retrieved passages.",
    ],
    "vectors.pdf": [
        "Vector databases store embeddings and use approximate nearest neighbour indexes such as HNSW.",
        "Cosine similarity compares the direction of two embedding vectors.",
    ],
}


def make_chunks(corpus: dict[str, list[str]] = CORPUS) -> list[Document]:
    pages = [
        Document(page_content=text, metadata={"source": source, "page": number, "doc_type": "pdf"})
        for source, texts in corpus.items()
        for number, text in enumerate(texts, start=1)
    ]
    return chunk_documents(pages, chunk_size=500, chunk_overlap=50)


@pytest.fixture()
def store(tmp_path: Path) -> VectorStoreManager:
    return VectorStoreManager(tmp_path / "chroma", HashingEmbeddings(), "test_collection")


# ------------------------------------------------------------------ indexing
def test_indexing_persists_chunks_and_metadata(store: VectorStoreManager) -> None:
    report = store.add_documents(make_chunks())
    assert (report.added, report.skipped_duplicates, report.removed_stale) == (4, 0, 0)
    assert store.count() == 4
    assert store.list_sources() == {"rag.pdf": 2, "vectors.pdf": 2}


def test_index_survives_reopening(tmp_path: Path) -> None:
    VectorStoreManager(tmp_path / "db", HashingEmbeddings(), "persist_test").add_documents(make_chunks())
    reopened = VectorStoreManager(tmp_path / "db", HashingEmbeddings(), "persist_test")
    assert reopened.count() == 4


def test_reindexing_same_documents_adds_no_duplicates(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    report = store.add_documents(make_chunks())
    assert (report.added, report.skipped_duplicates) == (0, 4)
    assert store.count() == 4


def test_changed_file_replaces_its_stale_chunks(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    edited = {**CORPUS, "rag.pdf": ["Completely new text about agentic RAG with tool use."]}
    report = store.add_documents(make_chunks(edited))
    assert report.removed_stale == 2
    assert store.list_sources()["rag.pdf"] == 1
    assert store.count() == 3


def test_add_new_document_to_existing_index(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    report = store.add_documents(make_chunks({"new.pdf": ["A brand new paper about hybrid search."]}))
    assert report.added == 1 and store.count() == 5


def test_chunks_without_required_metadata_are_rejected(store: VectorStoreManager) -> None:
    with pytest.raises(ValueError):
        store.add_documents([Document(page_content="x", metadata={"source": "a.pdf"})])


def test_delete_source_and_reset(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    assert store.delete_source("rag.pdf") == 2
    assert store.count() == 2
    store.reset()
    assert store.count() == 0
    store.add_documents(make_chunks())  # usable again after reset
    assert store.count() == 4


# ----------------------------------------------------------------- retrieval
def test_similarity_search_ranks_relevant_chunk_first(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    hits = store.similarity_search("approximate nearest neighbour HNSW vector databases", k=3)
    assert hits[0][0].metadata["source"] == "vectors.pdf"
    assert hits[0][0].metadata["page"] == 1
    scores = [score for _, score in hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_retriever_returns_metadata_and_respects_top_k(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    retriever = Retriever(store, top_k=2, min_relevance_score=0.0)
    result = retriever.retrieve("cross encoder reranking of retrieved passages")
    assert len(result.chunks) == 2
    best = result.chunks[0]
    assert (best.source, best.page) == ("rag.pdf", 2)
    assert best.citation == "rag.pdf, p. 2"
    assert len(retriever.retrieve("embeddings", top_k=4).chunks) == 4


def test_reranker_reorders_thresholded_chunks_and_preserves_metadata() -> None:
    documents = [
        Document(
            page_content="first passage",
            metadata={"source": "first.pdf", "page": 1, "chunk_id": "chunk-first"},
        ),
        Document(
            page_content="below threshold",
            metadata={"source": "low.pdf", "page": 2, "chunk_id": "chunk-low"},
        ),
        Document(
            page_content="last passage",
            metadata={"source": "last.pdf", "page": 3, "chunk_id": "chunk-last"},
        ),
    ]

    class StubStore:
        requested_k: int | None = None

        def similarity_search(self, query: str, k: int, where: dict[str, Any] | None = None):
            assert query == "question" and where is None
            self.requested_k = k
            return list(zip(documents, [0.9, 0.2, 0.8]))

        def count(self) -> int:
            return len(documents)

    class StubReranker:
        received: list[str] = []
        received_query: str | None = None

        def score(self, query: str, passages: list[str]) -> list[float]:
            self.received_query = query
            self.received = passages
            return [0.1, 0.9]

    store = StubStore()
    reranker = StubReranker()
    result = Retriever(
        store, top_k=1, min_relevance_score=0.3, reranker=reranker
    ).retrieve("question")

    assert store.requested_k == 15
    assert reranker.received_query == "question"
    assert reranker.received == ["first passage", "last passage"]
    assert len(result.chunks) == 1
    best = result.chunks[0]
    assert (best.source, best.page, best.chunk_id) == ("last.pdf", 3, "chunk-last")
    assert best.score == 0.8
    assert best.citation == "last.pdf, p. 3"


def test_reranker_expands_candidate_pool_for_top_k_override() -> None:
    class StubStore:
        requested_k: int | None = None

        def similarity_search(self, query: str, k: int, where: dict[str, Any] | None = None):
            assert query == "question" and where is None
            self.requested_k = k
            return [
                (
                    Document(
                        page_content=str(index),
                        metadata={"source": "policy.pdf", "page": index, "chunk_id": str(index)},
                    ),
                    0.9,
                )
                for index in range(3)
            ]

        def count(self) -> int:
            return 3

    class StubReranker:
        def score(self, query: str, passages: list[str]) -> list[float]:
            assert query == "question"
            return [float(index) for index in range(len(passages))]

    store = StubStore()
    result = Retriever(
        store, top_k=5, min_relevance_score=0.0, reranker=StubReranker()
    ).retrieve("question", top_k=18)

    assert store.requested_k == 18
    assert len(result.chunks) == 3
    assert len(result.chunks) <= 18


def test_retriever_on_empty_index(store: VectorStoreManager) -> None:
    result = Retriever(store).retrieve("anything at all")
    assert result.index_empty and not result.is_sufficient and result.chunks == []


def test_retriever_flags_insufficient_context(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    result = Retriever(store, top_k=3, min_relevance_score=0.3).retrieve("banana smoothie recipe with honey")
    assert not result.is_sufficient and not result.index_empty
    assert result.best_score < 0.3


def test_retriever_validates_arguments(store: VectorStoreManager) -> None:
    with pytest.raises(ValueError):
        Retriever(store, top_k=0)
    with pytest.raises(ValueError):
        Retriever(store, min_relevance_score=1.5)
    with pytest.raises(ValueError):
        Retriever(store).retrieve("   ")


def test_chunk_without_page_has_plain_citation() -> None:
    chunk = RetrievedChunk.from_document(
        Document(page_content="t", metadata={"source": "notes.docx", "doc_type": "docx"}), 0.5
    )
    assert chunk.page is None and chunk.citation == "notes.docx"


def test_format_context_contains_citable_headers() -> None:
    chunks = [
        RetrievedChunk("Alpha text", "a.pdf", 3, "Title A", "pdf", 0.9),
        RetrievedChunk("Beta text", "b.docx", None, None, "docx", 0.8),
    ]
    context = format_context(chunks)
    assert "source: a.pdf | page 3 | title: Title A" in context
    assert "source: b.docx | page n/a" in context
    assert "Alpha text" in context and "Beta text" in context


# ------------------------------------------------------------------ pipeline
class ExplodingLLM(FakeListChatModel):
    """Fails if the pipeline ever calls it."""

    def _call(self, *args: Any, **kwargs: Any) -> str:
        raise AssertionError("LLM must not be called when context is insufficient")


def build_pipeline(store: VectorStoreManager, llm: Any, min_score: float = 0.2) -> RAGPipeline:
    return RAGPipeline(Retriever(store, top_k=3, min_relevance_score=min_score), llm)


def test_pipeline_returns_answer_with_sources_from_metadata(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    llm = FakeListChatModel(responses=["**Answer:** RAG grounds answers [rag.pdf, p. 1]."])
    result = build_pipeline(store, llm).ask("How does retrieval augmented generation ground answers?")
    assert result.grounded and "RAG grounds answers" in result.answer
    assert result.sources and result.sources[0].source == "rag.pdf" and result.sources[0].page == 1
    assert "rag.pdf - p. 1" in result.format_sources()


def test_pipeline_skips_llm_when_context_is_insufficient(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())
    result = build_pipeline(store, ExplodingLLM(responses=["unused"])).ask("banana smoothie recipe with honey")
    assert not result.grounded and result.answer == INSUFFICIENT_CONTEXT_MESSAGE and result.sources == []


def test_pipeline_on_empty_index_explains_how_to_ingest(store: VectorStoreManager) -> None:
    result = build_pipeline(store, ExplodingLLM(responses=["unused"])).ask("What is RAG?")
    assert not result.grounded and "ingest.py" in result.answer


def test_pipeline_wraps_llm_failures(store: VectorStoreManager) -> None:
    store.add_documents(make_chunks())

    class FailingLLM(FakeListChatModel):
        def _call(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("connection refused")

    with pytest.raises(RAGPipelineError, match="generation failed"):
        build_pipeline(store, FailingLLM(responses=["x"])).ask("How does retrieval augmented generation work?")


def test_pipeline_rejects_blank_question(store: VectorStoreManager) -> None:
    with pytest.raises(ValueError):
        build_pipeline(store, FakeListChatModel(responses=["x"])).ask("  ")


# ------------------------------------------------- bundled HR corpus (offline)
HR_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
EMPLOYEES_CSV_PRESENT = (HR_RAW_DIR / "employees.csv").is_file()
needs_employees_csv = pytest.mark.skipif(
    not EMPLOYEES_CSV_PRESENT, reason="data/raw/employees.csv is not present (it is sensitive and optional)"
)


@pytest.fixture()
def hr_store(tmp_path: Path) -> VectorStoreManager:
    from src.document_loader import load_directory
    from src.document_processor import process_documents

    result = load_directory(HR_RAW_DIR)
    assert result.failed_files == [] and len(result.loaded_files) == (6 if EMPLOYEES_CSV_PRESENT else 5)
    store = VectorStoreManager(tmp_path / "hr", HashingEmbeddings(), "hr_test")
    store.add_documents(process_documents(result.documents, 1000, 200))
    return store


def test_hr_corpus_indexes_all_policy_files(hr_store: VectorStoreManager) -> None:
    expected_csv = {"employees.csv"} if EMPLOYEES_CSV_PRESENT else set()
    assert set(hr_store.list_sources()) == expected_csv | {
        "policy_01_annual_leave.pdf",
        "policy_02_remote_work.pdf",
        "policy_03_performance_review.pdf",
        "policy_04_code_of_conduct.pdf",
        "policy_05_training_development.pdf",
    }


@pytest.mark.parametrize(
    ("question", "expected_source"),
    [
        ("How many days of annual leave can be carried over?", "policy_01_annual_leave.pdf"),
        ("Who is eligible for remote work after probation?", "policy_02_remote_work.pdf"),
        ("What is the annual training budget for senior grades?", "policy_05_training_development.pdf"),
    ],
)
def test_hr_policy_questions_hit_the_right_policy(
    hr_store: VectorStoreManager, question: str, expected_source: str
) -> None:
    result = Retriever(hr_store, top_k=5, min_relevance_score=0.0).retrieve(question)
    assert expected_source in {chunk.source for chunk in result.chunks}


@needs_employees_csv
def test_hr_employee_question_retrieves_employee_record(hr_store: VectorStoreManager) -> None:
    result = Retriever(hr_store, top_k=3, min_relevance_score=0.0).retrieve(
        "What is the leave balance of Sara Al-Rashidi?"
    )
    top = result.chunks[0]
    assert (top.source, top.record_id) == ("employees.csv", "EMP001")
    assert top.citation == "employees.csv, record EMP001"


@needs_employees_csv
def test_hr_summary_record_lists_all_managers(hr_store: VectorStoreManager) -> None:
    result = Retriever(hr_store, top_k=5, min_relevance_score=0.0).retrieve("all distinct values of manager")
    summaries = [c for c in result.chunks if c.record_id == "summary-manager"]
    assert summaries, "the manager summary chunk should be retrievable for 'all managers' questions"
    for manager in ("Ahmad Al-Ghamdi", "Dina Al-Sayed", "Faisal Al-Zahrani", "Khalid Mansour", "Rania Khalil", "Salma Nasser"):
        assert manager in summaries[0].content
