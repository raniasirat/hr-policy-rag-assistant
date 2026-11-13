"""API tests for the FastAPI backend.

LEGEND - what is real and what is replaced:
  REAL:   FastAPI app + routes, request validation, the project's real loader / processor /
          VectorStoreManager (persistent ChromaDB in a temp dir) / Retriever / RAGPipeline /
          prompts, and the access-control filter.
  MOCKED: the embedding model (``HashingEmbeddings``, lexical, offline) and the chat model
          (``FakeListChatModel``). These tests therefore prove the wiring and the security
          rules, NOT the quality of BGE-M3 or Groq. Provider behavior is covered in
          ``tests/test_providers.py`` with mocks.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import httpx
import pymupdf
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.dependencies import ApiSettings, AppServices, build_services  # noqa: E402
from backend.main import create_app  # noqa: E402
from backend.routes.chat import to_chat_response  # noqa: E402
from backend.routes.documents import sanitize_filename  # noqa: E402
from src.config import Settings  # noqa: E402
from src.embeddings import HashingEmbeddings  # noqa: E402
from src.rag_pipeline import AnswerResult  # noqa: E402
from src.retriever import RetrievalResult, RetrievedChunk  # noqa: E402

ADMIN_KEY = "correct-horse-battery"
LEAVE_TEXT = (
    "Annual leave policy. Employees can carry over at most five unused annual leave days "
    "into the next calendar year. Carry over requests need manager approval."
)
REMOTE_TEXT = (
    "Remote work policy. Remote work requires six months of service and no probationary "
    "period. The Office First model means five office days and zero remote days."
)
EMPLOYEE_CSV = (
    "employee_id,full_name,department,manager,leave_balance,employment_status\n"
    "EMP001,Sara Al-Rashidi,Data,Khalid Mansour,18,Active\n"
    "EMP002,Omar Benali,Data,Khalid Mansour,17,Active\n"
    "EMP003,Lina Haddad,Engineering,Rania Khalil,8,Probation\n"
)


def make_pdf_bytes(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 545, 790), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


class ExplodingEmbeddings(Embeddings):
    """MOCK that behaves like local embedding generation failing."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise ConnectionError("local embedding failure")

    def embed_query(self, text: str) -> list[float]:
        raise ConnectionError("local embedding failure")


def build_test_services(
    tmp_path: Path,
    *,
    admin_key: str = ADMIN_KEY,
    llm: Any = None,
    embeddings: Embeddings | None = None,
    **api_overrides: Any,
) -> AppServices:
    settings = Settings(
        chroma_db_path=tmp_path / "chroma",
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        collection_name="api_test",
        min_relevance_score=0.15,
        ollama_base_url="http://127.0.0.1:9",  # closed port for the legacy health fields
    )
    # Most tests upload documents without a key; the secure default (uploads need the admin
    # key) has its own tests below.
    api_overrides.setdefault("uploads_require_admin", False)
    api = ApiSettings(admin_api_key=admin_key, **api_overrides)
    return build_services(
        settings,
        api,
        embeddings=embeddings if embeddings is not None else HashingEmbeddings(),
        llm=llm if llm is not None else FakeListChatModel(responses=["**Answer:** fake model answer [x.pdf, p. 1]."]),
    )


@pytest.fixture()
def services(tmp_path: Path) -> AppServices:
    return build_test_services(tmp_path)


@pytest.fixture()
def client(services: AppServices) -> Iterator[TestClient]:
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        yield test_client


def upload(client: TestClient, name: str, data: bytes, headers: dict[str, str] | None = None) -> httpx.Response:
    return client.post("/api/documents/upload", files={"file": (name, data)}, headers=headers or {})


@pytest.fixture()
def seeded(client: TestClient) -> TestClient:
    """Client whose index already contains a leave and a remote-work PDF (indexed through the API)."""
    assert upload(client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT)).status_code == 200
    assert upload(client, "policy_remote.pdf", make_pdf_bytes(REMOTE_TEXT)).status_code == 200
    return client


@pytest.fixture()
def seeded_with_csv(seeded: TestClient) -> TestClient:
    assert upload(seeded, "employees.csv", EMPLOYEE_CSV.encode(), {"X-Admin-Key": ADMIN_KEY}).status_code == 200
    return seeded


# ------------------------------------------------------------------ startup
def test_startup_builds_services_once_and_reuses_them(client: TestClient, services: AppServices) -> None:
    assert client.app.state.services is services
    first_store = client.app.state.services.store
    client.get("/api/health")
    client.post("/api/chat", json={"question": "anything"})
    assert client.app.state.services.store is first_store  # nothing re-created per request


def test_frontend_is_served_when_enabled(services: AppServices) -> None:
    with TestClient(create_app(services, serve_frontend=True)) as test_client:
        page = test_client.get("/")
        assert page.status_code == 200 and "HR ASSISTANT" in page.text
        assert test_client.get("/api/health").status_code == 200  # API still reachable beside the static files


def test_security_headers_present(client: TestClient) -> None:
    response = client.get("/api/health")
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


# ------------------------------------------------------------------- health
def test_health_adds_groq_and_local_embedding_status(client: TestClient) -> None:
    """The legacy Ollama fields remain while readiness uses the migrated providers."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["ollama"]["reachable"] is False and "not reachable" in body["ollama"]["error"]
    assert body["groq"] == {
        "reachable": True,
        "model": "openai/gpt-oss-120b",
        "model_available": True,
        "error": None,
    }
    assert body["sentence_transformers"]["available"] is True
    assert body["sentence_transformers"]["model"] == "BAAI/bge-m3"
    assert body["vector_store"]["available"] is True and body["vector_store"]["chunk_count"] == 0
    assert body["employee_data_enabled"] is True


def _fake_tags(models: list[str]):
    def fake_get(url: str, timeout: float) -> httpx.Response:
        assert url.endswith("/api/tags")
        return httpx.Response(200, json={"models": [{"name": m} for m in models]}, request=httpx.Request("GET", url))

    return fake_get


def test_health_ok_when_models_installed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """MOCKED Ollama /api/tags response."""
    monkeypatch.setattr("backend.routes.health.httpx.get", _fake_tags(["bge-m3:latest", "qwen2.5:3b"]))
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["ollama"]["error"] is None


def test_health_names_missing_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("backend.routes.health.httpx.get", _fake_tags(["bge-m3:latest"]))
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["ollama"]["llm_model_available"] is False
    assert "ollama pull qwen2.5:3b" in body["ollama"]["error"]


# --------------------------------------------------------------------- chat
def test_chat_returns_grounded_answer_with_real_sources(seeded: TestClient) -> None:
    response = seeded.post("/api/chat", json={"question": "How many annual leave days can be carried over?"})
    assert response.status_code == 200
    body = response.json()
    assert body["grounded"] is True and "fake model answer" in body["answer"]
    assert set(body) == {
        "answer", "conversation_id", "standalone_question", "grounded", "index_empty",
        "sources", "citations", "all_verified", "elapsed_ms",
    }
    assert body["citations"] == [
        {
            "citation": "[x.pdf, p. 1]",
            "source": "x.pdf",
            "filename": "x.pdf",
            "page": 1,
            "record_id": None,
            "verified": False,
            "status": "unverified",
        }
    ]
    assert body["all_verified"] is False
    top = body["sources"][0]
    assert top["source"] == "policy_leave.pdf" and top["page"] == 1
    assert top["citation"] == "policy_leave.pdf, p. 1"
    assert "carry over" in top["excerpt"] and 0 < top["relevance"] <= 1
    assert set(top) == {"source", "doc_type", "page", "record_id", "title", "citation", "relevance", "excerpt"}


def test_chat_with_nothing_relevant_skips_the_llm(client: TestClient, tmp_path: Path) -> None:
    class Exploding(FakeListChatModel):
        def _call(self, *args: Any, **kwargs: Any) -> str:
            raise AssertionError("LLM must not be called")

    services = build_test_services(tmp_path / "x", llm=Exploding(responses=["unused"]))
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        upload(test_client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT))
        body = test_client.post("/api/chat", json={"question": "banana smoothie honey recipe"}).json()
    assert body["grounded"] is False and body["sources"] == [] and "do not contain enough information" in body["answer"]


def test_chat_on_empty_index(client: TestClient) -> None:
    body = client.post("/api/chat", json={"question": "What is the leave policy?"}).json()
    assert body["index_empty"] is True and body["grounded"] is False and body["sources"] == []


@pytest.mark.parametrize("payload", [{"question": ""}, {"question": "   "}, {"question": "x" * 2001}, {}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 99}])
def test_chat_rejects_invalid_requests(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_chat_reports_llm_failure_as_503(seeded: TestClient, tmp_path: Path) -> None:
    class Failing(FakeListChatModel):
        def _call(self, *args: Any, **kwargs: Any) -> str:
            raise ConnectionError("connection refused")

    services = build_test_services(tmp_path / "f", llm=Failing(responses=["x"]))
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        upload(test_client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT))
        response = test_client.post("/api/chat", json={"question": "annual leave carry over days"})
    assert response.status_code == 503 and "generation failed" in response.json()["detail"]


def test_chat_on_empty_index_needs_no_embedding_call(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, embeddings=ExplodingEmbeddings())
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        response = test_client.post("/api/chat", json={"question": "anything"})
    assert response.status_code == 200 and response.json()["index_empty"] is True


def test_chat_embedding_failure_on_populated_index(seeded: TestClient, services: AppServices) -> None:
    services.store._embeddings = ExplodingEmbeddings()  # noqa: SLF001 - simulate Ollama dying after startup
    services.store._store._embedding_function = ExplodingEmbeddings()  # noqa: SLF001
    response = seeded.post("/api/chat", json={"question": "annual leave carry over"})
    assert response.status_code == 503 and "Retrieval failed" in response.json()["detail"]


def test_source_formatting_unit() -> None:
    long_text = "word " * 400
    chunks = [
        RetrievedChunk(long_text, "a.pdf", 3, "Title A", "pdf", 0.81234),
        RetrievedChunk("row text", "staff.csv", None, None, "csv", 0.5, record_id="EMP001"),
        RetrievedChunk("plain", "b.docx", None, None, "docx", 0.4),
    ]
    result = AnswerResult(
        "q",
        "Facts [a.pdf, p. 3] [staff.csv, record EMP001] [a.pdf, p. 4].",
        [],
        RetrievalResult("q", chunks),
        grounded=True,
    )
    body = to_chat_response(result, 12)
    assert [s.citation for s in body.sources] == ["a.pdf, p. 3", "staff.csv, record EMP001", "b.docx"]
    assert body.sources[0].excerpt.endswith("…") and len(body.sources[0].excerpt) <= 701
    assert body.sources[0].relevance == 0.8123 and body.elapsed_ms == 12
    assert [citation.status for citation in body.citations] == ["verified", "verified", "unverified"]
    assert body.all_verified is False
    ungrounded = to_chat_response(AnswerResult("q", "no", [], RetrievalResult("q", chunks), grounded=False), 1)
    assert ungrounded.sources == [] and ungrounded.citations == []
    assert ungrounded.all_verified is True


# ------------------------------------------------ access control (sensitive CSV)
def test_non_admin_never_receives_employee_data(seeded_with_csv: TestClient) -> None:
    question = "leave balance of Sara Al-Rashidi employee EMP001 manager Khalid Mansour"
    for headers in ({}, {"X-Admin-Key": "wrong-key-wrong-key"}):
        body = seeded_with_csv.post("/api/chat", json={"question": question, "top_k": 10}, headers=headers).json()
        assert all(s["doc_type"] != "csv" and s["source"] != "employees.csv" for s in body["sources"])
        assert "Sara" not in " ".join(s["excerpt"] for s in body["sources"])


def test_admin_can_retrieve_employee_data(seeded_with_csv: TestClient) -> None:
    body = seeded_with_csv.post(
        "/api/chat",
        json={"question": "leave balance of Sara Al-Rashidi employee EMP001", "top_k": 10},
        headers={"X-Admin-Key": ADMIN_KEY},
    ).json()
    csv_sources = [s for s in body["sources"] if s["source"] == "employees.csv"]
    assert csv_sources and csv_sources[0]["record_id"] == "EMP001"


def test_document_list_hides_employee_csv_from_non_admins(seeded_with_csv: TestClient) -> None:
    public = seeded_with_csv.get("/api/documents").json()
    assert [d["name"] for d in public["documents"]] == ["policy_leave.pdf", "policy_remote.pdf"]
    assert public["sensitive_visible"] is False and "employees.csv" not in str(public)
    private = seeded_with_csv.get("/api/documents", headers={"X-Admin-Key": ADMIN_KEY}).json()
    csv_entry = next(d for d in private["documents"] if d["name"] == "employees.csv")
    assert csv_entry["sensitive"] is True and private["sensitive_visible"] is True
    assert public["total_chunks"] < private["total_chunks"]


def test_health_does_not_count_sensitive_chunks_for_non_admins(seeded_with_csv: TestClient) -> None:
    public = seeded_with_csv.get("/api/health").json()
    private = seeded_with_csv.get("/api/health", headers={"X-Admin-Key": ADMIN_KEY}).json()
    assert public["documents_on_disk"] == 2 and private["documents_on_disk"] == 3
    assert public["vector_store"]["chunk_count"] < private["vector_store"]["chunk_count"]


def test_csv_upload_requires_admin(client: TestClient) -> None:
    assert upload(client, "employees.csv", EMPLOYEE_CSV.encode()).status_code == 403
    assert upload(client, "employees.csv", EMPLOYEE_CSV.encode(), {"X-Admin-Key": "nope-nope-nope"}).status_code == 403
    assert not (client.app.state.services.settings.raw_data_dir / "employees.csv").exists()


def test_everyone_is_non_admin_when_no_key_is_configured(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, admin_key="")
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        for supplied in ("", "anything-at-all-here"):
            response = upload(test_client, "employees.csv", EMPLOYEE_CSV.encode(), {"X-Admin-Key": supplied})
            assert response.status_code == 403 and "disabled" in response.json()["detail"]
        status = test_client.get("/api/auth/status", headers={"X-Admin-Key": "anything-at-all-here"}).json()
        assert status == {"admin": False, "employee_data_enabled": False, "uploads_require_admin": False}


def test_uploads_can_be_restricted_to_admins(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, uploads_require_admin=True)
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        assert upload(test_client, "p.pdf", make_pdf_bytes(LEAVE_TEXT)).status_code == 403
        assert upload(test_client, "p.pdf", make_pdf_bytes(LEAVE_TEXT), {"X-Admin-Key": "wrong-wrong-wrong"}).status_code == 403
        assert services.store.count() == 0
        assert upload(test_client, "p.pdf", make_pdf_bytes(LEAVE_TEXT), {"X-Admin-Key": ADMIN_KEY}).status_code == 200


def test_auth_status(client: TestClient) -> None:
    assert client.get("/api/auth/status").json()["admin"] is False
    assert client.get("/api/auth/status", headers={"X-Admin-Key": ADMIN_KEY}).json()["admin"] is True


def test_short_admin_key_is_rejected_at_configuration() -> None:
    with pytest.raises(ValueError):
        ApiSettings(admin_api_key="short")


# ------------------------------------------------------------------- upload
def test_upload_pdf_end_to_end(client: TestClient, services: AppServices) -> None:
    response = upload(client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT, "Second page about sick leave."))
    assert response.status_code == 200
    body = response.json()
    assert (body["filename"], body["doc_type"], body["replaced"], body["sections"]) == ("policy_leave.pdf", "pdf", False, 2)
    assert body["chunks_created"] == body["chunks_added"] >= 2 and body["duplicates_skipped"] == 0
    assert (services.settings.raw_data_dir / "policy_leave.pdf").is_file()
    listing = client.get("/api/documents").json()["documents"]
    assert listing[0]["name"] == "policy_leave.pdf" and listing[0]["status"] == "indexed"
    assert listing[0]["chunks"] == body["chunks_added"] and listing[0]["size_bytes"] > 0
    assert not list((services.settings.raw_data_dir / ".incoming").glob("*"))  # temp upload cleaned up


def test_reupload_same_file_adds_no_duplicates_and_changed_file_replaces(client: TestClient) -> None:
    upload(client, "p.pdf", make_pdf_bytes(LEAVE_TEXT))
    again = upload(client, "p.pdf", make_pdf_bytes(LEAVE_TEXT)).json()
    assert again["replaced"] is True and again["chunks_added"] == 0 and again["duplicates_skipped"] >= 1
    changed = upload(client, "p.pdf", make_pdf_bytes("Completely different text about parental leave.")).json()
    assert changed["stale_removed"] >= 1 and changed["chunks_added"] >= 1
    assert client.get("/api/documents").json()["total_chunks"] == changed["chunks_added"]


def test_upload_docx_is_supported(client: TestClient, tmp_path: Path) -> None:
    from docx import Document as DocxDocument

    path = tmp_path / "handbook.docx"
    document = DocxDocument()
    document.add_paragraph("Employee handbook. Core hours are nine to three.")
    document.save(path)
    response = upload(client, "handbook.docx", path.read_bytes())
    assert response.status_code == 200 and response.json()["doc_type"] == "docx"


@pytest.mark.parametrize(
    ("name", "data", "status"),
    [
        ("broken.pdf", b"this is not a pdf", 422),
        ("empty.pdf", b"", 422),
        ("blank.pdf", make_pdf_bytes(""), 422),
        ("broken.docx", b"not a zip", 422),
        ("notes.txt", b"hello", 400),
        ("script.exe", b"MZ", 400),
        ("noextension", b"data", 400),
        ("README.md", b"# hi", 400),
        ("README.pdf", make_pdf_bytes(LEAVE_TEXT), 400),
    ],
)
def test_upload_rejects_bad_files_and_changes_nothing(
    client: TestClient, services: AppServices, name: str, data: bytes, status: int
) -> None:
    response = upload(client, name, data)
    assert response.status_code == status and response.json()["detail"]
    assert services.store.count() == 0
    assert [p for p in services.settings.raw_data_dir.iterdir() if p.is_file()] == []


def test_upload_filename_cannot_escape_the_raw_directory(client: TestClient, services: AppServices, tmp_path: Path) -> None:
    response = upload(client, "../../evil name?.pdf", make_pdf_bytes(LEAVE_TEXT))
    assert response.status_code == 200
    saved = response.json()["filename"]
    assert "/" not in saved and ".." not in saved
    assert (services.settings.raw_data_dir / saved).is_file()
    assert not (tmp_path / "evil name?.pdf").exists()
    assert sanitize_filename("C:\\Users\\x\\report.PDF") == "report.pdf"


def test_upload_size_limit(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, max_upload_mb=1)
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        response = upload(test_client, "big.pdf", b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024))
    assert response.status_code == 413 and services.store.count() == 0


def test_upload_when_embedding_service_is_down_changes_nothing(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, embeddings=ExplodingEmbeddings())
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        response = upload(test_client, "p.pdf", make_pdf_bytes(LEAVE_TEXT))
    assert response.status_code == 503 and "local model" in response.json()["detail"]
    assert services.store.count() == 0 and not (services.settings.raw_data_dir / "p.pdf").exists()


def test_list_shows_files_on_disk_that_were_never_indexed(client: TestClient, services: AppServices) -> None:
    (services.settings.raw_data_dir / "manual.pdf").write_bytes(make_pdf_bytes(LEAVE_TEXT))
    (services.settings.raw_data_dir / "README.md").write_text("ignored")
    docs = client.get("/api/documents").json()["documents"]
    assert [(d["name"], d["status"], d["chunks"]) for d in docs] == [("manual.pdf", "not_indexed", 0)]


# -------------------------------------------------------------- concurrency
def test_many_parallel_chat_requests_all_succeed(seeded: TestClient) -> None:
    results: list[int] = []

    def ask() -> None:
        results.append(seeded.post("/api/chat", json={"question": "annual leave carry over days"}).status_code)

    threads = [threading.Thread(target=ask) for _ in range(12)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results == [200] * 12


def test_chat_returns_503_when_all_slots_stay_busy(tmp_path: Path) -> None:
    services = build_test_services(tmp_path, max_concurrent_chats=1, chat_queue_timeout_s=0.2)
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        assert services.chat_slots.acquire(timeout=1)  # another request is "generating"
        try:
            response = test_client.post("/api/chat", json={"question": "anything"})
        finally:
            services.chat_slots.release()
    assert response.status_code == 503 and "busy" in response.json()["detail"]


def test_parallel_uploads_are_serialised_and_all_indexed(client: TestClient) -> None:
    statuses: dict[str, int] = {}

    def send(i: int) -> None:
        statuses[f"doc{i}.pdf"] = upload(client, f"doc{i}.pdf", make_pdf_bytes(f"Policy number {i}. {LEAVE_TEXT}")).status_code

    threads = [threading.Thread(target=send, args=(i,)) for i in range(5)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert set(statuses.values()) == {200}
    assert len(client.get("/api/documents").json()["documents"]) == 5


def test_unhandled_errors_return_json_500(services: AppServices) -> None:
    app = create_app(services, serve_frontend=False)

    @app.get("/api/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/api/boom")
    assert response.status_code == 500 and "secret" not in response.text


# ------------------------------------------------- secure default for uploads
def test_uploads_require_the_admin_key_by_default() -> None:
    assert ApiSettings().uploads_require_admin is True


def test_default_settings_block_anonymous_uploads_and_name_the_fix(tmp_path: Path) -> None:
    for key in ("", ADMIN_KEY):
        services = build_services(
            Settings(
                chroma_db_path=tmp_path / f"c{bool(key)}", raw_data_dir=tmp_path / f"r{bool(key)}",
                processed_data_dir=tmp_path / "p", collection_name="secure_default",
            ),
            ApiSettings(admin_api_key=key),  # all other API settings at their defaults
            embeddings=HashingEmbeddings(),
            llm=FakeListChatModel(responses=["x"]),
        )
        with TestClient(create_app(services, serve_frontend=False)) as test_client:
            response = upload(test_client, "p.pdf", make_pdf_bytes(LEAVE_TEXT))
            assert response.status_code == 403
            assert ("ADMIN_API_KEY" in response.json()["detail"]) == (key == "")
            assert services.store.count() == 0
            assert not (services.settings.raw_data_dir / "p.pdf").exists()
            status = test_client.get("/api/auth/status").json()
            assert status["uploads_require_admin"] is True
            assert test_client.get("/api/health").json()["uploads_require_admin"] is True


# --------------------------------------------------- conversation context
from backend.conversation import ConversationStore, Turn, condense_question  # noqa: E402


def test_store_keeps_only_the_most_recent_turns() -> None:
    store = ConversationStore(max_turns=2)
    for i in range(4):
        store.add("conv-0001", False, f"q{i}", f"a{i}")
    assert [t.question for t in store.get("conv-0001", False)] == ["q2", "q3"]


def test_store_expires_idle_conversations_and_caps_their_number() -> None:
    now = [0.0]
    store = ConversationStore(max_turns=2, max_conversations=2, ttl_seconds=10, clock=lambda: now[0])
    store.add("conv-0001", False, "q", "a")
    now[0] = 11
    assert store.get("conv-0001", False) == []  # expired
    for name in ("conv-0002", "conv-0003", "conv-0004"):
        store.add(name, False, "q", "a")
    assert len(store) == 2 and store.get("conv-0002", False) == []  # oldest evicted


def test_store_separates_admin_and_non_admin_and_can_be_disabled() -> None:
    store = ConversationStore(max_turns=2)
    store.add("conv-0001", True, "secret question", "secret answer")
    assert store.get("conv-0001", False) == []
    assert len(store.get("conv-0001", True)) == 1
    off = ConversationStore(max_turns=0)
    off.add("conv-0001", False, "q", "a")
    assert off.get("conv-0001", False) == [] and not off.enabled


def test_condense_without_history_does_not_call_the_model() -> None:
    class Boom(FakeListChatModel):
        def _call(self, *a: Any, **k: Any) -> str:
            raise AssertionError("must not be called")

    assert condense_question(Boom(responses=["x"]), [], "Hello?") == "Hello?"


@pytest.mark.parametrize(
    ("model_output", "expected"),
    [
        ("Standalone question: How many days can juniors carry over?", "How many days can juniors carry over?"),
        ('"How many days can juniors carry over?"\nThis is because...', "How many days can juniors carry over?"),
        ("", "And for juniors?"),
        ("x" * 900, "And for juniors?"),
    ],
)
def test_condense_cleans_or_rejects_model_output(model_output: str, expected: str) -> None:
    history = [Turn("How many days can be carried over?", "Five days.")]
    assert condense_question(FakeListChatModel(responses=[model_output]), history, "And for juniors?") == expected


def test_condense_falls_back_when_the_model_fails() -> None:
    class Failing(FakeListChatModel):
        def _call(self, *a: Any, **k: Any) -> str:
            raise ConnectionError("down")

    assert condense_question(Failing(responses=["x"]), [Turn("q", "a")], "And juniors?") == "And juniors?"


def test_follow_up_question_is_rewritten_using_the_conversation(tmp_path: Path) -> None:
    # FakeListChatModel replays these in order: answer 1, rewrite, answer 2.
    llm = FakeListChatModel(responses=[
        "**Answer:** Five days [policy_leave.pdf, p. 1].",
        "How many annual leave days can be carried over to the next calendar year?",
        "**Answer:** Still five days [policy_leave.pdf, p. 1].",
    ])
    services = build_test_services(tmp_path, llm=llm)
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        assert upload(test_client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT)).status_code == 200
        assert upload(test_client, "policy_remote.pdf", make_pdf_bytes(REMOTE_TEXT)).status_code == 200
        first = test_client.post("/api/chat", json={
            "question": "How many annual leave days can be carried over?", "conversation_id": "conv-abc123"}).json()
        assert first["standalone_question"] is None and first["conversation_id"] == "conv-abc123"
        second = test_client.post("/api/chat", json={"question": "and next year?", "conversation_id": "conv-abc123"}).json()
        assert second["standalone_question"] == "How many annual leave days can be carried over to the next calendar year?"
        assert second["sources"][0]["source"] == "policy_leave.pdf"  # retrieval used the rewritten text
        assert "Still five days" in second["answer"]


def test_chat_without_conversation_id_is_stateless_and_ids_are_validated(seeded: TestClient) -> None:
    body = seeded.post("/api/chat", json={"question": "How many annual leave days can be carried over?"}).json()
    assert body["conversation_id"] is None and body["standalone_question"] is None
    assert seeded.post("/api/chat", json={"question": "ok question", "conversation_id": "bad id!"}).status_code == 422


def test_admin_conversation_history_is_not_used_for_a_non_admin_request(tmp_path: Path) -> None:
    llm = FakeListChatModel(responses=["**Answer:** one.", "**Answer:** two."])  # a rewrite call would consume a 3rd
    services = build_test_services(tmp_path, llm=llm)
    with TestClient(create_app(services, serve_frontend=False)) as test_client:
        assert upload(test_client, "policy_leave.pdf", make_pdf_bytes(LEAVE_TEXT)).status_code == 200
        question = {"question": "How many annual leave days can be carried over?", "conversation_id": "conv-shared1"}
        test_client.post("/api/chat", json=question, headers={"X-Admin-Key": ADMIN_KEY})
        result = test_client.post("/api/chat", json=question).json()  # same id, but not admin
        assert result["standalone_question"] is None  # no history was applied
        assert services.conversations.get("conv-shared1", False)[0].question == question["question"]
