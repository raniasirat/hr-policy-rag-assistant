"""Retired integration tests for the pre-migration Ollama providers."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.dependencies import ApiSettings, build_services  # noqa: E402
from backend.main import create_app  # noqa: E402
from src.config import Settings, load_settings  # noqa: E402

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skip(reason="The app uses Groq and local Sentence Transformers; see test_providers.py."),
]

POLICIES = sorted((Path(__file__).resolve().parent.parent / "data" / "raw").glob("policy_*.pdf"))


def _ollama_ready() -> tuple[bool, str]:
    settings = load_settings()
    try:
        tags = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).json()
    except (httpx.HTTPError, ValueError):
        return False, f"Ollama not reachable at {settings.ollama_base_url}"
    names = {m.get("name", "") for m in tags.get("models", [])}
    for model in (settings.embedding_model, settings.llm_model):
        if model not in names and f"{model}:latest" not in names:
            return False, f"model '{model}' is not installed (ollama pull {model})"
    return True, ""


_READY, _WHY = _ollama_ready()
requires_ollama = pytest.mark.skipif(not _READY, reason=_WHY or "Ollama unavailable")


@pytest.fixture()
def real_client(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for pdf in POLICIES:
        shutil.copy(pdf, raw)
    base = load_settings()
    settings = Settings(
        ollama_base_url=base.ollama_base_url,
        embedding_model=base.embedding_model,
        llm_model=base.llm_model,
        chroma_db_path=tmp_path / "chroma",
        raw_data_dir=raw,
        processed_data_dir=tmp_path / "processed",
        collection_name="ollama_it",
        min_relevance_score=base.min_relevance_score,
    )
    services = build_services(settings, ApiSettings(uploads_require_admin=False))  # real OllamaEmbeddings + ChatOllama; test-only: open uploads
    with TestClient(create_app(services, serve_frontend=False)) as client:
        yield client


@requires_ollama
def test_real_health_is_ok(real_client: TestClient) -> None:
    body = real_client.get("/api/health").json()
    assert body["status"] == "ok" and body["ollama"]["reachable"] is True


@requires_ollama
def test_real_upload_and_grounded_answer_with_sources(real_client: TestClient) -> None:
    for pdf in POLICIES:
        response = real_client.post("/api/documents/upload", files={"file": (pdf.name, pdf.read_bytes())})
        assert response.status_code == 200, response.text
    answer = real_client.post("/api/chat", json={"question": "How many days of annual leave can be carried over?"})
    assert answer.status_code == 200
    body = answer.json()
    assert body["grounded"] is True and body["answer"].strip()
    assert any(s["source"] == "policy_01_annual_leave.pdf" for s in body["sources"])
    assert all(s["page"] in (1, 2) for s in body["sources"])


@requires_ollama
def test_real_out_of_scope_question_is_not_answered_with_invented_sources(real_client: TestClient) -> None:
    for pdf in POLICIES:
        real_client.post("/api/documents/upload", files={"file": (pdf.name, pdf.read_bytes())})
    body = real_client.post("/api/chat", json={"question": "What is the recipe for Moroccan couscous?"}).json()
    # Either the relevance threshold or the model must refuse; we only require an honest outcome.
    assert "enough information" in body["answer"].lower() or "do not contain" in body["answer"].lower()
