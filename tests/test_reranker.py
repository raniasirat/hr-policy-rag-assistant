from __future__ import annotations

from pathlib import Path

import pytest

from backend.dependencies import ApiSettings, build_services
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from src.config import Settings, load_settings
from src.embeddings import HashingEmbeddings
from src.rag_pipeline import RAGPipeline
from src.reranker import get_reranker


@pytest.fixture(autouse=True)
def clear_reranker_cache():
    get_reranker.cache_clear()
    yield
    get_reranker.cache_clear()


def make_hashing_embeddings(settings: Settings) -> HashingEmbeddings:
    assert settings.embedding_model
    return HashingEmbeddings()


def make_fake_llm(settings: Settings) -> FakeListChatModel:
    assert settings.groq_model
    return FakeListChatModel(responses=["unused"])


def test_reranker_configuration_defaults_off_and_reads_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = tmp_path / "settings.env"
    monkeypatch.delenv("RERANKER_ENABLED", raising=False)
    assert load_settings(env_file=config_path).reranker_enabled is False

    monkeypatch.setenv("RERANKER_ENABLED", "true")
    assert load_settings(env_file=config_path).reranker_enabled is True


def test_cross_encoder_is_loaded_once_on_cpu_and_scores_query_passage_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[tuple[str, dict[str, object]]] = []
    predictions: list[tuple[list[tuple[str, str]], dict[str, object]]] = []

    class FakeCrossEncoder:
        def predict(self, pairs: list[tuple[str, str]], **kwargs: object) -> list[float]:
            predictions.append((pairs, kwargs))
            return [0.25, 0.75]

    def make_cross_encoder(model_name: str, **kwargs: object) -> FakeCrossEncoder:
        constructions.append((model_name, kwargs))
        return FakeCrossEncoder()

    monkeypatch.setattr("src.reranker.CrossEncoder", make_cross_encoder)

    first = get_reranker()
    second = get_reranker()
    scores = first.score("question", ["passage one", "passage two"])

    assert first is second
    assert constructions == [
        (
            "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
            {"device": "cpu", "local_files_only": True},
        )
    ]
    assert predictions == [
        (
            [("question", "passage one"), ("question", "passage two")],
            {"show_progress_bar": False},
        )
    ]
    assert scores == [0.25, 0.75]


def test_cross_encoder_rejects_non_finite_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossEncoder:
        def predict(self, _pairs: list[tuple[str, str]], **_kwargs: object) -> list[float]:
            del _pairs, _kwargs
            return [float("nan")]

    def make_cross_encoder(*_args: object, **_kwargs: object) -> FakeCrossEncoder:
        del _args, _kwargs
        return FakeCrossEncoder()

    monkeypatch.setattr("src.reranker.CrossEncoder", make_cross_encoder)
    reranker = get_reranker()

    with pytest.raises(RuntimeError, match="invalid scores"):
        reranker.score("question", ["passage"])


def test_backend_builds_with_the_reranker_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubReranker:
        def score(self, _query: str, passages: list[str]) -> list[float]:
            if not _query:
                raise AssertionError("query must be passed to reranker")
            return [0.0] * len(passages)

    reranker = StubReranker()
    monkeypatch.setattr("backend.dependencies.get_reranker", lambda: reranker)
    settings = Settings(
        reranker_enabled=True,
        chroma_db_path=tmp_path / "chroma",
        raw_data_dir=tmp_path / "raw",
    )

    services = build_services(
        settings=settings,
        api=ApiSettings(),
        embeddings=HashingEmbeddings(),
        llm=make_fake_llm(settings),
    )

    assert services.pipeline._retriever._reranker is reranker


def test_cli_pipeline_does_not_load_reranker_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.rag_pipeline.get_reranker", lambda: pytest.fail("must stay disabled"))
    monkeypatch.setattr("src.rag_pipeline.get_embeddings", make_hashing_embeddings)
    monkeypatch.setattr("src.rag_pipeline.get_llm", make_fake_llm)
    settings = Settings(
        chroma_db_path=tmp_path / "chroma",
        raw_data_dir=tmp_path / "raw",
    )

    pipeline = RAGPipeline.from_settings(settings)

    assert pipeline._retriever._reranker is None


def test_cli_pipeline_loads_reranker_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class StubReranker:
        def score(self, _query: str, passages: list[str]) -> list[float]:
            if not _query:
                raise AssertionError("query must be passed to reranker")
            return [0.0] * len(passages)

    reranker = StubReranker()
    monkeypatch.setattr("src.rag_pipeline.get_reranker", lambda: reranker)
    monkeypatch.setattr("src.rag_pipeline.get_embeddings", make_hashing_embeddings)
    monkeypatch.setattr("src.rag_pipeline.get_llm", make_fake_llm)
    settings = Settings(
        reranker_enabled=True,
        chroma_db_path=tmp_path / "chroma",
        raw_data_dir=tmp_path / "raw",
    )

    pipeline = RAGPipeline.from_settings(settings)

    assert pipeline._retriever._reranker is reranker
