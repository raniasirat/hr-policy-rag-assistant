"""Focused tests for local Sentence Transformers and Groq provider adapters."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import httpx
import numpy as np
import pytest
from groq import APIStatusError, APITimeoutError, RateLimitError
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import ConfigurationError, Settings
from src.embeddings import (
    EmbeddingServiceError,
    SentenceTransformerEmbeddings,
    _load_sentence_transformer,
    get_embeddings,
    verify_embedding_service,
)
from src.llm import GroqChatModel, GroqRequestError, get_llm


@pytest.fixture(autouse=True)
def clear_embedding_model_cache():
    _load_sentence_transformer.cache_clear()
    yield
    _load_sentence_transformer.cache_clear()


class FakeSentenceTransformer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], bool]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 1024

    def encode(
        self,
        texts: list[str],
        *,
        convert_to_numpy: bool,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> np.ndarray:
        assert convert_to_numpy is True and show_progress_bar is False
        self.calls.append((texts, normalize_embeddings))
        vectors = np.zeros((len(texts), 1024), dtype=np.float32)
        for index, text in enumerate(texts):
            vectors[index, 0 if text == "same text" else 1] = 1.0
        return vectors


def test_sentence_transformer_embeddings_share_one_model_and_vector_space(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    constructions: list[str] = []

    def create_model(name: str) -> FakeSentenceTransformer:
        constructions.append(name)
        return model

    monkeypatch.setattr("src.embeddings.SentenceTransformer", create_model)
    documents = get_embeddings(Settings())
    query = get_embeddings(Settings())

    document_vectors = documents.embed_documents(["same text", "different text"])
    query_vector = query.embed_query("same text")

    assert constructions == ["BAAI/bge-m3"]
    assert documents._model is query._model
    assert documents.dimension == 1024
    assert [len(vector) for vector in document_vectors] == [1024, 1024]
    assert len(query_vector) == 1024
    assert query_vector == document_vectors[0]
    assert math.isclose(math.sqrt(sum(value * value for value in query_vector)), 1.0)
    assert model.calls == [
        (["same text", "different text"], True),
        (["same text"], True),
    ]


def test_embedding_probe_returns_the_same_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    model = FakeSentenceTransformer()
    monkeypatch.setattr("src.embeddings._load_sentence_transformer", lambda name: model)
    embeddings = SentenceTransformerEmbeddings("BAAI/bge-m3")

    assert verify_embedding_service(embeddings, "BAAI/bge-m3") == 1024


def test_embedding_model_load_errors_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_to_load(name: str) -> FakeSentenceTransformer:
        raise OSError("model unavailable")

    monkeypatch.setattr("src.embeddings.SentenceTransformer", fail_to_load)

    with pytest.raises(EmbeddingServiceError, match="Could not load local embedding model"):
        SentenceTransformerEmbeddings("BAAI/bge-m3")


class FakeGroqClient:
    def __init__(self, content: str = "Grounded response [policy.pdf, p. 1].") -> None:
        self.payload: dict[str, Any] | None = None
        self.content = content
        self.failure: Exception | None = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.models = SimpleNamespace(
            list=lambda: SimpleNamespace(data=[SimpleNamespace(id="openai/gpt-oss-120b")])
        )

    def with_options(self, **kwargs: Any) -> "FakeGroqClient":
        return self

    def create(self, **kwargs: Any) -> SimpleNamespace:
        if self.failure:
            raise self.failure
        self.payload = kwargs
        return SimpleNamespace(
            model=kwargs["model"],
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
        )


def test_groq_client_initialization_uses_environment_key_without_exposing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeGroqClient()
    init_kwargs: dict[str, Any] = {}

    def make_client(**kwargs: Any) -> FakeGroqClient:
        init_kwargs.update(kwargs)
        return client

    monkeypatch.setenv("GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr("src.llm.Groq", make_client)

    llm = get_llm(Settings())

    assert isinstance(llm, GroqChatModel)
    assert llm.model == "openai/gpt-oss-120b"
    assert init_kwargs == {"api_key": "unit-test-key", "timeout": 90.0, "max_retries": 0}
    assert "unit-test-key" not in repr(llm)


def test_groq_client_requires_an_environment_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(ConfigurationError, match="GROQ_API_KEY"):
        get_llm(Settings())


def test_groq_generation_preserves_prompt_messages_and_citations() -> None:
    client = FakeGroqClient()
    llm = GroqChatModel(model="openai/gpt-oss-120b", client=client)

    response = llm.invoke(
        [
            SystemMessage(content="Use only the supplied policy context."),
            HumanMessage(content="CONTEXT: policy excerpt\nQUESTION: How many days?"),
        ]
    )

    assert response.content == "Grounded response [policy.pdf, p. 1]."
    assert client.payload == {
        "model": "openai/gpt-oss-120b",
        "messages": [
            {"role": "system", "content": "Use only the supplied policy context."},
            {"role": "user", "content": "CONTEXT: policy excerpt\nQUESTION: How many days?"},
        ],
        "temperature": 0.0,
    }


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (
            RateLimitError(
                "rate limited",
                response=httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com")),
                body=None,
            ),
            "rate limit",
        ),
        (
            APITimeoutError(request=httpx.Request("POST", "https://api.groq.com")),
            "timed out",
        ),
        (
            APIStatusError(
                "model unavailable",
                response=httpx.Response(404, request=httpx.Request("POST", "https://api.groq.com")),
                body=None,
            ),
            "unavailable",
        ),
    ],
)
def test_groq_api_errors_are_mapped_to_safe_messages(failure: Exception, message: str) -> None:
    client = FakeGroqClient()
    client.failure = failure
    llm = GroqChatModel(model="openai/gpt-oss-120b", client=client)

    with pytest.raises(GroqRequestError, match=message):
        llm.invoke([HumanMessage(content="question")])


def test_groq_model_list_uses_account_visible_ids() -> None:
    llm = GroqChatModel(model="openai/gpt-oss-120b", client=FakeGroqClient())

    assert llm.list_model_ids() == {"openai/gpt-oss-120b"}
