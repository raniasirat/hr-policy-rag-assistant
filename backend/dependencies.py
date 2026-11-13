"""Shared services, API settings and access control for the FastAPI backend.

The expensive objects (vector store, embedding client, LLM client, RAG pipeline)
are created once at startup and shared by all requests via ``AppServices``.

Access control for sensitive data
---------------------------------
CSV files (the employee table) are treated as sensitive. They are indexed in the
same ChromaDB collection as the policies, so every query made without a valid
admin key carries the metadata filter ``NON_SENSITIVE_FILTER``, which Chroma applies
inside the database. Without a valid ``X-Admin-Key`` header a client therefore can
neither retrieve employee chunks through the chat endpoint, nor see CSV files in
the document list, nor upload CSV files. If ``ADMIN_API_KEY`` is not configured
(the default) nobody is an admin. Uploads of any type need the admin key too
(``UPLOADS_REQUIRE_ADMIN``, on by default), so nobody can alter the knowledge base
anonymously.
"""

from __future__ import annotations

import hmac
import os
import threading
from dataclasses import dataclass
from typing import Any

from fastapi import Request
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from backend.conversation import ConversationStore
from src.config import ConfigurationError, Settings, load_settings
from src.embeddings import get_embeddings
from src.llm import get_llm
from src.rag_pipeline import RAGPipeline
from src.retriever import Retriever
from src.reranker import get_reranker
from src.vector_store import VectorStoreManager

SENSITIVE_DOC_TYPES: frozenset[str] = frozenset({"csv"})
SENSITIVE_EXTENSIONS: frozenset[str] = frozenset({".csv"})
NON_SENSITIVE_FILTER: dict[str, Any] = {"doc_type": {"$nin": sorted(SENSITIVE_DOC_TYPES)}}
ADMIN_HEADER = "X-Admin-Key"
MIN_ADMIN_KEY_LENGTH = 12


class ApiSettings(BaseModel):
    """Settings that only the web API needs (kept out of ``src.config``)."""

    model_config = ConfigDict(frozen=True)

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: tuple[str, ...] = ("http://localhost:8000", "http://127.0.0.1:8000")
    admin_api_key: str = ""
    max_upload_mb: int = Field(default=25, ge=1, le=500)
    uploads_require_admin: bool = True
    history_turns: int = Field(default=4, ge=0, le=10, description="Turns remembered per conversation (0 = off)")
    conversation_ttl_s: float = Field(default=3600.0, gt=0)
    max_concurrent_chats: int = Field(default=2, ge=1, le=16)
    chat_queue_timeout_s: float = Field(default=120.0, gt=0)

    @field_validator("admin_api_key")
    @classmethod
    def _validate_admin_key(cls, value: str) -> str:
        value = value.strip()
        if value and len(value) < MIN_ADMIN_KEY_LENGTH:
            raise ValueError(f"must be at least {MIN_ADMIN_KEY_LENGTH} characters (or empty to disable)")
        return value


_API_ENV_VARS: dict[str, str] = {
    "host": "API_HOST",
    "port": "API_PORT",
    "admin_api_key": "ADMIN_API_KEY",
    "max_upload_mb": "MAX_UPLOAD_MB",
    "uploads_require_admin": "UPLOADS_REQUIRE_ADMIN",
    "history_turns": "HISTORY_TURNS",
    "conversation_ttl_s": "CONVERSATION_TTL_S",
    "max_concurrent_chats": "MAX_CONCURRENT_CHATS",
    "chat_queue_timeout_s": "CHAT_QUEUE_TIMEOUT_S",
}


def load_api_settings() -> ApiSettings:
    """Read API settings from the environment (the .env file is loaded by ``load_settings``)."""
    values: dict[str, Any] = {
        field: os.environ[env].strip()
        for field, env in _API_ENV_VARS.items()
        if os.environ.get(env, "").strip() != ""
    }
    origins = os.environ.get("CORS_ORIGINS", "").strip()
    if origins:
        values["cors_origins"] = tuple(o.strip().rstrip("/") for o in origins.split(",") if o.strip())
    try:
        return ApiSettings(**values)
    except ValidationError as exc:
        problems = "; ".join(f"{err['loc'][0] if err['loc'] else ''}: {err['msg']}" for err in exc.errors())
        raise ConfigurationError(f"Invalid API configuration: {problems}") from exc


@dataclass
class AppServices:
    """Everything the routes need, created once per process."""

    settings: Settings
    api: ApiSettings
    embeddings: Embeddings
    store: VectorStoreManager
    pipeline: RAGPipeline
    llm: BaseChatModel  # same client the pipeline uses; also rewrites follow-up questions
    conversations: ConversationStore
    write_lock: threading.Lock  # serialises index changes (uploads)
    chat_slots: threading.BoundedSemaphore  # limits simultaneous LLM generations


def build_services(
    settings: Settings | None = None,
    api: ApiSettings | None = None,
    *,
    embeddings: Embeddings | None = None,
    llm: BaseChatModel | None = None,
) -> AppServices:
    """Create the shared services with the project's own RAG components.

    ``embeddings`` and ``llm`` can be injected (the unit tests do this); in
    production they default to local Sentence Transformers embeddings and Groq.
    One ``VectorStoreManager`` is shared by retrieval and ingestion, so the Chroma
    collection is opened exactly once.
    """
    settings = settings or load_settings()
    api = api or load_api_settings()
    embeddings = embeddings if embeddings is not None else get_embeddings(settings)
    store = VectorStoreManager(settings.chroma_db_path, embeddings, settings.collection_name)
    reranker = get_reranker() if settings.reranker_enabled else None
    retriever = Retriever(
        store,
        settings.top_k,
        settings.min_relevance_score,
        reranker=reranker,
    )
    llm = llm if llm is not None else get_llm(settings)
    pipeline = RAGPipeline(retriever, llm)
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    return AppServices(
        settings=settings,
        api=api,
        embeddings=embeddings,
        store=store,
        pipeline=pipeline,
        llm=llm,
        conversations=ConversationStore(api.history_turns, ttl_seconds=api.conversation_ttl_s),
        write_lock=threading.Lock(),
        chat_slots=threading.BoundedSemaphore(api.max_concurrent_chats),
    )


def get_services(request: Request) -> AppServices:
    """FastAPI dependency returning the process-wide services."""
    return request.app.state.services


def is_admin(request: Request, services: AppServices) -> bool:
    """True only if a non-empty ``ADMIN_API_KEY`` is configured and the header matches it."""
    expected = services.api.admin_api_key
    supplied = request.headers.get(ADMIN_HEADER, "")
    if not expected or not supplied:
        return False
    return hmac.compare_digest(supplied.encode("utf-8"), expected.encode("utf-8"))


def access_filter(admin: bool) -> dict[str, Any] | None:
    """Chroma metadata filter for a caller: admins see everything, others no sensitive data."""
    return None if admin else NON_SENSITIVE_FILTER
