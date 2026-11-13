"""Health and authentication-status endpoints."""

from __future__ import annotations

import logging

import httpx
from fastapi import APIRouter, Depends, Request

from backend.dependencies import (
    SENSITIVE_EXTENSIONS,
    AppServices,
    get_services,
    is_admin,
)
from backend.schemas import (
    AuthStatus,
    EmbeddingHealth,
    GroqHealth,
    HealthResponse,
    OllamaHealth,
    VectorStoreHealth,
)
from src.document_loader import SUPPORTED_EXTENSIONS, is_ignored_file
from src.embeddings import SentenceTransformerEmbeddings
from src.llm import GroqChatModel, GroqRequestError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])

_HEALTH_TIMEOUT_S = 2.5


def _model_present(model: str, installed: set[str]) -> bool:
    """``bge-m3`` matches ``bge-m3:latest``; a model with an explicit tag must match exactly."""
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def check_ollama(services: AppServices) -> OllamaHealth:
    """Keep the legacy Ollama status fields accurate for existing API clients."""
    settings = services.settings
    embedding_model = "bge-m3" if settings.embedding_model.casefold() == "baai/bge-m3" else settings.embedding_model
    health = OllamaHealth(
        reachable=False,
        embedding_model=embedding_model,
        embedding_model_available=False,
        llm_model=settings.llm_model,
        llm_model_available=False,
    )
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=_HEALTH_TIMEOUT_S)
        response.raise_for_status()
        installed = {str(item.get("name", "")) for item in response.json().get("models", [])}
    except (httpx.HTTPError, ValueError) as exc:
        health.error = f"Ollama is not reachable at {settings.ollama_base_url} ({type(exc).__name__})"
        return health
    health.reachable = True
    health.embedding_model_available = _model_present(embedding_model, installed)
    health.llm_model_available = _model_present(settings.llm_model, installed)
    missing = [
        name
        for name, ok in ((embedding_model, health.embedding_model_available), (settings.llm_model, health.llm_model_available))
        if not ok
    ]
    if missing:
        health.error = "Missing Ollama model(s): " + ", ".join(f"'ollama pull {m}'" for m in missing)
    return health


def check_groq(services: AppServices) -> GroqHealth:
    """Check that the configured Groq model is visible to the current API key."""
    model = services.settings.groq_model
    if not isinstance(services.llm, GroqChatModel):
        return GroqHealth(reachable=True, model=model, model_available=True)
    try:
        available = model in services.llm.list_model_ids(timeout=_HEALTH_TIMEOUT_S)
    except GroqRequestError as exc:
        return GroqHealth(reachable=False, model=model, model_available=False, error=str(exc))
    if not available:
        return GroqHealth(
            reachable=True,
            model=model,
            model_available=False,
            error=f"Groq model '{model}' is unavailable to this account.",
        )
    return GroqHealth(reachable=True, model=model, model_available=True)


def check_sentence_transformers(services: AppServices) -> EmbeddingHealth:
    """Report the local embedding model and its output dimension."""
    model = services.settings.embedding_model
    if not isinstance(services.embeddings, SentenceTransformerEmbeddings):
        return EmbeddingHealth(available=True, model=model)
    try:
        return EmbeddingHealth(
            available=True,
            model=model,
            dimension=services.embeddings.dimension,
        )
    except Exception as exc:
        logger.error("Sentence Transformers health check failed (%s)", type(exc).__name__)
        return EmbeddingHealth(
            available=False,
            model=model,
            error=f"Local embedding model check failed ({type(exc).__name__}).",
        )


def check_vector_store(services: AppServices, admin: bool) -> VectorStoreHealth:
    """Open the collection and count chunks (non-sensitive chunks only for non-admins)."""
    health = VectorStoreHealth(available=False, collection=services.settings.collection_name)
    try:
        if admin:
            health.chunk_count = services.store.count()
        else:
            health.chunk_count = sum(
                count
                for source, count in services.store.list_sources().items()
                if not source.lower().endswith(tuple(SENSITIVE_EXTENSIONS))
            )
        health.available = True
    except Exception as exc:  # Chroma raises several unrelated exception types
        logger.exception("Vector store health check failed")
        health.error = str(exc)
    return health


def count_documents_on_disk(services: AppServices, admin: bool) -> int:
    raw_dir = services.settings.raw_data_dir
    if not raw_dir.is_dir():
        return 0
    return sum(
        1
        for path in raw_dir.iterdir()
        if path.is_file()
        and not is_ignored_file(path)
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and (admin or path.suffix.lower() not in SENSITIVE_EXTENSIONS)
    )


@router.get("/health", response_model=HealthResponse)
def health(request: Request, services: AppServices = Depends(get_services)) -> HealthResponse:
    """Report provider readiness, legacy Ollama status and vector-store state."""
    admin = is_admin(request, services)
    ollama = check_ollama(services)
    groq = check_groq(services)
    sentence_transformers = check_sentence_transformers(services)
    store = check_vector_store(services, admin)
    if (
        groq.reachable
        and groq.model_available
        and sentence_transformers.available
        and store.available
    ):
        status = "ok"
    elif not store.available:
        status = "error"
    else:
        status = "degraded"
    return HealthResponse(
        status=status,
        ollama=ollama,
        vector_store=store,
        documents_on_disk=count_documents_on_disk(services, admin),
        employee_data_enabled=bool(services.api.admin_api_key),
        uploads_require_admin=services.api.uploads_require_admin,
        groq=groq,
        sentence_transformers=sentence_transformers,
    )


@router.get("/auth/status", response_model=AuthStatus)
def auth_status(request: Request, services: AppServices = Depends(get_services)) -> AuthStatus:
    """Tell the frontend whether the supplied ``X-Admin-Key`` is valid."""
    return AuthStatus(
        admin=is_admin(request, services),
        employee_data_enabled=bool(services.api.admin_api_key),
        uploads_require_admin=services.api.uploads_require_admin,
    )
