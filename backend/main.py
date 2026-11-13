"""FastAPI application: HR Assistant API + static frontend.

Run locally (from the project root)::

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000

then open http://127.0.0.1:8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.dependencies import ADMIN_HEADER, AppServices, build_services, load_api_settings
from backend.history import ConversationHistoryStore
from backend.routes import chat, conversations, documents, health
from src.config import configure_logging, load_settings

logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
HISTORY_DB_PATH = Path(__file__).resolve().parent.parent / "storage" / "conversations.sqlite3"
API_VERSION = "1.0.0"


def create_app(
    services: AppServices | None = None,
    *,
    serve_frontend: bool = True,
    history_db_path: Path | None = None,
) -> FastAPI:
    """Application factory.

    Args:
        services: Pre-built services (used by tests). If omitted they are created at
            startup from the environment: the vector store, embedding client and LLM
            client are built once and reused for every request.
        serve_frontend: Serve ``frontend/`` at ``/`` (local development).
        history_db_path: Override the SQLite history database path (for tests).
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        history = ConversationHistoryStore(history_db_path or HISTORY_DB_PATH)
        history.initialize()
        app.state.conversation_history = history
        if services is not None:
            app.state.services = services
        else:
            settings = load_settings()
            configure_logging(settings.log_level)
            app.state.services = build_services(settings, load_api_settings())
            logger.info("HR Assistant API ready (collection '%s', %d chunk(s))",
                        settings.collection_name, app.state.services.store.count())
        yield

    app = FastAPI(title="HR Assistant API", version=API_VERSION, lifespan=lifespan)

    origins = list(services.api.cors_origins) if services is not None else list(load_api_settings().cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", ADMIN_HEADER],
    )

    @app.middleware("http")
    async def security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.exception_handler(Exception)
    async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error. See the server log."})

    app.include_router(health.router, prefix="/api")
    app.include_router(chat.router, prefix="/api")
    app.include_router(conversations.router, prefix="/api")
    app.include_router(documents.router, prefix="/api")

    if serve_frontend and FRONTEND_DIR.is_dir():
        app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
    return app


app = create_app()
