"""Document management: list indexed documents and upload new ones.

Uploads reuse the project's own pipeline unchanged:
``load_file`` -> ``process_documents`` -> embeddings -> ``VectorStoreManager.add_documents``.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile

from backend.dependencies import (
    ADMIN_HEADER,
    SENSITIVE_EXTENSIONS,
    AppServices,
    get_services,
    is_admin,
)
from backend.schemas import DocumentInfo, DocumentListResponse, ErrorResponse, UploadResponse
from src.document_loader import (
    SUPPORTED_EXTENSIONS,
    DocumentLoadError,
    is_ignored_file,
    load_file,
)
from src.document_processor import process_documents
from src.embeddings import EmbeddingServiceError, verify_embedding_service

logger = logging.getLogger(__name__)
router = APIRouter(tags=["documents"])

_INCOMING_DIR_NAME = ".incoming"
_READ_BLOCK = 1024 * 1024
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ \-()]")
_MAX_NAME_LENGTH = 120


def sanitize_filename(raw_name: str | None) -> str:
    """Return a safe base name (no directories, no odd characters).

    Raises:
        HTTPException(400): If nothing usable is left or the extension is unsupported.
    """
    base = (raw_name or "").replace("\\", "/").split("/")[-1].strip()
    base = _SAFE_NAME_RE.sub("_", base).lstrip(". ")
    if not base:
        raise HTTPException(status_code=400, detail="The file has no usable name.")
    path = Path(base)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix or 'none'}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}.",
        )
    stem = path.stem[: _MAX_NAME_LENGTH - len(suffix)]
    name = f"{stem}{suffix}"
    if is_ignored_file(Path(name)):
        raise HTTPException(status_code=400, detail="This file name is reserved.")
    return name


def _utc_iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec="seconds")


def _is_sensitive_name(name: str) -> bool:
    return Path(name).suffix.lower() in SENSITIVE_EXTENSIONS


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(request: Request, services: AppServices = Depends(get_services)) -> DocumentListResponse:
    """List the files in ``data/raw`` together with their real chunk counts in the index.

    Sensitive files (CSV) are omitted entirely unless the caller is an admin.
    """
    admin = is_admin(request, services)
    try:
        indexed = services.store.list_sources()
    except Exception as exc:
        logger.exception("Could not read the vector store")
        raise HTTPException(status_code=503, detail=f"Vector store unavailable: {exc}") from exc

    raw_dir = services.settings.raw_data_dir
    documents: dict[str, DocumentInfo] = {}
    if raw_dir.is_dir():
        for path in sorted(raw_dir.iterdir()):
            if not path.is_file() or is_ignored_file(path) or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            sensitive = _is_sensitive_name(path.name)
            if sensitive and not admin:
                continue
            stat = path.stat()
            chunks = indexed.get(path.name, 0)
            documents[path.name] = DocumentInfo(
                name=path.name,
                doc_type=path.suffix.lower().lstrip("."),
                size_bytes=stat.st_size,
                modified=_utc_iso(stat.st_mtime),
                chunks=chunks,
                status="indexed" if chunks else "not_indexed",
                sensitive=sensitive,
            )
    for name, chunks in indexed.items():
        if name in documents or (_is_sensitive_name(name) and not admin):
            continue
        documents[name] = DocumentInfo(
            name=name,
            doc_type=Path(name).suffix.lower().lstrip(".") or "unknown",
            chunks=chunks,
            status="index_only",
            sensitive=_is_sensitive_name(name),
        )
    ordered = sorted(documents.values(), key=lambda d: d.name.lower())
    return DocumentListResponse(
        documents=ordered, total_chunks=sum(d.chunks for d in ordered), sensitive_visible=admin
    )


def _save_upload(upload: UploadFile, destination: Path, max_bytes: int) -> int:
    """Stream the upload to ``destination``, enforcing the size limit. Returns bytes written."""
    written = 0
    with destination.open("wb") as handle:
        while True:
            block = upload.file.read(_READ_BLOCK)
            if not block:
                break
            written += len(block)
            if written > max_bytes:
                raise HTTPException(
                    status_code=413, detail=f"File too large (limit {max_bytes // (1024 * 1024)} MB)."
                )
            handle.write(block)
    if written == 0:
        raise HTTPException(status_code=422, detail="The uploaded file is empty.")
    return written


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    responses={400: {"model": ErrorResponse}, 403: {"model": ErrorResponse}, 413: {"model": ErrorResponse},
               422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
def upload_document(
    request: Request,
    file: UploadFile = File(...),
    services: AppServices = Depends(get_services),
) -> UploadResponse:
    """Validate, process and index one uploaded PDF, DOCX or CSV file.

    Nothing is changed (no file stored, no chunks indexed) unless every step
    succeeds. Uploading a file whose name already exists replaces it; chunks of the
    old version that no longer exist are removed from the index.
    """
    name = sanitize_filename(file.filename)
    sensitive = _is_sensitive_name(name)
    if services.api.uploads_require_admin and not is_admin(request, services):
        if not services.api.admin_api_key:
            raise HTTPException(
                status_code=403,
                detail="Uploads are disabled: set ADMIN_API_KEY in the server's .env file (or "
                "UPLOADS_REQUIRE_ADMIN=false for a private local test) and restart the server.",
            )
        raise HTTPException(status_code=403, detail=f"Uploading documents requires a valid {ADMIN_HEADER}.")
    if sensitive:
        if not services.api.admin_api_key:
            raise HTTPException(
                status_code=403,
                detail="Uploading employee data (CSV) is disabled: no admin key is configured on the server.",
            )
        if not is_admin(request, services):
            raise HTTPException(
                status_code=403, detail=f"Uploading employee data (CSV) requires a valid {ADMIN_HEADER}."
            )

    settings = services.settings
    raw_dir = settings.raw_data_dir
    incoming = raw_dir / _INCOMING_DIR_NAME / uuid.uuid4().hex
    incoming.mkdir(parents=True, exist_ok=True)
    temp_path = incoming / name  # same file name, so chunk metadata carries the final name
    try:
        _save_upload(file, temp_path, services.api.max_upload_mb * 1024 * 1024)
        try:
            documents = load_file(temp_path)
        except DocumentLoadError as exc:
            raise HTTPException(status_code=422, detail=f"{name}: {exc}") from exc
        chunks = process_documents(documents, settings.chunk_size, settings.chunk_overlap)
        if not chunks:
            raise HTTPException(status_code=422, detail=f"{name}: no usable text after cleaning.")

        try:
            verify_embedding_service(services.embeddings, settings.embedding_model)
        except EmbeddingServiceError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        final_path = raw_dir / name
        replaced = final_path.exists()
        with services.write_lock:
            try:
                report = services.store.add_documents(chunks)
            except Exception as exc:
                logger.exception("Indexing %s failed", name)
                raise HTTPException(status_code=503, detail=f"Indexing failed: {exc}") from exc
            os.replace(temp_path, final_path)
    finally:
        shutil.rmtree(incoming, ignore_errors=True)

    logger.info("Uploaded %s: %d chunk(s) added, %d duplicate(s), %d stale removed",
                name, report.added, report.skipped_duplicates, report.removed_stale)
    return UploadResponse(
        filename=name,
        doc_type=Path(name).suffix.lower().lstrip("."),
        replaced=replaced,
        # CSV summary records are derived documents, not rows of the uploaded file
        sections=sum(1 for d in documents if not str(d.metadata.get("record_id", "")).startswith("summary-")),
        chunks_created=len(chunks),
        chunks_added=report.added,
        duplicates_skipped=report.skipped_duplicates,
        stale_removed=report.removed_stale,
        sensitive=sensitive,
        message=(
            f"{name} was indexed: {report.added} new chunk(s) added"
            + (f", {report.skipped_duplicates} unchanged chunk(s) skipped" if report.skipped_duplicates else "")
            + (f", {report.removed_stale} outdated chunk(s) removed" if report.removed_stale else "")
            + "."
        ),
    )
