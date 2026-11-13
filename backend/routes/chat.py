"""POST /api/chat - question answering through the existing RAG pipeline."""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.citation_verifier import verify_citations
from backend.conversation import condense_question
from backend.dependencies import AppServices, access_filter, get_services, is_admin
from backend.schemas import ChatRequest, ChatResponse, CitationVerificationOut, ErrorResponse, SourceOut
from src.rag_pipeline import AnswerResult, RAGPipelineError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["chat"])

EXCERPT_MAX_CHARS = 700


def to_chat_response(
    result: AnswerResult,
    elapsed_ms: int,
    conversation_id: str | None = None,
    standalone_question: str | None = None,
) -> ChatResponse:
    """Convert the pipeline's ``AnswerResult`` into the API response.

    Sources are built from the retrieved chunks (``result.retrieval.chunks``), which
    carry the file name, page / record id, relevance and text. Nothing comes from
    the model's own text. Ungrounded answers (nothing relevant found, or empty
    index) carry no sources.
    """
    sources: list[SourceOut] = []
    if result.grounded and result.retrieval is not None:
        for chunk in result.retrieval.chunks:
            excerpt = chunk.content.strip()
            if len(excerpt) > EXCERPT_MAX_CHARS:
                excerpt = excerpt[:EXCERPT_MAX_CHARS].rstrip() + "…"
            sources.append(
                SourceOut(
                    source=chunk.source,
                    doc_type=chunk.doc_type,
                    page=chunk.page,
                    record_id=chunk.record_id,
                    title=chunk.title,
                    citation=chunk.citation,
                    relevance=round(chunk.score, 4),
                    excerpt=excerpt,
                )
            )
    citations = verify_citations(
        result.answer,
        result.retrieval.chunks if result.retrieval is not None else [],
    )
    return ChatResponse(
        answer=result.answer,
        conversation_id=conversation_id,
        standalone_question=standalone_question,
        grounded=result.grounded,
        index_empty=bool(result.retrieval and result.retrieval.index_empty),
        sources=sources,
        citations=[
            CitationVerificationOut(
                citation=citation.citation,
                source=citation.source,
                filename=citation.source,
                page=citation.page,
                record_id=citation.record_id,
                verified=citation.status == "verified",
                status=citation.status,
            )
            for citation in citations
        ],
        all_verified=all(citation.status == "verified" for citation in citations),
        elapsed_ms=elapsed_ms,
    )


@router.post(
    "/chat",
    response_model=ChatResponse,
    responses={503: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
def chat(body: ChatRequest, request: Request, services: AppServices = Depends(get_services)) -> ChatResponse:
    """Answer a question from the indexed HR documents.

    Callers without a valid admin key only ever see non-sensitive documents.
    With a ``conversation_id``, earlier turns of the same conversation are used to turn
    a follow-up into a standalone question first (see ``backend.conversation``).
    Runs in FastAPI's thread pool; at most ``MAX_CONCURRENT_CHATS`` generations run
    at once, the others wait (and get a 503 if they wait too long).
    """
    admin = is_admin(request, services)
    where = access_filter(admin)
    history = services.conversations.get(body.conversation_id, admin) if body.conversation_id else []
    if not services.chat_slots.acquire(timeout=services.api.chat_queue_timeout_s):
        raise HTTPException(status_code=503, detail="The assistant is busy. Please try again in a moment.")
    started = time.perf_counter()
    try:
        standalone = condense_question(services.llm, history, body.question)
        result = services.pipeline.ask(standalone, top_k=body.top_k, where=where)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RAGPipelineError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        services.chat_slots.release()
    if body.conversation_id:
        services.conversations.add(body.conversation_id, admin, body.question, result.answer)
    return to_chat_response(
        result,
        int((time.perf_counter() - started) * 1000),
        conversation_id=body.conversation_id,
        standalone_question=standalone if standalone != body.question else None,
    )
