"""Persistent conversation-history endpoints."""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.history import ConversationHistoryStore
from backend.schemas import ConversationCreate, HistoryMessageCreate, LegacyHistoryImport

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


def _history(request: Request) -> ConversationHistoryStore:
    return request.app.state.conversation_history


def _database_unavailable(exc: sqlite3.Error) -> HTTPException:
    logger.exception("Conversation history database operation failed (%s)", type(exc).__name__)
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Conversation history is temporarily unavailable.",
    )


@router.get("")
def list_conversations(request: Request) -> list[dict[str, Any]]:
    try:
        return _history(request).list_conversations()
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc


@router.post("")
def create_conversation(body: ConversationCreate, request: Request) -> dict[str, Any]:
    try:
        history = _history(request)
        history.create_conversation(body.id, body.title, body.updated)
        conversation = history.get_conversation(body.id)
        if conversation is None:
            raise HTTPException(status_code=503, detail="Conversation could not be created.")
        return conversation
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc


@router.get("/{conversation_id}/messages")
def get_messages(conversation_id: str, request: Request) -> list[dict[str, Any]]:
    try:
        messages = _history(request).get_messages(conversation_id)
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc
    if messages is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return messages


@router.post("/{conversation_id}/messages", status_code=status.HTTP_204_NO_CONTENT)
def save_message(
    conversation_id: str,
    body: HistoryMessageCreate,
    request: Request,
) -> Response:
    try:
        saved = _history(request).save_message(
            conversation_id,
            body.position,
            body.role,
            body.content,
            body.metadata,
            body.updated,
        )
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc
    if not saved:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/migrate")
def migrate_legacy_history(body: LegacyHistoryImport, request: Request) -> dict[str, Any]:
    conversations = [
        {
            "id": conversation.id,
            "title": conversation.title,
            "updated": conversation.updated,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "metadata": message.model_dump(exclude={"role", "content"}),
                }
                for message in conversation.messages
            ],
        }
        for conversation in body.conversations
    ]
    try:
        return _history(request).import_legacy(conversations)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, request: Request) -> Response:
    try:
        deleted = _history(request).delete_conversation(conversation_id)
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.delete("")
def delete_all_conversations(request: Request) -> dict[str, int]:
    try:
        deleted_count = _history(request).delete_all()
    except sqlite3.Error as exc:
        raise _database_unavailable(exc) from exc
    return {"deleted_count": deleted_count}
