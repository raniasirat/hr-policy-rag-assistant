"""Tests for persistent conversation-history storage and API behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.main import create_app


def _client(database_path: Path) -> TestClient:
    services = SimpleNamespace(api=SimpleNamespace(cors_origins=()))
    return TestClient(
        create_app(
            services,  # type: ignore[arg-type]
            serve_frontend=False,
            history_db_path=database_path,
        )
    )


def test_legacy_migration_preserves_metadata_and_is_idempotent(tmp_path: Path) -> None:
    legacy = {
        "conversations": [
            {
                "id": "legacy123456",
                "title": "Annual leave",
                "updated": 1730000000123,
                "messages": [
                    {"role": "user", "content": "How much leave can I carry?"},
                    {
                        "role": "assistant",
                        "content": "Up to five days.",
                        "sources": [{"source": "leave.pdf", "page": 2}],
                        "citations": [{"citation": "leave.pdf, p. 2", "verified": True}],
                        "grounded": True,
                        "elapsed_ms": 456,
                        "searched_for": None,
                    },
                ],
            }
        ]
    }

    with _client(tmp_path / "conversations.sqlite3") as client:
        first = client.post("/api/conversations/migrate", json=legacy)
        second = client.post("/api/conversations/migrate", json=legacy)

        assert first.status_code == second.status_code == 200
        assert first.json()["verified"] is True
        assert client.get("/api/conversations").json() == [
            {
                "id": "legacy123456",
                "title": "Annual leave",
                "updated": 1730000000123,
                "message_count": 2,
                "question_count": 1,
            }
        ]
        messages = client.get("/api/conversations/legacy123456/messages").json()
        assert len(messages) == 2
        assert messages[1]["sources"] == [{"source": "leave.pdf", "page": 2}]
        assert messages[1]["citations"] == [{"citation": "leave.pdf, p. 2", "verified": True}]
        assert messages[1]["grounded"] is True
        assert messages[1]["elapsed_ms"] == 456


def test_created_history_survives_app_restart_and_can_be_deleted(tmp_path: Path) -> None:
    database_path = tmp_path / "conversations.sqlite3"
    conversation = {
        "id": "newchat12345",
        "title": "New question",
        "updated": 1730000000000,
    }
    with _client(database_path) as client:
        assert client.post("/api/conversations", json=conversation).status_code == 200
        assert client.post(
            "/api/conversations/newchat12345/messages",
            json={
                "position": 0,
                "role": "user",
                "content": "Where is the leave policy?",
                "metadata": {},
                "updated": 1730000000001,
            },
        ).status_code == 204
        assert client.post(
            "/api/conversations/newchat12345/messages",
            json={
                "position": 1,
                "role": "assistant",
                "content": "The policy document has the details.",
                "metadata": {
                    "sources": [{"source": "leave.pdf", "page": 2}],
                    "citations": [{"citation": "leave.pdf, p. 2", "verified": True}],
                    "grounded": True,
                },
                "updated": 1730000000002,
            },
        ).status_code == 204

    with _client(database_path) as restarted_client:
        history = restarted_client.get("/api/conversations")
        assert history.json()[0]["question_count"] == 1
        messages = restarted_client.get("/api/conversations/newchat12345/messages").json()
        assert messages[0]["content"] == "Where is the leave policy?"
        assert messages[1]["sources"] == [{"source": "leave.pdf", "page": 2}]
        assert messages[1]["citations"] == [{"citation": "leave.pdf, p. 2", "verified": True}]
        assert messages[1]["grounded"] is True
        assert restarted_client.delete("/api/conversations/newchat12345").status_code == 204
        assert restarted_client.get("/api/conversations").json() == []
        assert restarted_client.post(
            "/api/conversations",
            json={"id": "secondchat123", "title": "Another", "updated": 1730000000003},
        ).status_code == 200
        assert restarted_client.delete("/api/conversations").json() == {"deleted_count": 1}
