"""SQLite persistence for the browser-visible conversation history."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


class ConversationHistoryStore:
    """Persist conversations and their ordered messages in a local SQLite database."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    @contextmanager
    def _connection(self) -> Generator[sqlite3.Connection, None, None]:
        connection = sqlite3.connect(self.database_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    updated INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    conversation_id TEXT NOT NULL,
                    position INTEGER NOT NULL CHECK (position >= 0),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    PRIMARY KEY (conversation_id, position),
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated DESC);
                """
            )

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.id, c.title, c.updated,
                       COUNT(m.position) AS message_count,
                       COALESCE(SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END), 0)
                           AS question_count
                FROM conversations AS c
                LEFT JOIN messages AS m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated DESC, c.id
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, title, updated FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_messages(self, conversation_id: str) -> list[dict[str, Any]] | None:
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                return None
            rows = connection.execute(
                """
                SELECT position, role, content, metadata_json
                FROM messages WHERE conversation_id = ? ORDER BY position
                """,
                (conversation_id,),
            ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            message = {
                "position": row["position"],
                "role": row["role"],
                "content": row["content"],
            }
            message.update(json.loads(row["metadata_json"]))
            messages.append(message)
        return messages

    def create_conversation(self, conversation_id: str, title: str, updated: int) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO conversations (id, title, updated)
                VALUES (?, ?, ?)
                """,
                (conversation_id, title, updated),
            )

    def save_message(
        self,
        conversation_id: str,
        position: int,
        role: str,
        content: str,
        metadata: dict[str, Any],
        updated: int,
    ) -> bool:
        metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        with self._connection() as connection:
            exists = connection.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
            ).fetchone()
            if not exists:
                return False
            connection.execute(
                """
                INSERT INTO messages (conversation_id, position, role, content, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (conversation_id, position) DO UPDATE SET
                    role = excluded.role,
                    content = excluded.content,
                    metadata_json = excluded.metadata_json
                """,
                (conversation_id, position, role, content, metadata_json),
            )
            connection.execute(
                "UPDATE conversations SET updated = MAX(updated, ?) WHERE id = ?",
                (updated, conversation_id),
            )
        return True

    def import_legacy(self, conversations: list[dict[str, Any]]) -> dict[str, int | bool]:
        imported_messages = 0
        with self._connection() as connection:
            for conversation in conversations:
                conversation_id = conversation["id"]
                connection.execute(
                    """
                    INSERT OR IGNORE INTO conversations (id, title, updated)
                    VALUES (?, ?, ?)
                    """,
                    (conversation_id, conversation["title"], conversation["updated"]),
                )
                stored_conversation = connection.execute(
                    "SELECT title, updated FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if (
                    stored_conversation is None
                    or stored_conversation["title"] != conversation["title"]
                    or stored_conversation["updated"] < conversation["updated"]
                ):
                    raise ValueError(f"Existing history conflicts with conversation {conversation_id}")
                for position, message in enumerate(conversation["messages"]):
                    metadata = message["metadata"]
                    metadata_json = json.dumps(metadata, ensure_ascii=False, sort_keys=True)
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO messages
                            (conversation_id, position, role, content, metadata_json)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            conversation_id,
                            position,
                            message["role"],
                            message["content"],
                            metadata_json,
                        ),
                    )
                    stored = connection.execute(
                        """
                        SELECT role, content, metadata_json FROM messages
                        WHERE conversation_id = ? AND position = ?
                        """,
                        (conversation_id, position),
                    ).fetchone()
                    if (
                        stored is None
                        or stored["role"] != message["role"]
                        or stored["content"] != message["content"]
                        or json.loads(stored["metadata_json"]) != metadata
                    ):
                        raise ValueError(
                            f"Existing history conflicts with imported message "
                            f"{conversation_id}:{position}"
                        )
                    imported_messages += 1
        return {
            "conversation_count": len(conversations),
            "message_count": imported_messages,
            "verified": True,
        }

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._connection() as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def delete_all(self) -> int:
        with self._connection() as connection:
            cursor = connection.execute("DELETE FROM conversations")
        return cursor.rowcount
