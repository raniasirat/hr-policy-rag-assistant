"""In-memory conversation context for follow-up questions.

The existing ``RAGPipeline.ask`` answers one self-contained question and keeps no
history. To support follow-ups such as "And for juniors?", the API remembers the
last few turns of each conversation and, when there are earlier turns, asks the LLM
to rewrite the new question into a standalone question *before* retrieval. The
rewritten question is then passed to the unchanged pipeline.

Context lives only in server memory (lost on restart, expires after a period of
inactivity). It is keyed by ``(conversation_id, is_admin)`` so a conversation held
with administrator rights can never leak into a request without them.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

_ANSWER_CHARS_KEPT = 600
_MAX_REWRITE_CHARS = 400

REWRITE_SYSTEM_PROMPT = (
    "You rewrite follow-up questions for a search system. You are given an earlier conversation "
    "and a new question. Rewrite the new question so that it can be understood on its own, "
    "replacing pronouns and implicit references with what they refer to in the conversation. "
    "Keep the language of the new question. If the new question is already self-contained, "
    "return it unchanged. Output ONLY the rewritten question on one line. Never answer it."
)


@dataclass(frozen=True)
class Turn:
    """One question and the answer that was given."""

    question: str
    answer: str


class ConversationStore:
    """Thread-safe store of recent turns with a size cap and an inactivity timeout."""

    def __init__(
        self,
        max_turns: int = 4,
        max_conversations: int = 500,
        ttl_seconds: float = 3600.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_turns = max_turns
        self._max_conversations = max_conversations
        self._ttl = ttl_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._items: OrderedDict[tuple[str, bool], tuple[float, deque[Turn]]] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self._max_turns > 0

    def _evict_expired(self, now: float) -> None:
        expired = [key for key, (stamp, _) in self._items.items() if now - stamp > self._ttl]
        for key in expired:
            del self._items[key]

    def get(self, conversation_id: str, admin: bool) -> list[Turn]:
        """Return the remembered turns (oldest first); empty if unknown or expired."""
        if not self.enabled:
            return []
        with self._lock:
            self._evict_expired(self._clock())
            entry = self._items.get((conversation_id, admin))
            return list(entry[1]) if entry else []

    def add(self, conversation_id: str, admin: bool, question: str, answer: str) -> None:
        """Remember a turn, dropping the oldest turns / conversations beyond the limits."""
        if not self.enabled:
            return
        now = self._clock()
        key = (conversation_id, admin)
        turn = Turn(question=question, answer=answer[:_ANSWER_CHARS_KEPT])
        with self._lock:
            self._evict_expired(now)
            turns = self._items[key][1] if key in self._items else deque(maxlen=self._max_turns)
            turns.append(turn)
            self._items[key] = (now, turns)
            self._items.move_to_end(key)
            while len(self._items) > self._max_conversations:
                self._items.popitem(last=False)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def condense_question(llm: BaseChatModel, history: list[Turn], question: str) -> str:
    """Rewrite ``question`` as a standalone question using the earlier turns.

    Falls back to the original question if the model fails or returns something
    unusable (empty, multi-paragraph or much longer than the input), so a rewrite
    problem can never block an answer.
    """
    if not history:
        return question
    transcript = "\n".join(f"User: {t.question}\nAssistant: {t.answer}" for t in history)
    messages = [
        SystemMessage(content=REWRITE_SYSTEM_PROMPT),
        HumanMessage(content=f"Conversation:\n{transcript}\n\nNew question: {question}\n\nStandalone question:"),
    ]
    try:
        raw = llm.invoke(messages).content
    except Exception:  # any LLM/network failure: let the pipeline report it properly
        logger.warning("Question rewriting failed; using the original question", exc_info=True)
        return question
    text = raw if isinstance(raw, str) else ""
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    rewritten = lines[0] if lines else ""
    for prefix in ("Standalone question:", "Rewritten question:", "Question:"):
        if rewritten.lower().startswith(prefix.lower()):
            rewritten = rewritten[len(prefix):].strip()
    rewritten = rewritten.strip("\"'“”« »")
    if not rewritten or len(rewritten) > max(_MAX_REWRITE_CHARS, 3 * len(question)):
        logger.info("Unusable rewrite; using the original question")
        return question
    return rewritten
