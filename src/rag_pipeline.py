"""End-to-end RAG pipeline (HR policies and employee records).

Question -> Retriever -> formatted context -> prompt -> Groq -> answer + sources.

If retrieval finds nothing relevant the LLM is NOT called; a fixed "not enough
information" answer is returned. This is the cheapest and most reliable way to
avoid hallucinated answers for out-of-scope questions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from src.config import Settings
from src.embeddings import get_embeddings
from src.llm import get_llm
from src.prompts import INSUFFICIENT_CONTEXT_MESSAGE, RAG_PROMPT, format_context
from src.retriever import RetrievalResult, RetrievedChunk, Retriever
from src.reranker import get_reranker
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class RAGPipelineError(RuntimeError):
    """Raised when retrieval or generation fails."""


@dataclass(frozen=True)
class SourceReference:
    """One cited location (file + optional page)."""

    source: str
    page: int | None
    score: float
    record_id: str | None = None

    def __str__(self) -> str:
        if self.page is not None:
            location = f"p. {self.page}"
        elif self.record_id is not None:
            location = f"record {self.record_id}"
        else:
            location = "page n/a"
        return f"{self.source} - {location} (relevance {self.score:.2f})"


@dataclass(frozen=True)
class AnswerResult:
    """Final answer plus the evidence it was based on."""

    question: str
    answer: str
    sources: list[SourceReference] = field(default_factory=list)
    retrieval: RetrievalResult | None = None
    grounded: bool = False

    def format_sources(self) -> str:
        """Human-readable source list ('' when there are none)."""
        if not self.sources:
            return ""
        return "\n".join(f"  - {ref}" for ref in self.sources)


def extract_sources(chunks: list[RetrievedChunk]) -> list[SourceReference]:
    """Unique (file, page or record) references, in relevance order, with their best score.

    Sources come from retrieval metadata, never from the model's text, so the
    LLM cannot invent entries in this list.
    """
    seen: dict[tuple[str, int | None, str | None], SourceReference] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page, chunk.record_id)
        if key not in seen:
            seen[key] = SourceReference(chunk.source, chunk.page, chunk.score, chunk.record_id)
    return list(seen.values())


class RAGPipeline:
    """Retrieval-augmented question answering over the indexed papers."""

    def __init__(self, retriever: Retriever, llm: BaseChatModel) -> None:
        self._retriever = retriever
        self._chain = RAG_PROMPT | llm | StrOutputParser()

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGPipeline":
        """Wire up embeddings, vector store, retriever and LLM from settings."""
        store = VectorStoreManager(
            persist_directory=settings.chroma_db_path,
            embeddings=get_embeddings(settings),
            collection_name=settings.collection_name,
        )
        reranker = get_reranker() if settings.reranker_enabled else None
        retriever = Retriever(
            store,
            settings.top_k,
            settings.min_relevance_score,
            reranker=reranker,
        )
        return cls(retriever, get_llm(settings))

    def ask(
        self, question: str, top_k: int | None = None, where: dict[str, Any] | None = None
    ) -> AnswerResult:
        """Answer ``question`` from the indexed documents.

        Raises:
            ValueError: If the question is empty.
            RAGPipelineError: If retrieval or generation fails.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        question = question.strip()

        try:
            retrieval = self._retriever.retrieve(question, top_k=top_k, where=where)
        except ValueError:
            raise
        except Exception as exc:
            logger.error("Retrieval failed (%s)", type(exc).__name__)
            raise RAGPipelineError(
                f"Retrieval failed ({type(exc).__name__})."
            ) from exc

        if retrieval.index_empty:
            return AnswerResult(
                question=question,
                answer="The document index is empty. Add files to data/raw/ and run scripts/ingest.py first.",
                retrieval=retrieval,
            )
        if not retrieval.is_sufficient:
            return AnswerResult(
                question=question,
                answer=INSUFFICIENT_CONTEXT_MESSAGE,
                retrieval=retrieval,
            )

        try:
            answer = self._chain.invoke(
                {"context": format_context(retrieval.chunks), "question": question}
            )
        except Exception as exc:
            logger.error("Generation failed (%s)", type(exc).__name__)
            detail = str(exc) if isinstance(exc, RuntimeError) else "Groq request failed."
            raise RAGPipelineError(f"Answer generation failed: {detail}") from exc

        return AnswerResult(
            question=question,
            answer=answer.strip(),
            sources=extract_sources(retrieval.chunks),
            retrieval=retrieval,
            grounded=True,
        )
