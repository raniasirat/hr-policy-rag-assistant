"""Top-k semantic retrieval with a relevance threshold."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document

from src.reranker import PassageReranker
from src.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with the metadata needed for citation."""

    content: str
    source: str
    page: int | None
    title: str | None
    doc_type: str | None
    score: float
    record_id: str | None = None
    chunk_id: str | None = None

    @classmethod
    def from_document(cls, document: Document, score: float) -> "RetrievedChunk":
        metadata = document.metadata
        page = metadata.get("page")
        return cls(
            content=document.page_content,
            source=str(metadata.get("source", "unknown")),
            page=int(page) if page is not None else None,
            title=metadata.get("title"),
            doc_type=metadata.get("doc_type"),
            score=score,
            record_id=str(metadata["record_id"]) if metadata.get("record_id") else None,
            chunk_id=str(metadata["chunk_id"]) if metadata.get("chunk_id") else None,
        )

    @property
    def citation(self) -> str:
        """Short citation label such as ``policy.pdf, p. 2`` or ``staff.csv, record E01``."""
        if self.page is not None:
            return f"{self.source}, p. {self.page}"
        if self.record_id is not None:
            return f"{self.source}, record {self.record_id}"
        return self.source


@dataclass(frozen=True)
class RetrievalResult:
    """Result of a retrieval call.

    Attributes:
        chunks: Chunks at or above the relevance threshold, ranked best first.
        best_score: Highest relevance among *all* candidates (0 if none).
        index_empty: True if nothing has been indexed yet.
    """

    query: str
    chunks: list[RetrievedChunk] = field(default_factory=list)
    best_score: float = 0.0
    index_empty: bool = False

    @property
    def is_sufficient(self) -> bool:
        """True when at least one chunk passed the relevance threshold."""
        return bool(self.chunks)


class Retriever:
    """Retrieves the most relevant chunks for a question."""

    def __init__(
        self,
        store: VectorStoreManager,
        top_k: int = 5,
        min_relevance_score: float = 0.30,
        reranker: PassageReranker | None = None,
        candidate_k: int = 15,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0.0 <= min_relevance_score <= 1.0:
            raise ValueError("min_relevance_score must be between 0 and 1")
        if candidate_k < 1:
            raise ValueError("candidate_k must be >= 1")
        self._store = store
        self._top_k = top_k
        self._min_relevance_score = min_relevance_score
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(
        self, query: str, top_k: int | None = None, where: dict[str, Any] | None = None
    ) -> RetrievalResult:
        """Return the top-k chunks whose relevance passes the threshold.

        Raises:
            ValueError: If the query is blank or ``top_k`` is invalid.
        """
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")
        k = top_k if top_k is not None else self._top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")

        candidate_k = max(k, self._candidate_k) if self._reranker is not None else k
        hits = self._store.similarity_search(query, k=candidate_k, where=where)
        if not hits:
            return RetrievalResult(query=query, index_empty=self._store.count() == 0)

        relevant = [
            RetrievedChunk.from_document(doc, score)
            for doc, score in hits
            if score >= self._min_relevance_score
        ]
        best = max(score for _, score in hits)
        if self._reranker is not None and relevant:
            rerank_scores = self._reranker.score(query, [chunk.content for chunk in relevant])
            if len(rerank_scores) != len(relevant):
                raise RuntimeError("Reranker returned a different number of scores than passages.")
            relevant = [
                chunk
                for _, chunk in sorted(
                    zip(rerank_scores, relevant),
                    key=lambda scored_chunk: scored_chunk[0],
                    reverse=True,
                )[:k]
            ]
        else:
            relevant = relevant[:k]

        logger.debug(
            "Query %r: %d candidate(s), %d returned above threshold %.2f (best %.3f)",
            query,
            len(hits),
            len(relevant),
            self._min_relevance_score,
            best,
        )
        return RetrievalResult(query=query, chunks=relevant, best_score=best)
