"""Persistent ChromaDB vector store.

Design notes:
    * The collection uses cosine distance, so ``relevance = 1 - distance`` is a
      meaningful 0..1 score (for non-negative similarities).
    * Chunk ids are deterministic (see ``document_processor.make_chunk_id``).
      Re-indexing unchanged files therefore adds nothing, and when a file has
      changed its outdated chunks are removed.
    * ``reset`` only deletes this application's collection, never arbitrary
      directories.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

_EMBED_BATCH_SIZE = 32


@dataclass(frozen=True)
class IndexingReport:
    """What an ``add_documents`` call actually did."""

    added: int = 0
    skipped_duplicates: int = 0
    removed_stale: int = 0


class VectorStoreManager:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: Path,
        embeddings: Embeddings,
        collection_name: str = "hr_documents",
    ) -> None:
        persist_directory.mkdir(parents=True, exist_ok=True)
        self._embeddings = embeddings
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._store = self._open_store()

    def _open_store(self) -> Chroma:
        return Chroma(
            client=self._client,
            collection_name=self._collection_name,
            embedding_function=self._embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------ queries
    def count(self) -> int:
        """Number of chunks currently indexed."""
        return self._client.get_collection(self._collection_name).count()

    def list_sources(self) -> dict[str, int]:
        """Map each indexed source file name to its number of chunks."""
        records = self._store.get(include=["metadatas"])
        counts: dict[str, int] = defaultdict(int)
        for metadata in records.get("metadatas") or []:
            counts[str((metadata or {}).get("source", "unknown"))] += 1
        return dict(sorted(counts.items()))

    def similarity_search(
        self, query: str, k: int = 5, where: dict[str, Any] | None = None
    ) -> list[tuple[Document, float]]:
        """Return up to ``k`` ``(chunk, relevance)`` pairs, best first.

        Relevance is ``1 - cosine_distance`` clipped to ``[0, 1]``.
        ``where`` is an optional Chroma metadata filter, applied inside the database
        (for example ``{"doc_type": {"$ne": "csv"}}``) so that excluded chunks can
        never reach the prompt.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        total = self.count()
        if total == 0:
            return []
        hits = self._store.similarity_search_with_score(query, k=min(k, total), filter=where)
        return [(doc, max(0.0, min(1.0, 1.0 - float(distance)))) for doc, distance in hits]

    # ---------------------------------------------------------------- mutations
    def _ids_for_source(self, source: str) -> set[str]:
        records = self._store.get(where={"source": source}, include=["metadatas"])
        return set(records.get("ids") or [])

    def add_documents(self, chunks: list[Document]) -> IndexingReport:
        """Index chunks, skipping duplicates and removing stale chunks.

        Every chunk must have ``source`` and ``chunk_id`` metadata. For each
        source file present in ``chunks``, previously indexed chunks that are no
        longer part of the file are deleted (the file was edited).
        """
        by_source: dict[str, dict[str, Document]] = defaultdict(dict)
        for chunk in chunks:
            source = chunk.metadata.get("source")
            chunk_id = chunk.metadata.get("chunk_id")
            if not source or not chunk_id:
                raise ValueError("every chunk needs 'source' and 'chunk_id' metadata")
            by_source[str(source)][str(chunk_id)] = chunk  # in-batch duplicates collapse

        added = skipped = removed = 0
        for source, new_chunks in by_source.items():
            existing_ids = self._ids_for_source(source)
            stale_ids = existing_ids - new_chunks.keys()
            if stale_ids:
                self._store.delete(ids=sorted(stale_ids))
                removed += len(stale_ids)
                logger.info("Removed %d outdated chunk(s) of %s", len(stale_ids), source)

            to_add = [(cid, doc) for cid, doc in new_chunks.items() if cid not in existing_ids]
            skipped += len(new_chunks) - len(to_add)
            for start in range(0, len(to_add), _EMBED_BATCH_SIZE):
                batch = to_add[start : start + _EMBED_BATCH_SIZE]
                self._store.add_documents(
                    documents=[doc for _, doc in batch], ids=[cid for cid, _ in batch]
                )
                added += len(batch)
                logger.info("%s: indexed %d/%d new chunk(s)", source, start + len(batch), len(to_add))
        return IndexingReport(added=added, skipped_duplicates=skipped, removed_stale=removed)

    def delete_source(self, source: str) -> int:
        """Remove every chunk of one source file; returns how many were removed."""
        ids = self._ids_for_source(source)
        if ids:
            self._store.delete(ids=sorted(ids))
        return len(ids)

    def reset(self) -> None:
        """Delete this application's collection and recreate it empty.

        Raw documents in ``data/raw`` are untouched and can be re-ingested.
        """
        try:
            self._client.delete_collection(self._collection_name)
        except Exception as exc:  # collection may not exist yet
            logger.debug("delete_collection ignored: %s", exc)
        self._store = self._open_store()
        logger.info("Index '%s' has been reset", self._collection_name)
