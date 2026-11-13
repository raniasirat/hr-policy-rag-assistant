"""Local Sentence Transformers embeddings and offline test embeddings."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import Settings


class EmbeddingServiceError(RuntimeError):
    """Raised when local embedding generation fails."""


@lru_cache(maxsize=1)
def _load_sentence_transformer(model_name: str) -> SentenceTransformer:
    """Load one process-wide model instance for document and query embeddings."""
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not load local embedding model '{model_name}' ({type(exc).__name__})."
        ) from exc


class SentenceTransformerEmbeddings(Embeddings):
    """LangChain adapter using the same normalized local model for all text."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = _load_sentence_transformer(model_name)

    @property
    def dimension(self) -> int:
        get_dimension = getattr(self._model, "get_embedding_dimension", None)
        if get_dimension is None:
            get_dimension = self._model.get_sentence_embedding_dimension
        dimension = get_dimension()
        if dimension is None:
            raise EmbeddingServiceError(f"Embedding model '{self.model_name}' has no output dimension.")
        return dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        try:
            vectors = self._model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingServiceError(
                f"Local embedding generation failed for '{self.model_name}' ({type(exc).__name__})."
            ) from exc
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([text])[0]


def get_embeddings(settings: Settings) -> SentenceTransformerEmbeddings:
    """Create the local BGE-M3 adapter backed by the cached model instance."""
    return SentenceTransformerEmbeddings(settings.embedding_model)


def verify_embedding_service(embeddings: Embeddings, model_name: str) -> int:
    """Embed a probe string and return its vector dimension."""
    try:
        vector = embeddings.embed_query("connectivity check")
    except Exception as exc:
        raise EmbeddingServiceError(
            f"Could not generate embeddings with local model '{model_name}' ({type(exc).__name__})."
        ) from exc
    return len(vector)


_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be by for from has in is it of on or that the this to was were with what how why which".split()
)


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words hashing embeddings (offline, test-only)."""

    def __init__(self, dimension: int = 512) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN_RE.findall(text.lower()):
            if token in _STOPWORDS:
                continue
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            vector[0] = 1.0
            return vector
        return [value / norm for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
