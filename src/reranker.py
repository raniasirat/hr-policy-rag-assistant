"""Local cross-encoder reranking for retrieved passages."""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Protocol

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RerankerError(RuntimeError):
    """Raised when the local reranker cannot load or score passages."""


class PassageReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score for each passage, in input order."""


class CrossEncoderReranker:
    """Sentence Transformers CrossEncoder pinned to CPU."""

    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name = model_name
        try:
            self._model = CrossEncoder(
                model_name,
                device="cpu",
                local_files_only=True,
            )
        except Exception as exc:
            logger.error("Could not load local reranker (%s)", type(exc).__name__)
            raise RerankerError(
                f"Could not load local reranker model '{model_name}' ({type(exc).__name__})."
            ) from exc

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []

        pairs = [(query, passage) for passage in passages]
        try:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
            scores = [float(score) for score in raw_scores]
        except Exception as exc:
            logger.error("Could not score passages with reranker (%s)", type(exc).__name__)
            raise RerankerError(f"Reranker inference failed ({type(exc).__name__}).") from exc

        if len(scores) != len(passages) or any(not math.isfinite(score) for score in scores):
            raise RerankerError("Reranker returned invalid scores.")
        return scores


@lru_cache(maxsize=1)
def get_reranker(model_name: str = RERANKER_MODEL) -> CrossEncoderReranker:
    """Load one local CPU model instance per process."""
    return CrossEncoderReranker(model_name)
