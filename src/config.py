"""Application configuration.

Settings are read from environment variables (optionally loaded from a ``.env``
file with python-dotenv) and validated with Pydantic. Every other module
receives a :class:`Settings` instance instead of reading the environment
itself, which keeps the code easy to test.
"""

from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"

_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
_COLLECTION_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{1,61}[A-Za-z0-9]$")

# Maps Settings field names to environment variable names.
_ENV_VARS: dict[str, str] = {
    "ollama_base_url": "OLLAMA_BASE_URL",
    "embedding_model": "EMBEDDING_MODEL",
    "llm_model": "LLM_MODEL",
    "groq_model": "GROQ_MODEL",
    "llm_num_ctx": "LLM_NUM_CTX",
    "chunk_size": "CHUNK_SIZE",
    "chunk_overlap": "CHUNK_OVERLAP",
    "top_k": "TOP_K",
    "min_relevance_score": "MIN_RELEVANCE_SCORE",
    "reranker_enabled": "RERANKER_ENABLED",
    "chroma_db_path": "CHROMA_DB_PATH",
    "collection_name": "COLLECTION_NAME",
    "log_level": "LOG_LEVEL",
}


class ConfigurationError(ValueError):
    """Raised when the environment configuration is invalid."""


class Settings(BaseModel):
    """Validated, immutable application settings."""

    model_config = ConfigDict(frozen=True)

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "qwen2.5:3b"
    groq_model: str = "openai/gpt-oss-120b"
    llm_num_ctx: int = Field(default=4096, ge=512)
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    top_k: int = Field(default=5, ge=1, le=50)
    min_relevance_score: float = Field(default=0.30, ge=0.0, le=1.0)
    reranker_enabled: bool = False
    chroma_db_path: Path = Field(default=Path("storage/chroma_db"), validate_default=True)
    collection_name: str = "hr_documents"
    log_level: str = "INFO"
    raw_data_dir: Path = RAW_DATA_DIR
    processed_data_dir: Path = PROCESSED_DATA_DIR

    @field_validator("ollama_base_url")
    @classmethod
    def _validate_base_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("must start with http:// or https://")
        return value

    @field_validator("embedding_model")
    @classmethod
    def _normalize_embedding_model(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return "BAAI/bge-m3" if value.casefold() == "bge-m3" else value

    @field_validator("llm_model", "groq_model")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("collection_name")
    @classmethod
    def _validate_collection_name(cls, value: str) -> str:
        if not _COLLECTION_NAME_RE.match(value):
            raise ValueError(
                "must be 3-63 characters, start/end with a letter or digit and "
                "contain only letters, digits, '_' or '-'"
            )
        return value

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, value: str) -> str:
        value = value.strip().upper()
        if value not in _VALID_LOG_LEVELS:
            raise ValueError(f"must be one of {sorted(_VALID_LOG_LEVELS)}")
        return value

    @field_validator("chroma_db_path")
    @classmethod
    def _resolve_chroma_path(cls, value: Path) -> Path:
        # Relative paths are interpreted relative to the project root so the
        # scripts work no matter which directory they are launched from.
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()

    @model_validator(mode="after")
    def _validate_chunking(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})"
            )
        return self


def load_settings(env_file: Path | None = None) -> Settings:
    """Load and validate settings from the environment and an optional .env file.

    Real environment variables take precedence over values in the .env file.
    Unset or empty variables fall back to the defaults declared on
    :class:`Settings`.

    Raises:
        ConfigurationError: If any value is invalid.
    """
    load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)
    values: dict[str, str] = {}
    for field_name, env_name in _ENV_VARS.items():
        raw_value = os.environ.get(env_name)
        if raw_value is not None and raw_value.strip() != "":
            values[field_name] = raw_value.strip()
    try:
        return Settings(**values)
    except ValidationError as exc:
        problems = "; ".join(
            f"{_ENV_VARS.get(str(err['loc'][0]), err['loc'])}: {err['msg']}" if err["loc"] else err["msg"]
            for err in exc.errors()
        )
        raise ConfigurationError(f"Invalid configuration: {problems}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return process-wide settings (loaded once)."""
    return load_settings()


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging for the command-line scripts."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # Keep third-party libraries quiet unless the user asks for DEBUG output.
    if level.upper() != "DEBUG":
        for noisy in ("httpx", "httpcore", "chromadb"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
