"""Request and response models of the HR Assistant API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_QUESTION_CHARS = 2000


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    conversation_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9_-]{8,64}$",
        description="Client-generated id. Requests with the same id share conversation context.",
    )
    top_k: int | None = Field(default=None, ge=1, le=20, description="Override TOP_K for this question")

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class ConversationCreate(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    title: str = Field(max_length=60)
    updated: int = Field(ge=0)


class HistoryMessageCreate(BaseModel):
    position: int = Field(ge=0)
    role: Literal["user", "assistant"]
    content: str
    metadata: dict[str, object] = Field(default_factory=dict)
    updated: int = Field(ge=0)


class LegacyHistoryMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["user", "assistant"]
    content: str


class LegacyConversation(BaseModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{8,64}$")
    title: str = Field(max_length=60)
    updated: int = Field(ge=0)
    messages: list[LegacyHistoryMessage]


class LegacyHistoryImport(BaseModel):
    conversations: list[LegacyConversation]


class SourceOut(BaseModel):
    """One retrieved excerpt that the answer was generated from."""

    source: str
    doc_type: str | None = None
    page: int | None = None
    record_id: str | None = None
    title: str | None = None
    citation: str
    relevance: float = Field(ge=0.0, le=1.0)
    excerpt: str


class CitationVerificationOut(BaseModel):
    """Verification status for one citation found in the generated answer."""

    citation: str
    source: str
    filename: str
    page: int | None = None
    record_id: str | None = None
    verified: bool
    status: Literal["verified", "unverified"]


class ChatResponse(BaseModel):
    answer: str
    conversation_id: str | None = None
    standalone_question: str | None = Field(
        default=None,
        description="Set when a follow-up was rewritten using earlier turns; this is what was searched for",
    )
    grounded: bool = Field(description="True when the LLM answered from retrieved excerpts")
    index_empty: bool = False
    sources: list[SourceOut] = Field(default_factory=list)
    citations: list[CitationVerificationOut] = Field(default_factory=list)
    all_verified: bool = True
    elapsed_ms: int


class DocumentInfo(BaseModel):
    name: str
    doc_type: str
    size_bytes: int | None = None
    modified: str | None = Field(default=None, description="UTC ISO-8601 time of the file on disk")
    chunks: int = Field(description="Chunks of this file currently in the index")
    status: Literal["indexed", "not_indexed", "index_only"] = Field(
        description="indexed: file and chunks present; not_indexed: file present but no chunks; "
        "index_only: chunks exist but the file is no longer in data/raw"
    )
    sensitive: bool = False


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
    sensitive_visible: bool


class UploadResponse(BaseModel):
    filename: str
    doc_type: str
    status: Literal["indexed"] = "indexed"
    replaced: bool = Field(description="True if a file with this name already existed")
    sections: int = Field(description="PDF pages / DOCX documents / CSV rows loaded")
    chunks_created: int
    chunks_added: int
    duplicates_skipped: int
    stale_removed: int
    sensitive: bool = False
    message: str


class OllamaHealth(BaseModel):
    reachable: bool
    embedding_model: str
    embedding_model_available: bool
    llm_model: str
    llm_model_available: bool
    error: str | None = None


class GroqHealth(BaseModel):
    reachable: bool
    model: str
    model_available: bool
    error: str | None = None


class EmbeddingHealth(BaseModel):
    available: bool
    model: str
    dimension: int | None = None
    error: str | None = None


class VectorStoreHealth(BaseModel):
    available: bool
    collection: str
    chunk_count: int | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    ollama: OllamaHealth
    vector_store: VectorStoreHealth
    documents_on_disk: int
    employee_data_enabled: bool = Field(description="True when an admin key is configured on the server")
    uploads_require_admin: bool = True
    groq: GroqHealth
    sentence_transformers: EmbeddingHealth


class AuthStatus(BaseModel):
    admin: bool
    employee_data_enabled: bool
    uploads_require_admin: bool = True


class ErrorResponse(BaseModel):
    detail: str
