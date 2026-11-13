"""Groq chat model backed by the official Groq Python SDK."""

from __future__ import annotations

import os
from typing import Any

from groq import APIConnectionError, APIStatusError, APITimeoutError, Groq, RateLimitError
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from src.config import ConfigurationError, Settings


class GroqRequestError(RuntimeError):
    """Safe, credential-free description of a failed Groq request."""


def _safe_groq_error(exc: Exception, model: str) -> GroqRequestError:
    if isinstance(exc, RateLimitError):
        message = "Groq rate limit exceeded."
    elif isinstance(exc, APITimeoutError):
        message = "Groq request timed out."
    elif isinstance(exc, APIConnectionError):
        message = "Could not connect to Groq."
    elif isinstance(exc, APIStatusError) and exc.status_code == 404:
        message = f"Groq model '{model}' is unavailable to this account."
    elif isinstance(exc, APIStatusError):
        message = f"Groq request failed with HTTP {exc.status_code}."
    else:
        message = "Groq request failed."
    return GroqRequestError(message)


class GroqChatModel(BaseChatModel):
    """LangChain chat-model adapter that calls Groq's official SDK."""

    model: str
    temperature: float = 0.0
    _client: Groq = PrivateAttr()

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        client: Groq | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, temperature=0.0, **kwargs)
        if client is None:
            if not api_key:
                raise ConfigurationError("GROQ_API_KEY must be set in the environment.")
            client = Groq(api_key=api_key, timeout=90.0, max_retries=0)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "groq"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": self.model}

    @staticmethod
    def _message_payload(message: BaseMessage) -> dict[str, str]:
        role = {"human": "user", "ai": "assistant"}.get(message.type, message.type)
        if role not in {"system", "user", "assistant"} or not isinstance(message.content, str):
            raise ValueError(f"Unsupported Groq chat message type: {message.type}")
        return {"role": role, "content": message.content}

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        payload = [self._message_payload(message) for message in messages]
        request_options: dict[str, Any] = {
            "model": self.model,
            "messages": payload,
            "temperature": self.temperature,
        }
        if stop:
            request_options["stop"] = stop
        try:
            response = self._client.chat.completions.create(**request_options)
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _safe_groq_error(exc, self.model) from None
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise GroqRequestError("Groq returned an empty response.")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])

    def list_model_ids(self, *, timeout: float = 5.0) -> set[str]:
        """Return models visible to this key, using a short health-check timeout."""
        try:
            response = self._client.with_options(timeout=timeout, max_retries=0).models.list()
        except (RateLimitError, APITimeoutError, APIConnectionError, APIStatusError) as exc:
            raise _safe_groq_error(exc, self.model) from None
        return {item.id for item in response.data}


def get_llm(settings: Settings) -> GroqChatModel:
    """Create the shared Groq client without retaining or logging its API key."""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise ConfigurationError("GROQ_API_KEY must be set in the environment.")
    return GroqChatModel(model=settings.groq_model, api_key=api_key)
