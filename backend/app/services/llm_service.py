"""LLM service abstraction for AI Debugging Assistant.

This module provides a single, testable entry point for OpenAI chat
completion calls, including streaming support. The implementation keeps the
OpenAI dependency isolated so the rest of the application can depend on a
stable interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence, Union

from openai import AsyncOpenAI


@dataclass(frozen=True)
class LLMMessage:
    """Represents a single chat message sent to or returned from the model."""

    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Represents a non-streaming response from the LLM service."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None


class LLMServiceError(RuntimeError):
    """Raised when the LLM service cannot complete a request."""


class LLMService:
    """Provides chat completion helpers with optional streaming output."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o") -> None:
        """Initialize the service with an OpenAI client and model name."""

        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        """Return the configured model name."""

        return self._model

    @staticmethod
    def _normalize_messages(messages: Sequence[Union[LLMMessage, Dict[str, str]]]) -> List[Dict[str, str]]:
        """Convert supported message formats into the OpenAI request shape."""

        normalized_messages: List[Dict[str, str]] = []
        for message in messages:
            if isinstance(message, LLMMessage):
                normalized_messages.append({"role": message.role, "content": message.content})
                continue

            role = message.get("role")
            content = message.get("content")
            if not role or content is None:
                raise ValueError("Each message must include 'role' and 'content'.")
            normalized_messages.append({"role": role, "content": content})
        return normalized_messages

    async def generate_response(
        self,
        messages: Sequence[Union[LLMMessage, Dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a non-streaming assistant response for the given messages."""

        request_messages = self._normalize_messages(messages)
        if system_prompt:
            request_messages = [{"role": "system", "content": system_prompt}, *request_messages]

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Failed to generate LLM response: {exc}") from exc

        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice and choice.message and choice.message.content else ""
        usage_payload: Optional[Dict[str, int]] = None
        if response.usage is not None:
            usage_payload = {
                "prompt_tokens": int(response.usage.prompt_tokens or 0),
                "completion_tokens": int(response.usage.completion_tokens or 0),
                "total_tokens": int(response.usage.total_tokens or 0),
            }

        return LLMResponse(content=content, model=self._model, usage=usage_payload)

    async def stream_response(
        self,
        messages: Sequence[Union[LLMMessage, Dict[str, str]]],
        *,
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream assistant tokens as they are produced by the model."""

        request_messages = self._normalize_messages(messages)
        if system_prompt:
            request_messages = [{"role": "system", "content": system_prompt}, *request_messages]

        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=request_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError(f"Failed to start LLM stream: {exc}") from exc

        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            token = delta.content if delta and delta.content else ""
            if token:
                yield token


__all__ = [
    "LLMMessage",
    "LLMResponse",
    "LLMService",
    "LLMServiceError",
]
