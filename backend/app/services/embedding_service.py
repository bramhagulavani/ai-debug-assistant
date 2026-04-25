"""Embedding Service for AI Debugging Assistant.

Converts error events (error type + message + code context) into
high-dimensional vectors using OpenAI's text-embedding-3-small model.

These vectors are stored in Pinecone and queried at debug time to find
semantically similar past bugs — even if the error message is worded
differently from before.

Why text-embedding-3-small:
    - 1536 dimensions — enough for semantic nuance
    - Fastest and cheapest OpenAI embedding model
    - Outperforms ada-002 on code-related similarity tasks
"""

from __future__ import annotations

import logging
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI

from backend.app.core.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Embedding model — do not change without updating Pinecone index dimensions
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ---------------------------------------------------------------------------
# Embedding input builder
# ---------------------------------------------------------------------------

def build_embedding_text(
    error_type: str,
    error_message: str,
    language: str,
    error_function: Optional[str] = None,
    filename: Optional[str] = None,
    code_window: Optional[str] = None,
) -> str:
    """
    Builds a single string that captures the semantic meaning of a bug.

    This string is what gets embedded into a vector. The quality of
    similarity search depends entirely on what goes into this string —
    it must capture the ESSENCE of the bug, not just the error message.

    Strategy:
        - Lead with language + error type (most discriminating features)
        - Add error message (the human-readable description)
        - Add function name (narrows down where in code it happened)
        - Add code window (actual code context — makes vectors unique)

    Args:
        error_type:     e.g. "IndexError", "TypeError"
        error_message:  e.g. "list index out of range"
        language:       e.g. "python", "javascript"
        error_function: e.g. "get_user" (optional)
        filename:       e.g. "main.py" (optional)
        code_window:    The ±10 lines of code around the error (optional)

    Returns:
        A single string ready to be embedded.
    """
    parts: list[str] = [
        f"Language: {language}",
        f"Error type: {error_type}",
        f"Error message: {error_message}",
    ]

    if error_function:
        parts.append(f"Function: {error_function}")

    if filename:
        parts.append(f"File: {filename}")

    if code_window:
        # Limit code window to 500 chars — enough context without noise
        trimmed = code_window[:500].strip()
        parts.append(f"Code context:\n{trimmed}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Embedding service
# ---------------------------------------------------------------------------

class EmbeddingService:
    """
    Converts text into vector embeddings using OpenAI's embedding API.

    Usage:
        svc = EmbeddingService()
        vector = await svc.embed_error(
            error_type="IndexError",
            error_message="list index out of range",
            language="python",
            error_function="get_user",
        )
        # vector is a list of 1536 floats
    """

    def __init__(self) -> None:
        self._client = AsyncOpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
        self._model = EMBEDDING_MODEL

    async def embed_text(self, text: str) -> list[float]:
        """
        Embeds a raw text string into a 1536-dimensional vector.

        Args:
            text: Any text string to embed.

        Returns:
            List of 1536 floats representing the semantic vector.

        Raises:
            RuntimeError: If the embedding API call fails.
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        try:
            response = await self._client.embeddings.create(
                model=self._model,
                input=text.strip(),
            )
            return response.data[0].embedding

        except Exception as exc:
            logger.error("Embedding API call failed: %s", exc)
            raise RuntimeError(f"Failed to generate embedding: {exc}") from exc

    async def embed_error(
        self,
        error_type: str,
        error_message: str,
        language: str,
        error_function: Optional[str] = None,
        filename: Optional[str] = None,
        code_window: Optional[str] = None,
    ) -> list[float]:
        """
        Embeds a structured error event into a vector.

        Builds the embedding text from all available error context,
        then calls the OpenAI embedding API.

        Args:
            error_type:     The error class name (e.g. "IndexError").
            error_message:  The error message string.
            language:       Programming language of the code.
            error_function: Name of the function where error occurred.
            filename:       Source filename.
            code_window:    Lines of code surrounding the error.

        Returns:
            List of 1536 floats — the semantic vector for this bug.
        """
        text = build_embedding_text(
            error_type=error_type,
            error_message=error_message,
            language=language,
            error_function=error_function,
            filename=filename,
            code_window=code_window,
        )

        logger.debug("Embedding error: %s — %s", error_type, error_message)
        return await self.embed_text(text)

    @staticmethod
    def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
        """
        Computes cosine similarity between two vectors.
        Used for local testing — in production Pinecone does this.

        Args:
            vec_a: First vector (list of floats).
            vec_b: Second vector (list of floats).

        Returns:
            Float between -1.0 and 1.0.
            1.0 = identical, 0.0 = unrelated, -1.0 = opposite.
        """
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have the same dimensions.")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = sum(a ** 2 for a in vec_a) ** 0.5
        magnitude_b = sum(b ** 2 for b in vec_b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0

        return dot_product / (magnitude_a * magnitude_b)


__all__ = [
    "EmbeddingService",
    "build_embedding_text",
    "EMBEDDING_DIMENSIONS",
    "EMBEDDING_MODEL",
]