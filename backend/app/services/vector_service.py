"""Vector Service for AI Debugging Assistant.

Handles all Pinecone operations — storing bug vectors and querying
for similar past bugs. Works together with EmbeddingService:

    EmbeddingService  →  converts error to vector (list of floats)
    VectorService     →  stores that vector in Pinecone
                      →  queries Pinecone for similar vectors
                      →  returns matching past bugs with metadata

Each vector stored in Pinecone has this metadata:
    user_id:        scopes results to the current user
    project_id:     scopes results to the current project
    session_id:     links back to the DebugSession in PostgreSQL
    error_type:     e.g. "IndexError"
    error_message:  e.g. "list index out of range"
    language:       e.g. "python"
    error_function: e.g. "get_user"
    filename:       e.g. "main.py"
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from backend.app.core.config import settings

load_dotenv()

logger = logging.getLogger(__name__)

# Similarity threshold — results below this score are too different
# to be useful. Tuned based on embedding test results (gap at ~0.75)
SIMILARITY_THRESHOLD = 0.75
DEFAULT_TOP_K = 3   # number of similar bugs to return


# ---------------------------------------------------------------------------
# Data model for a similar bug result
# ---------------------------------------------------------------------------

@dataclass
class SimilarBug:
    """
    Represents one similar past bug returned from Pinecone.
    Used to build the RAG context injected into the LLM prompt.
    """

    session_id: str
    score: float
    error_type: str
    error_message: str
    language: str
    error_function: Optional[str]
    filename: Optional[str]

    def to_prompt_string(self) -> str:
        """
        Formats this similar bug as a concise block for LLM injection.
        Tells the AI about a past bug the user already solved.
        """
        lines = [
            f"Past bug (similarity: {self.score:.0%}):",
            f"  Error: {self.error_type} — {self.error_message}",
            f"  Language: {self.language}",
        ]
        if self.error_function:
            lines.append(f"  Function: {self.error_function}")
        if self.filename:
            lines.append(f"  File: {self.filename}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Vector service
# ---------------------------------------------------------------------------

class VectorService:
    """
    Manages all Pinecone vector storage and similarity search operations.

    Usage:
        svc = VectorService()

        # Store a bug vector
        vector_id = await svc.upsert_bug(
            session_id="uuid-string",
            vector=[0.1, 0.2, ...],   # 1536 floats from EmbeddingService
            user_id="uuid-string",
            project_id="uuid-string",
            error_type="IndexError",
            error_message="list index out of range",
            language="python",
        )

        # Find similar bugs
        similar = await svc.query_similar_bugs(
            vector=[0.1, 0.2, ...],
            user_id="uuid-string",
            project_id="uuid-string",
        )
    """

    def __init__(self) -> None:
        if not settings.pinecone_api_key:
            raise RuntimeError(
                "PINECONE_API_KEY is not set. Add it to your .env file."
            )

        self._pc        = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._index     = self._get_or_create_index()

    def _get_or_create_index(self):
        """
        Returns the Pinecone index, creating it if it does not exist.
        Uses serverless spec on AWS us-east-1 (free tier compatible).
        Recreates the index if it exists with the wrong dimension.
        """
        existing_indexes = [i.name for i in self._pc.list_indexes()]

        if self._index_name in existing_indexes:
            # Index exists — check dimension
            index_desc = self._pc.describe_index(self._index_name)
            existing_dim = index_desc.dimension
            if existing_dim != 1536:
                logger.warning(
                    "Index '%s' exists with dimension %d, but need 1536. "
                    "Deleting and recreating...",
                    self._index_name,
                    existing_dim,
                )
                self._pc.delete_index(self._index_name)
                self._pc.create_index(
                    name=self._index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                logger.info("Recreated index '%s' with dimension 1536.", self._index_name)
            else:
                logger.info("Index '%s' found with correct dimension.", self._index_name)
        else:
            logger.info(
                "Index '%s' not found — creating it now.",
                self._index_name,
            )
            self._pc.create_index(
                name=self._index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info("Index '%s' created with dimension 1536.", self._index_name)

        return self._pc.Index(self._index_name)

    # -----------------------------------------------------------------------
    # Upsert — store a bug vector
    # -----------------------------------------------------------------------

    async def upsert_bug(
        self,
        session_id: str,
        vector: list[float],
        user_id: str,
        project_id: str,
        error_type: str,
        error_message: str,
        language: str,
        error_function: Optional[str] = None,
        filename: Optional[str] = None,
    ) -> str:
        """
        Stores a bug vector in Pinecone with full metadata.

        The vector ID is a new UUID — stored back on the DebugSession
        in PostgreSQL so we can cross-reference later.

        Args:
            session_id:     PostgreSQL DebugSession UUID as string.
            vector:         1536-float embedding from EmbeddingService.
            user_id:        Owner user UUID as string.
            project_id:     Owner project UUID as string.
            error_type:     Error class name.
            error_message:  Error message string.
            language:       Programming language.
            error_function: Function where error occurred (optional).
            filename:       Source filename (optional).

        Returns:
            The vector ID string stored in Pinecone.
        """
        vector_id = str(uuid.uuid4())

        metadata = {
            "session_id":     session_id,
            "user_id":        user_id,
            "project_id":     project_id,
            "error_type":     error_type,
            "error_message":  error_message,
            "language":       language,
            "error_function": error_function or "",
            "filename":       filename or "",
        }

        try:
            self._index.upsert(
                vectors=[{
                    "id":       vector_id,
                    "values":   vector,
                    "metadata": metadata,
                }]
            )
            logger.debug(
                "Upserted vector %s for session %s", vector_id, session_id
            )
            return vector_id

        except Exception as exc:
            logger.error("Pinecone upsert failed: %s", exc)
            raise RuntimeError(f"Failed to store vector: {exc}") from exc

    # -----------------------------------------------------------------------
    # Query — find similar bugs
    # -----------------------------------------------------------------------

    async def query_similar_bugs(
        self,
        vector: list[float],
        user_id: str,
        project_id: str,
        top_k: int = DEFAULT_TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
        exclude_session_id: Optional[str] = None,
    ) -> list[SimilarBug]:
        """
        Finds the most similar past bugs for a given vector.

        Filters by user_id and project_id so users only see their
        own past bugs — never another user's data.

        Args:
            vector:             The query vector (1536 floats).
            user_id:            Filter to this user's bugs only.
            project_id:         Filter to this project's bugs only.
            top_k:              Maximum number of results to return.
            threshold:          Minimum similarity score (0.0 to 1.0).
            exclude_session_id: Exclude this session (current session).

        Returns:
            List of SimilarBug objects ordered by similarity descending.
            Empty list if no bugs meet the threshold.
        """
        try:
            response = self._index.query(
                vector=vector,
                top_k=top_k + 1,   # fetch one extra to allow exclusion
                include_metadata=True,
                filter={
                    "user_id":    {"$eq": user_id},
                    "project_id": {"$eq": project_id},
                },
            )
        except Exception as exc:
            logger.error("Pinecone query failed: %s", exc)
            return []   # fail gracefully — don't break the main pipeline

        results: list[SimilarBug] = []
        for match in response.matches:
            # Skip if below threshold
            if match.score < threshold:
                continue

            # Skip the current session if provided
            meta = match.metadata or {}
            if (exclude_session_id and
                    meta.get("session_id") == exclude_session_id):
                continue

            results.append(SimilarBug(
                session_id=meta.get("session_id", ""),
                score=match.score,
                error_type=meta.get("error_type", ""),
                error_message=meta.get("error_message", ""),
                language=meta.get("language", ""),
                error_function=meta.get("error_function") or None,
                filename=meta.get("filename") or None,
            ))

        return results[:top_k]

    # -----------------------------------------------------------------------
    # Delete — clean up when a session is deleted
    # -----------------------------------------------------------------------

    async def delete_vector(self, vector_id: str) -> None:
        """
        Deletes a vector from Pinecone by its vector ID.
        Called when the corresponding DebugSession is deleted.

        Args:
            vector_id: The Pinecone vector ID to delete.
        """
        try:
            self._index.delete(ids=[vector_id])
            logger.debug("Deleted vector %s", vector_id)
        except Exception as exc:
            logger.warning("Failed to delete vector %s: %s", vector_id, exc)

    # -----------------------------------------------------------------------
    # Stats — useful for dashboard
    # -----------------------------------------------------------------------

    def get_index_stats(self) -> dict:
        """
        Returns statistics about the Pinecone index.
        Useful for health checks and dashboard display.

        Returns:
            Dict with total_vector_count and other index stats.
        """
        try:
            stats = self._index.describe_index_stats()
            return {
                "total_vector_count": stats.total_vector_count,
                "dimension": stats.dimension,
                "index_name": self._index_name,
            }
        except Exception as exc:
            logger.warning("Could not fetch index stats: %s", exc)
            return {}


__all__ = [
    "VectorService",
    "SimilarBug",
    "SIMILARITY_THRESHOLD",
    "DEFAULT_TOP_K",
]