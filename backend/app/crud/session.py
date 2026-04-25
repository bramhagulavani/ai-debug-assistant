"""CRUD operations for DebugSession and BugReport models."""

from __future__ import annotations

import re
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.app.models.database import BugReport, DebugSession


async def create_debug_session(
    db: AsyncSession,
    project_id: uuid.UUID,
    raw_code: str,
    raw_stack_trace: str,
    filename: Optional[str],
    language: Optional[str],
    error_type: Optional[str],
    error_message: Optional[str],
    error_line: Optional[int],
    error_function: Optional[str],
    duration_ms: Optional[int],
) -> DebugSession:
    """
    Saves one debug event to the database.
    Called after the AI analysis completes successfully.

    Returns:
        The newly created DebugSession instance.
    """
    session = DebugSession(
        project_id=project_id,
        raw_code=raw_code,
        raw_stack_trace=raw_stack_trace,
        filename=filename,
        language=language,
        error_type=error_type,
        error_message=error_message,
        error_line=error_line,
        error_function=error_function,
        duration_ms=duration_ms,
    )
    db.add(session)
    await db.flush()
    await db.refresh(session)
    return session


async def create_bug_report(
    db: AsyncSession,
    session_id: uuid.UUID,
    full_ai_response: str,
) -> BugReport:
    """
    Saves the AI-generated bug report for a debug session.
    Automatically parses the four sections from the AI response.

    Args:
        db:               Active database session.
        session_id:       UUID of the parent DebugSession.
        full_ai_response: The complete AI response string.

    Returns:
        The newly created BugReport instance.
    """
    sections = _parse_ai_sections(full_ai_response)

    report = BugReport(
        session_id=session_id,
        full_ai_response=full_ai_response,
        root_cause=sections.get("root_cause"),
        explanation=sections.get("explanation"),
        fixed_code=sections.get("fixed_code"),
        prevention=sections.get("prevention"),
    )
    db.add(report)
    await db.flush()
    await db.refresh(report)
    return report


def _parse_ai_sections(ai_response: str) -> dict[str, str]:
    """
    Parses the four sections from an AI response string.

    The AI is instructed to always respond with:
        ## Root Cause
        ## Explanation
        ## Fixed Code
        ## Prevention

    Args:
        ai_response: The full AI response text.

    Returns:
        Dict with keys: root_cause, explanation, fixed_code, prevention.
        Any missing section defaults to empty string.
    """
    section_map = {
        "root_cause":  r"##\s*Root Cause\s*(.*?)(?=##|\Z)",
        "explanation": r"##\s*Explanation\s*(.*?)(?=##|\Z)",
        "fixed_code":  r"##\s*Fixed Code\s*(.*?)(?=##|\Z)",
        "prevention":  r"##\s*Prevention\s*(.*?)(?=##|\Z)",
    }

    result: dict[str, str] = {}
    for key, pattern in section_map.items():
        match = re.search(pattern, ai_response, re.DOTALL | re.IGNORECASE)
        result[key] = match.group(1).strip() if match else ""

    return result


async def get_session_by_id(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> Optional[DebugSession]:
    """
    Fetches a single debug session with its bug report eagerly loaded.

    Args:
        db:         Active database session.
        session_id: The session's UUID.

    Returns:
        DebugSession with bug_report populated, or None if not found.
    """
    result = await db.execute(
        select(DebugSession)
        .where(DebugSession.id == session_id)
        .options(selectinload(DebugSession.bug_report))
    )
    return result.scalar_one_or_none()


async def get_sessions_by_project(
    db: AsyncSession,
    project_id: uuid.UUID,
    skip: int = 0,
    limit: int = 20,
) -> list[DebugSession]:
    """
    Fetches all debug sessions for a project with pagination.
    Ordered by most recent first.

    Args:
        db:         Active database session.
        project_id: The project's UUID.
        skip:       Records to skip.
        limit:      Maximum records to return.

    Returns:
        List of DebugSession instances.
    """
    result = await db.execute(
        select(DebugSession)
        .where(DebugSession.project_id == project_id)
        .order_by(DebugSession.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def get_recent_sessions_by_project(
    db: AsyncSession,
    project_id: uuid.UUID,
    limit: int = 5,
) -> list[DebugSession]:
    """
    Fetches the most recent N sessions for a project.
    Used by the similar bug finder to seed the RAG context.

    Args:
        db:         Active database session.
        project_id: The project's UUID.
        limit:      How many recent sessions to return.

    Returns:
        List of DebugSession instances, most recent first.
    """
    result = await db.execute(
        select(DebugSession)
        .where(DebugSession.project_id == project_id)
        .order_by(DebugSession.created_at.desc())
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_session_vector_id(
    db: AsyncSession,
    session: DebugSession,
    vector_id: str,
) -> DebugSession:
    """
    Sets the Pinecone vector ID on a debug session after embedding.
    Called by the embedding service once the vector is stored.

    Args:
        db:        Active database session.
        session:   The DebugSession to update.
        vector_id: The Pinecone vector ID string.

    Returns:
        Updated DebugSession instance.
    """
    session.vector_id = vector_id
    await db.flush()
    await db.refresh(session)
    return session