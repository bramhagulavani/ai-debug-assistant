"""SQLAlchemy database models for AI Debugging Assistant.

Four tables:
    User         - registered developers using the tool
    Project      - a codebase/repo a user is debugging
    DebugSession - one debugging event (one error analysed)
    BugReport    - structured output from the AI for a session

Relationships:
    User -> has many -> Projects
    Project -> has many -> DebugSessions
    DebugSession -> has one -> BugReport
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared base for all SQLAlchemy models."""


class TimestampMixin:
    """Reusable created/updated timestamp columns for models."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class User(TimestampMixin, Base):
    """Represents a registered developer account."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
    )
    username: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        index=True,
    )
    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
    )

    projects: Mapped[list[Project]] = relationship(
        "Project",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return a concise debugging representation of the user."""

        return f"<User id={self.id} email={self.email}>"


class Project(TimestampMixin, Base):
    """Represents a repository/codebase under debugging."""

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    language: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    user: Mapped[User] = relationship(
        "User",
        back_populates="projects",
    )
    sessions: Mapped[list[DebugSession]] = relationship(
        "DebugSession",
        back_populates="project",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """Return a concise debugging representation of the project."""

        return f"<Project id={self.id} name={self.name}>"


class DebugSession(TimestampMixin, Base):
    """Represents one analyzed debugging event in a project."""

    __tablename__ = "debug_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    project_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    raw_code: Mapped[str] = mapped_column(Text, nullable=False)
    raw_stack_trace: Mapped[str] = mapped_column(Text, nullable=False)
    filename: Mapped[str | None] = mapped_column(String(255), nullable=True)
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)

    error_type: Mapped[str | None] = mapped_column(String(100), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_function: Mapped[str | None] = mapped_column(String(200), nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vector_id: Mapped[str | None] = mapped_column(String(100), nullable=True)

    project: Mapped[Project] = relationship(
        "Project",
        back_populates="sessions",
    )
    bug_report: Mapped[BugReport | None] = relationship(
        "BugReport",
        back_populates="session",
        cascade="all, delete-orphan",
        uselist=False,
    )

    def __repr__(self) -> str:
        """Return a concise debugging representation of the debug session."""

        return f"<DebugSession id={self.id} error_type={self.error_type}>"


class BugReport(TimestampMixin, Base):
    """Stores structured AI output for a single debug session."""

    __tablename__ = "bug_reports"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("debug_sessions.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    root_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    fixed_code: Mapped[str | None] = mapped_column(Text, nullable=True)
    prevention: Mapped[str | None] = mapped_column(Text, nullable=True)

    full_ai_response: Mapped[str] = mapped_column(Text, nullable=False)

    session: Mapped[DebugSession] = relationship(
        "DebugSession",
        back_populates="bug_report",
    )

    def __repr__(self) -> str:
        """Return a concise debugging representation of the bug report."""

        return f"<BugReport id={self.id} session_id={self.session_id}>"


__all__ = [
    "Base",
    "User",
    "Project",
    "DebugSession",
    "BugReport",
]
