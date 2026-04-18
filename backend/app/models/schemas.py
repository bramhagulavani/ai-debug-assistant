"""Pydantic schemas for AI Debugging Assistant.

These schemas serve three purposes:
    1. Validate incoming HTTP request bodies
    2. Shape outgoing HTTP response bodies
    3. Provide a clean boundary between API and database models

Naming convention:
    UserCreate     — data needed to CREATE a resource
    UserUpdate     — data allowed to UPDATE a resource
    UserResponse   — data returned TO the client
    UserInDB       — data as stored IN the database (includes sensitive fields)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator


# ---------------------------------------------------------------------------
# Shared base config
# ---------------------------------------------------------------------------

class AppBaseModel(BaseModel):
    """
    Base model for all schemas.
    from_attributes=True allows building from SQLAlchemy model instances.
    """

    model_config = {"from_attributes": True}


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------

class UserCreate(AppBaseModel):
    """Data required to register a new user."""

    email: EmailStr = Field(
        ...,
        description="Valid email address — used for login.",
    )
    username: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Unique username, 3–100 characters.",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=100,
        description="Password, minimum 8 characters.",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, value: str) -> str:
        """Ensures username contains only letters, numbers, and underscores."""
        if not all(c.isalnum() or c == "_" for c in value):
            raise ValueError(
                "Username may only contain letters, numbers, and underscores."
            )
        return value.lower()


class UserLogin(AppBaseModel):
    """Data required to log in."""

    email: EmailStr
    password: str


class UserResponse(AppBaseModel):
    """User data returned to the client — never includes the password."""

    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    created_at: datetime


class UserInDB(UserResponse):
    """
    Internal representation including the hashed password.
    Never serialise this to an API response directly.
    """

    hashed_password: str


# ---------------------------------------------------------------------------
# Token schemas (JWT auth)
# ---------------------------------------------------------------------------

class Token(AppBaseModel):
    """JWT access token returned after successful login."""

    access_token: str
    token_type: str = "bearer"


class TokenData(AppBaseModel):
    """Data decoded from a JWT token."""

    user_id: Optional[uuid.UUID] = None
    email: Optional[str] = None


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------

class ProjectCreate(AppBaseModel):
    """Data required to create a new project."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Project name.",
    )
    description: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Optional project description.",
    )
    language: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Primary language: 'python', 'javascript', etc.",
    )


class ProjectUpdate(AppBaseModel):
    """Fields allowed when updating a project — all optional."""

    name: Optional[str] = Field(default=None, min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, max_length=1000)
    language: Optional[str] = Field(default=None, max_length=50)


class ProjectResponse(AppBaseModel):
    """Project data returned to the client."""

    id: uuid.UUID
    user_id: uuid.UUID
    name: str
    description: Optional[str]
    language: Optional[str]
    created_at: datetime
    updated_at: datetime


# ---------------------------------------------------------------------------
# DebugSession schemas
# ---------------------------------------------------------------------------

class DebugSessionCreate(AppBaseModel):
    """Data required to create a new debug session."""

    project_id: uuid.UUID
    raw_code: str = Field(..., min_length=1)
    raw_stack_trace: str = Field(..., min_length=1)
    filename: Optional[str] = Field(default=None, max_length=255)
    language: Optional[str] = Field(default=None, max_length=50)


class DebugSessionResponse(AppBaseModel):
    """Debug session data returned to the client."""

    id: uuid.UUID
    project_id: uuid.UUID
    filename: Optional[str]
    language: Optional[str]
    error_type: Optional[str]
    error_message: Optional[str]
    error_line: Optional[int]
    error_function: Optional[str]
    duration_ms: Optional[int]
    created_at: datetime


class DebugSessionDetailResponse(DebugSessionResponse):
    """
    Extended session response that includes the full AI bug report.
    Used when fetching a single session by ID.
    """

    bug_report: Optional[BugReportResponse] = None


# ---------------------------------------------------------------------------
# BugReport schemas
# ---------------------------------------------------------------------------

class BugReportResponse(AppBaseModel):
    """AI-generated bug report returned to the client."""

    id: uuid.UUID
    session_id: uuid.UUID
    root_cause: Optional[str]
    explanation: Optional[str]
    fixed_code: Optional[str]
    prevention: Optional[str]
    full_ai_response: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Analysis schemas (used by /api/analyze and /ws/analyze)
# ---------------------------------------------------------------------------

class AnalyzeRequest(AppBaseModel):
    """
    Request body for the analyze endpoint.
    Replaces the inline Pydantic model in analyze.py.
    """

    code: str = Field(..., min_length=1)
    stack_trace: str = Field(..., min_length=1)
    filename: Optional[str] = None
    language: Optional[str] = None
    project_id: Optional[uuid.UUID] = None   # optional — saves to DB if provided


class AnalyzeResponse(AppBaseModel):
    """Structured response from the analyze endpoint."""

    success: bool
    language: str
    error_type: str
    error_message: str
    error_line: Optional[int]
    error_function: Optional[str]
    ai_response: str
    duration_ms: int
    session_id: Optional[uuid.UUID] = None   # set if saved to DB


# ---------------------------------------------------------------------------
# Pagination schema (reused across list endpoints)
# ---------------------------------------------------------------------------

class PaginatedResponse(AppBaseModel):
    """Generic wrapper for paginated list responses."""

    total: int
    page: int
    page_size: int
    items: list


# ---------------------------------------------------------------------------
# Forward reference resolution
# ---------------------------------------------------------------------------

DebugSessionDetailResponse.model_rebuild()


__all__ = [
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "UserInDB",
    "Token",
    "TokenData",
    "ProjectCreate",
    "ProjectUpdate",
    "ProjectResponse",
    "DebugSessionCreate",
    "DebugSessionResponse",
    "DebugSessionDetailResponse",
    "BugReportResponse",
    "AnalyzeRequest",
    "AnalyzeResponse",
    "PaginatedResponse",
]