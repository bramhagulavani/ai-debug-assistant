"""Test that all Pydantic schemas validate correctly."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from backend.app.models.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    BugReportResponse,
    DebugSessionResponse,
    ProjectCreate,
    ProjectUpdate,
    Token,
    UserCreate,
    UserLogin,
    UserResponse,
)


def test_user_create_valid() -> None:
    """Valid user creation data passes validation."""
    user = UserCreate(
        email="bramha@example.com",
        username="bramha_dev",
        password="securepass123",
    )
    assert user.email == "bramha@example.com"
    assert user.username == "bramha_dev"
    print("UserCreate valid:        OK")


def test_user_create_invalid_username() -> None:
    """Username with special characters should fail."""
    try:
        UserCreate(
            email="test@example.com",
            username="bramha dev",   # space not allowed
            password="securepass123",
        )
        print("UserCreate invalid:      FAILED — should have raised")
    except Exception:
        print("UserCreate invalid:      OK — correctly rejected")


def test_user_create_short_password() -> None:
    """Password shorter than 8 characters should fail."""
    try:
        UserCreate(
            email="test@example.com",
            username="bramha",
            password="short",
        )
        print("Short password:         FAILED — should have raised")
    except Exception:
        print("Short password:         OK — correctly rejected")


def test_project_create() -> None:
    """Valid project creation passes."""
    project = ProjectCreate(
        name="My FastAPI App",
        description="Backend for a web app",
        language="python",
    )
    assert project.name == "My FastAPI App"
    print("ProjectCreate:          OK")


def test_project_update_all_optional() -> None:
    """ProjectUpdate with no fields should be valid — all fields optional."""
    update = ProjectUpdate()
    assert update.name is None
    assert update.language is None
    print("ProjectUpdate empty:    OK")


def test_analyze_request_valid() -> None:
    """Valid analyze request passes."""
    req = AnalyzeRequest(
        code="def foo(): pass",
        stack_trace="IndexError: list index out of range",
        filename="app.py",
    )
    assert req.code == "def foo(): pass"
    assert req.project_id is None
    print("AnalyzeRequest valid:   OK")


def test_analyze_request_empty_code() -> None:
    """Empty code string should fail validation."""
    try:
        AnalyzeRequest(code="", stack_trace="some error")
        print("Empty code:             FAILED — should have raised")
    except Exception:
        print("Empty code:             OK — correctly rejected")


def test_analyze_response() -> None:
    """AnalyzeResponse builds correctly from dict."""
    resp = AnalyzeResponse(
        success=True,
        language="python",
        error_type="IndexError",
        error_message="list index out of range",
        error_line=5,
        error_function="get_user",
        ai_response="## Root Cause\nTest",
        duration_ms=1200,
    )
    assert resp.session_id is None
    print("AnalyzeResponse:        OK")


def test_token_schema() -> None:
    """Token schema defaults token_type to bearer."""
    token = Token(access_token="sometoken123")
    assert token.token_type == "bearer"
    print("Token schema:           OK")


def test_user_response_from_dict() -> None:
    """UserResponse builds from a plain dict."""
    data = {
        "id": uuid.uuid4(),
        "email": "test@example.com",
        "username": "testuser",
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    user = UserResponse(**data)
    assert user.username == "testuser"
    print("UserResponse from dict: OK")


if __name__ == "__main__":
    test_user_create_valid()
    test_user_create_invalid_username()
    test_user_create_short_password()
    test_project_create()
    test_project_update_all_optional()
    test_analyze_request_valid()
    test_analyze_request_empty_code()
    test_analyze_response()
    test_token_schema()
    test_user_response_from_dict()
    print("\nAll schema tests passed.")