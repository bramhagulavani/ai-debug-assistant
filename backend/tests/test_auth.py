"""Tests for JWT authentication endpoints and dependencies."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.auth import router as auth_router
from backend.app.core.database import get_db
import backend.app.api.auth as auth_api
import backend.app.core.security as security


@dataclass
class _FakeUser:
    """Simple user object used to mimic SQLAlchemy User instances in tests."""

    id: uuid.UUID
    email: str
    username: str
    is_active: bool
    created_at: datetime
    hashed_password: str


class _FakeDB:
    """Placeholder DB session object for dependency overrides."""


async def _fake_get_db() -> AsyncGenerator[_FakeDB, None]:
    """Yield a fake DB session for endpoint tests."""

    yield _FakeDB()


def _build_test_app() -> FastAPI:
    """Create a lightweight FastAPI app that only mounts auth routes."""

    app = FastAPI(title="Auth Test App")
    app.include_router(auth_router)
    app.dependency_overrides[get_db] = _fake_get_db
    return app


def test_register_success() -> None:
    """POST /api/auth/register should create and return a new user profile."""

    app = _build_test_app()
    user_id = uuid.uuid4()

    async def _email_exists(_db: _FakeDB, _email: str) -> bool:
        return False

    async def _username_exists(_db: _FakeDB, _username: str) -> bool:
        return False

    async def _create_user(_db: _FakeDB, data):  # type: ignore[no-untyped-def]
        return _FakeUser(
            id=user_id,
            email=data.email,
            username=data.username,
            is_active=True,
            created_at=datetime.now(timezone.utc),
            hashed_password="hashed",
        )

    auth_api.email_exists = _email_exists
    auth_api.username_exists = _username_exists
    auth_api.create_user = _create_user

    client = TestClient(app)
    response = client.post(
        "/api/auth/register",
        json={
            "email": "student@example.com",
            "username": "student_user",
            "password": "strongpassword123",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(user_id)
    assert body["email"] == "student@example.com"
    assert body["username"] == "student_user"
    assert body["is_active"] is True


def test_register_duplicate_email() -> None:
    """POST /api/auth/register should reject duplicate emails with 409."""

    app = _build_test_app()

    async def _email_exists(_db: _FakeDB, _email: str) -> bool:
        return True

    async def _username_exists(_db: _FakeDB, _username: str) -> bool:
        return False

    auth_api.email_exists = _email_exists
    auth_api.username_exists = _username_exists

    client = TestClient(app)
    response = client.post(
        "/api/auth/register",
        json={
            "email": "existing@example.com",
            "username": "new_user",
            "password": "strongpassword123",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Email is already registered"


def test_login_and_me_flow() -> None:
    """POST /login returns JWT and GET /me resolves the authenticated user."""

    app = _build_test_app()
    user_id = uuid.uuid4()
    fake_user = _FakeUser(
        id=user_id,
        email="student@example.com",
        username="student_user",
        is_active=True,
        created_at=datetime.now(timezone.utc),
        hashed_password="hashed",
    )

    async def _authenticate_user(_db: _FakeDB, email: str, password: str) -> _FakeUser | None:
        if email == "student@example.com" and password == "strongpassword123":
            return fake_user
        return None

    async def _get_user_by_id(_db: _FakeDB, requested_user_id: uuid.UUID) -> _FakeUser | None:
        if requested_user_id == fake_user.id:
            return fake_user
        return None

    auth_api.authenticate_user = _authenticate_user
    security.get_user_by_id = _get_user_by_id

    client = TestClient(app)

    login_response = client.post(
        "/api/auth/login",
        json={
            "email": "student@example.com",
            "password": "strongpassword123",
        },
    )
    assert login_response.status_code == 200

    token_payload = login_response.json()
    assert token_payload["token_type"] == "bearer"
    assert token_payload["access_token"]

    me_response = client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {token_payload['access_token']}"},
    )
    assert me_response.status_code == 200
    me_body = me_response.json()
    assert me_body["id"] == str(fake_user.id)
    assert me_body["email"] == fake_user.email


def test_login_invalid_credentials() -> None:
    """POST /api/auth/login should reject invalid credentials with 401."""

    app = _build_test_app()

    async def _authenticate_user(_db: _FakeDB, _email: str, _password: str) -> None:
        return None

    auth_api.authenticate_user = _authenticate_user

    client = TestClient(app)
    response = client.post(
        "/api/auth/login",
        json={
            "email": "wrong@example.com",
            "password": "wrongpass",
        },
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid email or password"


def test_me_invalid_token() -> None:
    """GET /api/auth/me should reject malformed/invalid bearer tokens."""

    app = _build_test_app()
    client = TestClient(app)

    response = client.get(
        "/api/auth/me",
        headers={"Authorization": "Bearer invalid.token.value"},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"
