"""Authentication API endpoints for user registration and JWT login."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.database import get_db
from backend.app.core.security import create_access_token, get_current_user
from backend.app.crud.user import (
    authenticate_user,
    create_user,
    email_exists,
    username_exists,
)
from backend.app.models.database import User
from backend.app.models.schemas import Token, UserCreate, UserLogin, UserResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> UserResponse:
    """Register a new user account.

    Args:
        payload: Validated user registration payload.
        db: Active database session.

    Returns:
        Sanitized user profile without password hash.

    Raises:
        HTTPException: If email or username is already taken.
    """
    if await email_exists(db, payload.email):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email is already registered",
        )

    if await username_exists(db, payload.username):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username is already taken",
        )

    user = await create_user(db, payload)
    return UserResponse.model_validate(user)


@router.post("/login", response_model=Token)
async def login_user(
    payload: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Authenticate a user and return a JWT access token.

    Args:
        payload: Login credentials (email + password).
        db: Active database session.

    Returns:
        Bearer token payload for client-side authorization headers.

    Raises:
        HTTPException: If credentials are invalid.
    """
    user = await authenticate_user(db, payload.email, payload.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(user_id=user.id, email=user.email)
    return Token(access_token=token, token_type="bearer")


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_user)) -> UserResponse:
    """Return the currently authenticated user profile.

    Args:
        current_user: Resolved authenticated user via bearer token.

    Returns:
        Authenticated user profile.
    """
    return UserResponse.model_validate(current_user)


__all__ = ["router", "register_user", "login_user", "get_me"]
