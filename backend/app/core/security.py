"""JWT security utilities and authentication dependencies.

This module centralizes token generation, token validation, and the
`get_current_user` dependency used by protected API routes.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.database import get_db
from backend.app.crud.user import get_user_by_id
from backend.app.models.database import User
from backend.app.models.schemas import TokenData

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


def create_access_token(
    user_id: uuid.UUID,
    email: str,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token for an authenticated user.

    Args:
        user_id: Authenticated user UUID.
        email: User email included as token context.
        expires_delta: Optional custom token lifetime.

    Returns:
        Encoded JWT string using configured secret and algorithm.
    """
    expire = datetime.now(timezone.utc) + (
        expires_delta
        if expires_delta is not None
        else timedelta(minutes=settings.access_token_expire_minutes)
    )

    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> TokenData:
    """Decode a JWT token and extract token data.

    Args:
        token: Raw bearer token string.

    Returns:
        Parsed `TokenData` containing `user_id` and `email` when valid.

    Raises:
        HTTPException: If token is malformed, expired, or missing required claims.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
        user_id_raw = payload.get("sub")
        email = payload.get("email")
        if user_id_raw is None:
            raise credentials_exception
        user_id = uuid.UUID(str(user_id_raw))
    except (JWTError, ValueError) as exc:
        raise credentials_exception from exc

    return TokenData(user_id=user_id, email=email)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Return the authenticated user from the bearer token.

    Args:
        token: Bearer token extracted from the Authorization header.
        db: Active database session.

    Returns:
        Active `User` model instance.

    Raises:
        HTTPException: If token is invalid, user does not exist, or user is inactive.
    """
    token_data = decode_access_token(token)
    if token_data.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user = await get_user_by_id(db, token_data.user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )

    return user


__all__ = [
    "oauth2_scheme",
    "create_access_token",
    "decode_access_token",
    "get_current_user",
]
