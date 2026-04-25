"""CRUD operations for the User model.

All functions are async and accept a SQLAlchemy AsyncSession.
Passwords are never stored in plain text — always hashed before saving.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import User
from backend.app.models.schemas import UserCreate


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def hash_password(plain_password: str) -> str:
    """
    Hashes a plain text password using bcrypt.
    Always use this before storing a password in the database.

    Args:
        plain_password: The raw password string from the user.

    Returns:
        A bcrypt hash string safe to store in the database.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain text password against a stored bcrypt hash.

    Args:
        plain_password:  The raw password to check.
        hashed_password: The stored bcrypt hash from the database.

    Returns:
        True if the password matches, False otherwise.
    """
    from passlib.context import CryptContext
    pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return pwd_context.verify(plain_password, hashed_password)


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

async def create_user(db: AsyncSession, data: UserCreate) -> User:
    """
    Creates a new user in the database.
    Hashes the password before storing.

    Args:
        db:   Active database session.
        data: Validated UserCreate schema.

    Returns:
        The newly created User model instance.
    """
    user = User(
        email=data.email,
        username=data.username,
        hashed_password=hash_password(data.password),
    )
    db.add(user)
    await db.flush()   # assigns the UUID without committing
    await db.refresh(user)
    return user


async def get_user_by_id(db: AsyncSession, user_id: uuid.UUID) -> Optional[User]:
    """
    Fetches a user by their UUID.

    Args:
        db:      Active database session.
        user_id: The user's UUID.

    Returns:
        User instance if found, None otherwise.
    """
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """
    Fetches a user by their email address.
    Used during login to look up the account.

    Args:
        db:    Active database session.
        email: Email address to search for.

    Returns:
        User instance if found, None otherwise.
    """
    result = await db.execute(
        select(User).where(User.email == email)
    )
    return result.scalar_one_or_none()


async def get_user_by_username(
    db: AsyncSession, username: str
) -> Optional[User]:
    """
    Fetches a user by their username.

    Args:
        db:       Active database session.
        username: Username to search for.

    Returns:
        User instance if found, None otherwise.
    """
    result = await db.execute(
        select(User).where(User.username == username)
    )
    return result.scalar_one_or_none()


async def email_exists(db: AsyncSession, email: str) -> bool:
    """
    Checks whether an email address is already registered.
    Used during registration to prevent duplicate accounts.

    Args:
        db:    Active database session.
        email: Email address to check.

    Returns:
        True if the email is already in use, False otherwise.
    """
    result = await db.execute(
        select(User.id).where(User.email == email)
    )
    return result.scalar_one_or_none() is not None


async def username_exists(db: AsyncSession, username: str) -> bool:
    """
    Checks whether a username is already taken.

    Args:
        db:       Active database session.
        username: Username to check.

    Returns:
        True if the username is already taken, False otherwise.
    """
    result = await db.execute(
        select(User.id).where(User.username == username)
    )
    return result.scalar_one_or_none() is not None


async def authenticate_user(
    db: AsyncSession, email: str, password: str
) -> Optional[User]:
    """
    Validates email + password and returns the user if correct.
    Returns None if the email doesn't exist or password is wrong.

    Args:
        db:       Active database session.
        email:    Email address from login form.
        password: Plain text password from login form.

    Returns:
        Authenticated User instance, or None if credentials are invalid.
    """
    user = await get_user_by_email(db, email)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user