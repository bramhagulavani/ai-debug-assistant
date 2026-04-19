"""CRUD operations for the Project model."""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.models.database import Project
from backend.app.models.schemas import ProjectCreate, ProjectUpdate


async def create_project(
    db: AsyncSession,
    user_id: uuid.UUID,
    data: ProjectCreate,
) -> Project:
    """
    Creates a new project owned by the specified user.

    Args:
        db:      Active database session.
        user_id: UUID of the user who owns this project.
        data:    Validated ProjectCreate schema.

    Returns:
        The newly created Project instance.
    """
    project = Project(
        user_id=user_id,
        name=data.name,
        description=data.description,
        language=data.language,
    )
    db.add(project)
    await db.flush()
    await db.refresh(project)
    return project


async def get_project_by_id(
    db: AsyncSession,
    project_id: uuid.UUID,
) -> Optional[Project]:
    """
    Fetches a project by its UUID.

    Args:
        db:         Active database session.
        project_id: The project's UUID.

    Returns:
        Project instance if found, None otherwise.
    """
    result = await db.execute(
        select(Project).where(Project.id == project_id)
    )
    return result.scalar_one_or_none()


async def get_projects_by_user(
    db: AsyncSession,
    user_id: uuid.UUID,
    skip: int = 0,
    limit: int = 20,
) -> list[Project]:
    """
    Fetches all projects belonging to a user with pagination.

    Args:
        db:      Active database session.
        user_id: The user's UUID.
        skip:    Number of records to skip (for pagination).
        limit:   Maximum number of records to return.

    Returns:
        List of Project instances ordered by creation date descending.
    """
    result = await db.execute(
        select(Project)
        .where(Project.user_id == user_id)
        .order_by(Project.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    return list(result.scalars().all())


async def update_project(
    db: AsyncSession,
    project: Project,
    data: ProjectUpdate,
) -> Project:
    """
    Updates allowed fields on an existing project.
    Only updates fields that are explicitly provided (not None).

    Args:
        db:      Active database session.
        project: The existing Project instance to update.
        data:    Validated ProjectUpdate schema.

    Returns:
        The updated Project instance.
    """
    if data.name is not None:
        project.name = data.name
    if data.description is not None:
        project.description = data.description
    if data.language is not None:
        project.language = data.language

    await db.flush()
    await db.refresh(project)
    return project


async def delete_project(
    db: AsyncSession,
    project: Project,
) -> None:
    """
    Deletes a project and all its sessions via cascade.

    Args:
        db:      Active database session.
        project: The Project instance to delete.
    """
    await db.delete(project)
    await db.flush()