"""Test that SQLAlchemy models are defined correctly.

This test does not require a real database. It verifies model metadata,
column definitions, and relationship registration.
"""

from __future__ import annotations

from backend.app.models.database import Base, BugReport, DebugSession, Project, User


def test_table_names() -> None:
    """Verify all four tables have the correct names."""

    assert User.__tablename__ == "users"
    assert Project.__tablename__ == "projects"
    assert DebugSession.__tablename__ == "debug_sessions"
    assert BugReport.__tablename__ == "bug_reports"
    print("Table names:  OK")


def test_user_columns() -> None:
    """Verify User model includes all required columns."""

    cols = [c.name for c in User.__table__.columns]
    for expected in [
        "id",
        "email",
        "username",
        "hashed_password",
        "is_active",
        "created_at",
        "updated_at",
    ]:
        assert expected in cols, f"Missing column: {expected}"
    print("User columns: OK")


def test_session_columns() -> None:
    """Verify DebugSession includes key debug and trace columns."""

    cols = [c.name for c in DebugSession.__table__.columns]
    for expected in [
        "id",
        "project_id",
        "raw_code",
        "raw_stack_trace",
        "error_type",
        "error_line",
        "vector_id",
    ]:
        assert expected in cols, f"Missing column: {expected}"
    print("DebugSession columns: OK")


def test_relationships() -> None:
    """Verify declared relationships are present on all models."""

    assert hasattr(User, "projects")
    assert hasattr(Project, "sessions")
    assert hasattr(DebugSession, "bug_report")
    assert hasattr(BugReport, "session")
    print("Relationships: OK")


if __name__ == "__main__":
    test_table_names()
    test_user_columns()
    test_session_columns()
    test_relationships()
    print("\nAll model tests passed.")
