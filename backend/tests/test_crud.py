"""Test CRUD operations without a real database.

Tests the pure logic functions — password hashing, AI section parsing —
that do not require a database connection.
Database-dependent functions are tested in integration tests later.
"""

from __future__ import annotations

from backend.app.crud.user import hash_password, verify_password
from backend.app.crud.session import _parse_ai_sections


def test_password_hash_and_verify() -> None:
    """Hashed password verifies correctly against original."""
    plain = "mysecurepassword123"
    hashed = hash_password(plain)

    assert hashed != plain
    assert hashed.startswith("$2b$")   # bcrypt prefix
    assert verify_password(plain, hashed) is True
    print("Password hash + verify: OK")


def test_wrong_password_rejected() -> None:
    """Wrong password does not verify against hash."""
    hashed = hash_password("correctpassword")
    assert verify_password("wrongpassword", hashed) is False
    print("Wrong password reject:  OK")


def test_parse_ai_sections_all_present() -> None:
    """All four sections are parsed correctly from AI response."""
    ai_response = """
## Root Cause
The list was indexed with a string key.

## Explanation
Python lists only accept integer indices.
Using a string causes an IndexError.

## Fixed Code
```python
def get_user(users: dict, user_id: str):
    return users[user_id]['name']
```

## Prevention
Always validate that the data structure matches the expected type.
"""
    sections = _parse_ai_sections(ai_response)

    assert "list was indexed" in sections["root_cause"]
    assert "integer indices" in sections["explanation"]
    assert "get_user" in sections["fixed_code"]
    assert "validate" in sections["prevention"]
    print("AI section parsing:     OK")


def test_parse_ai_sections_missing_section() -> None:
    """Missing section returns empty string, not an error."""
    ai_response = """
## Root Cause
Something went wrong.

## Explanation
Here is why.
"""
    sections = _parse_ai_sections(ai_response)

    assert sections["root_cause"] != ""
    assert sections["fixed_code"] == ""
    assert sections["prevention"] == ""
    print("Missing sections:       OK")


def test_hash_is_unique() -> None:
    """Same password hashed twice produces different hashes (bcrypt salt)."""
    password = "samepassword123"
    hash1 = hash_password(password)
    hash2 = hash_password(password)

    assert hash1 != hash2
    assert verify_password(password, hash1) is True
    assert verify_password(password, hash2) is True
    print("Hash uniqueness:        OK")


if __name__ == "__main__":
    test_password_hash_and_verify()
    test_wrong_password_rejected()
    test_parse_ai_sections_all_present()
    test_parse_ai_sections_missing_section()
    test_hash_is_unique()
    print("\nAll CRUD tests passed.")