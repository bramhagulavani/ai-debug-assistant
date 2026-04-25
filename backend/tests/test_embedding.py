"""Test the embedding service.

Two types of tests:
    1. Unit tests — test build_embedding_text and cosine_similarity
       without any API calls (no OpenAI key needed)
    2. Integration test — calls OpenAI API to get real vectors
       (requires valid OPENAI_API_KEY in .env)
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv
load_dotenv()

from backend.app.services.embedding_service import (
    EmbeddingService,
    build_embedding_text,
)


# ---------------------------------------------------------------------------
# Unit tests — no API calls needed
# ---------------------------------------------------------------------------

def test_build_embedding_text_full() -> None:
    """All fields produce a well-formed embedding string."""
    text = build_embedding_text(
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
        error_function="get_user",
        filename="main.py",
        code_window="   5 |     return users[user_id]['name']",
    )
    assert "IndexError" in text
    assert "list index out of range" in text
    assert "python" in text
    assert "get_user" in text
    assert "main.py" in text
    print("build_embedding_text full:    OK")


def test_build_embedding_text_minimal() -> None:
    """Minimal fields (no optional args) still produces valid string."""
    text = build_embedding_text(
        error_type="TypeError",
        error_message="cannot unpack non-sequence int",
        language="python",
    )
    assert "TypeError" in text
    assert "python" in text
    assert "Function:" not in text
    print("build_embedding_text minimal: OK")


def test_cosine_similarity_identical() -> None:
    """Identical vectors have similarity of 1.0."""
    vec = [1.0, 0.0, 0.0, 1.0]
    result = EmbeddingService.cosine_similarity(vec, vec)
    assert abs(result - 1.0) < 0.0001
    print("Cosine similarity identical:  OK")


def test_cosine_similarity_orthogonal() -> None:
    """Orthogonal vectors have similarity of 0.0."""
    vec_a = [1.0, 0.0]
    vec_b = [0.0, 1.0]
    result = EmbeddingService.cosine_similarity(vec_a, vec_b)
    assert abs(result - 0.0) < 0.0001
    print("Cosine similarity orthogonal: OK")


def test_cosine_similarity_similar() -> None:
    """Similar vectors have high similarity score."""
    vec_a = [1.0, 0.9, 0.8]
    vec_b = [0.9, 1.0, 0.7]
    result = EmbeddingService.cosine_similarity(vec_a, vec_b)
    assert result > 0.95
    print("Cosine similarity similar:    OK")


# ---------------------------------------------------------------------------
# Integration test — calls real OpenAI API
# ---------------------------------------------------------------------------

async def test_embed_error_real() -> None:
    """
    Calls the real embedding API and verifies:
        - Vector has correct dimensions (1536)
        - Similar errors produce higher similarity than unrelated errors
    """
    svc = EmbeddingService()

    print("\nCalling OpenAI embedding API...")

    # Embed two similar IndexError bugs
    vec_index_1 = await svc.embed_error(
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
        error_function="get_user",
    )

    vec_index_2 = await svc.embed_error(
        error_type="IndexError",
        error_message="list assignment index out of range",
        language="python",
        error_function="update_user",
    )

    # Embed an unrelated TypeError bug
    vec_type_error = await svc.embed_error(
        error_type="TypeError",
        error_message="unsupported operand type for +: int and str",
        language="python",
        error_function="calculate_total",
    )

    # Check dimensions
    assert len(vec_index_1) == 1536, f"Expected 1536, got {len(vec_index_1)}"
    print(f"Vector dimensions:            OK ({len(vec_index_1)})")

    # Similar errors should be more similar to each other
    sim_similar   = EmbeddingService.cosine_similarity(vec_index_1, vec_index_2)
    sim_different = EmbeddingService.cosine_similarity(vec_index_1, vec_type_error)

    print(f"Similar errors similarity:    {sim_similar:.4f}")
    print(f"Different errors similarity:  {sim_different:.4f}")

    assert sim_similar > sim_different, (
        f"Expected similar errors ({sim_similar:.4f}) to score higher "
        f"than different errors ({sim_different:.4f})"
    )
    print("Similarity ranking:           OK")


if __name__ == "__main__":
    # Unit tests first — no API needed
    test_build_embedding_text_full()
    test_build_embedding_text_minimal()
    test_cosine_similarity_identical()
    test_cosine_similarity_orthogonal()
    test_cosine_similarity_similar()
    print("\nUnit tests passed.")

    # Integration test — needs real API key
    print("\nRunning integration test (needs OPENAI_API_KEY)...")
    asyncio.run(test_embed_error_real())
    print("\nAll embedding tests passed.")  