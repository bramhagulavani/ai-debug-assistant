"""Test the Pinecone vector service.

Two types of tests:
    1. Unit tests — test SimilarBug.to_prompt_string() without API calls
    2. Integration test — stores and queries real vectors in Pinecone
       (requires PINECONE_API_KEY and PINECONE_INDEX_NAME in .env)
"""

from __future__ import annotations

import asyncio
import uuid

from dotenv import load_dotenv
load_dotenv()

from backend.app.services.vector_service import SimilarBug, VectorService
from backend.app.services.embedding_service import EmbeddingService


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_similar_bug_prompt_string_full() -> None:
    """SimilarBug formats correctly with all fields."""
    bug = SimilarBug(
        session_id=str(uuid.uuid4()),
        score=0.92,
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
        error_function="get_user",
        filename="main.py",
    )
    text = bug.to_prompt_string()
    assert "92%" in text
    assert "IndexError" in text
    assert "get_user" in text
    assert "main.py" in text
    print("SimilarBug prompt string:     OK")


def test_similar_bug_prompt_string_minimal() -> None:
    """SimilarBug formats correctly without optional fields."""
    bug = SimilarBug(
        session_id=str(uuid.uuid4()),
        score=0.80,
        error_type="TypeError",
        error_message="unsupported operand",
        language="python",
        error_function=None,
        filename=None,
    )
    text = bug.to_prompt_string()
    assert "TypeError" in text
    assert "Function:" not in text
    print("SimilarBug minimal:           OK")


# ---------------------------------------------------------------------------
# Integration test — real Pinecone API
# ---------------------------------------------------------------------------

async def test_upsert_and_query() -> None:
    """
    Stores two similar bug vectors and one different bug vector
    in Pinecone, then queries to verify the similar ones are returned.
    """
    embedding_svc = EmbeddingService()
    vector_svc    = VectorService()

    # Generate test IDs
    test_user_id    = str(uuid.uuid4())
    test_project_id = str(uuid.uuid4())
    session_id_1    = str(uuid.uuid4())
    session_id_2    = str(uuid.uuid4())
    session_id_3    = str(uuid.uuid4())

    print("\nGenerating embeddings...")

    # Two similar IndexError bugs
    vec_1 = await embedding_svc.embed_error(
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
        error_function="get_user",
    )
    vec_2 = await embedding_svc.embed_error(
        error_type="IndexError",
        error_message="list assignment index out of range",
        language="python",
        error_function="update_user",
    )

    # One unrelated TypeError bug
    vec_3 = await embedding_svc.embed_error(
        error_type="TypeError",
        error_message="unsupported operand type for +: int and str",
        language="python",
        error_function="calculate_total",
    )

    print("Upserting vectors to Pinecone...")

    # Store all three
    vid_1 = await vector_svc.upsert_bug(
        session_id=session_id_1,
        vector=vec_1,
        user_id=test_user_id,
        project_id=test_project_id,
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
        error_function="get_user",
        filename="main.py",
    )
    vid_2 = await vector_svc.upsert_bug(
        session_id=session_id_2,
        vector=vec_2,
        user_id=test_user_id,
        project_id=test_project_id,
        error_type="IndexError",
        error_message="list assignment index out of range",
        language="python",
        error_function="update_user",
        filename="main.py",
    )
    vid_3 = await vector_svc.upsert_bug(
        session_id=session_id_3,
        vector=vec_3,
        user_id=test_user_id,
        project_id=test_project_id,
        error_type="TypeError",
        error_message="unsupported operand type",
        language="python",
        error_function="calculate_total",
        filename="utils.py",
    )

    print(f"Stored vector IDs: {vid_1[:8]}... {vid_2[:8]}... {vid_3[:8]}...")

    # Wait for Pinecone to index the vectors
    print("Waiting for Pinecone to index vectors...")
    await asyncio.sleep(5)

    # Query with a new IndexError vector
    query_vec = await embedding_svc.embed_error(
        error_type="IndexError",
        error_message="list index out of range",
        language="python",
    )

    print("Querying for similar bugs...")
    results = await vector_svc.query_similar_bugs(
        vector=query_vec,
        user_id=test_user_id,
        project_id=test_project_id,
        top_k=3,
        threshold=0.5,   # lower threshold for testing
    )

    print(f"\nResults found: {len(results)}")
    for r in results:
        print(f"  {r.error_type}: {r.error_message} "
              f"(score: {r.score:.4f})")

    assert len(results) >= 1, "Expected at least 1 similar bug"
    assert results[0].error_type == "IndexError", (
        f"Expected IndexError as top result, got {results[0].error_type}"
    )
    print("\nUpsert and query:             OK")

    # Clean up test vectors
    print("Cleaning up test vectors...")
    await vector_svc.delete_vector(vid_1)
    await vector_svc.delete_vector(vid_2)
    await vector_svc.delete_vector(vid_3)
    print("Cleanup:                      OK")

    # Check index stats
    stats = vector_svc.get_index_stats()
    print(f"Index stats: {stats}")
    print("\nAll vector tests passed.")


if __name__ == "__main__":
    # Unit tests
    test_similar_bug_prompt_string_full()
    test_similar_bug_prompt_string_minimal()
    print("\nUnit tests passed.")

    # Integration test
    print("\nRunning Pinecone integration test...")
    asyncio.run(test_upsert_and_query())