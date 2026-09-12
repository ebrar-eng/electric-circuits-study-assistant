"""Retrieval (Plan Phase 2, Week 3): given a query, find the most relevant chunks.

For a dataset this small, brute-force cosine similarity over every stored
embedding is fast and simple.
"""

import math

import db


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def get_scored_candidates(embedding_client, query: str, top_k: int = 5) -> list[dict]:
    """Embed the query and return the top_k chunks by score, unfiltered.

    Unlike get_top_chunks, this does NOT apply min_score - it's meant for
    debug output, so you can see what the retriever actually found (and
    its score) even for chunks that would otherwise be dropped.

    Each returned dict has: id, source, content, embedding, score.
    """
    conn = db.get_connection()
    chunks = db.fetch_all_chunks(conn)
    conn.close()

    if not chunks:
        return []

    query_response = embedding_client.generate_embedding(query)
    query_embedding = query_response.data[0].embedding

    scored = [
        {**chunk, "score": cosine_similarity(query_embedding, chunk["embedding"])}
        for chunk in chunks
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


def get_top_chunks(embedding_client, query: str, top_k: int = 3, min_score: float = 0.5) -> list[dict]:
    """Embed the query and return the top_k most similar stored chunks.

    Chunks scoring below min_score are dropped entirely rather than being
    handed to the chat model as "context" - a weak local model will often
    answer from its own training data if it's given any context at all,
    regardless of how irrelevant that context actually is. Filtering here,
    in code, doesn't depend on the model choosing to follow instructions.

    Each returned dict has: id, source, content, embedding, score.
    """
    candidates = get_scored_candidates(embedding_client, query, top_k=max(top_k, 5))
    relevant = [c for c in candidates if c["score"] >= min_score]
    return relevant[:top_k]
