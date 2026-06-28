"""
Retrieval layer: top-k similarity search against a session's Chroma store,
returning each result with its cosine similarity score so the frontend's
Source Citation Drawer can display it.

R-02/R-03: Uses similarity_search_with_score() directly instead of
re-embedding candidates, which eliminates redundant computation.
"""
from typing import List, Tuple

from langchain_core.documents import Document

from .ingest import get_vectorstore

TOP_K = 4
RERANK_TOP_N = 3


def retrieve(session_id: str, query: str) -> List[Tuple[Document, float]]:
    """
    Returns a list of (Document, cosine_similarity_score) tuples, sorted by
    score descending, limited to RERANK_TOP_N.
    """
    vectorstore = get_vectorstore(session_id)

    # R-02/R-03: similarity_search_with_score returns (doc, distance) tuples
    # directly from the Chroma HNSW index — no need to re-embed candidates.
    # Chroma returns L2 distance by default; lower is better.
    results_with_scores = vectorstore.similarity_search_with_score(query, k=TOP_K)
    if not results_with_scores:
        return []

    # Convert L2 distance to a cosine-like similarity score (0–1 range).
    # Chroma's default metric is L2; for normalized embeddings:
    # cosine_similarity ≈ 1 - (l2_distance² / 2)
    scored = []
    for doc, distance in results_with_scores:
        similarity = max(0.0, 1.0 - (distance ** 2) / 2.0) if distance >= 0 else 0.0
        scored.append((doc, round(similarity, 4)))

    # Sort by similarity descending and keep the top N.
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:RERANK_TOP_N]
