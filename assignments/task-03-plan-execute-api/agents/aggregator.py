"""
Aggregator agent — collects all chunks from parallel researcher branches,
deduplicates them, and stores a clean list in `aggregated_context`.
"""
from __future__ import annotations

from state import PlanExecuteState

_SIMILARITY_THRESHOLD = 0.95  # Jaccard similarity for near-duplicate detection


def _jaccard(a: str, b: str) -> float:
    """Simple token-level Jaccard similarity."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _deduplicate(chunks: list[str]) -> list[str]:
    """Remove chunks that are exact duplicates or very similar (Jaccard ≥ threshold)."""
    unique: list[str] = []
    for candidate in chunks:
        is_dup = any(_jaccard(candidate, kept) >= _SIMILARITY_THRESHOLD for kept in unique)
        if not is_dup:
            unique.append(candidate)
    return unique


def aggregator_node(state: PlanExecuteState) -> dict:
    raw_docs = state.get("retrieved_docs", [])
    original_query = state["original_query"]

    deduped = _deduplicate(raw_docs)

    # Optional: sort by rough relevance (word overlap with original query)
    query_tokens = set(original_query.lower().split())
    def relevance_score(chunk: str) -> int:
        return len(query_tokens & set(chunk.lower().split()))

    deduped.sort(key=relevance_score, reverse=True)

    print(
        f"[Aggregator] {len(raw_docs)} raw chunk(s) → "
        f"{len(deduped)} after dedup (removed {len(raw_docs) - len(deduped)})"
    )

    return {"aggregated_context": deduped}