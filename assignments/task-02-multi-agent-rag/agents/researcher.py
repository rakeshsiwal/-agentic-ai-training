"""
agents/researcher.py — Researcher node: retrieves relevant chunks from ChromaDB.

The Researcher is called when the Supervisor determines that retrieved_docs
is empty (or that a re-retrieval is needed after the human rejected the context).
"""

from __future__ import annotations

from langchain_core.messages import AIMessage

from rag.retriever import retrieve
from state import PipelineState


def researcher_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: retrieve top-k relevant chunks for the current query.

    Side-effects:
        - Populates `retrieved_docs` in state.
        - Resets `approved` to False (new docs need fresh approval).
        - Appends an AI message summarising retrieval to `messages`.
    """
    query = state.get("query", "")
    if not query:
        raise ValueError("Researcher node received an empty query.")

    print(f"\n[researcher] Retrieving docs for query: '{query}'")
    chunks = retrieve(query, k=4)
    print(f"[researcher] Retrieved {len(chunks)} chunk(s).")

    summary = (
        f"Retrieved {len(chunks)} context chunk(s) for query: '{query}'.\n"
        + "\n".join(f"  [Chunk {i+1}] {c[:120]}…" for i, c in enumerate(chunks))
    )

    return {
        "retrieved_docs": chunks,
        "approved": False,          # Human must approve the new context
        "draft": "",                # Reset any stale draft
        "messages": [AIMessage(content=summary)],
    }
