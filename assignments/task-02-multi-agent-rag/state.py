"""
state.py — Shared PipelineState for the multi-agent RAG system.

All agents read from and write to this TypedDict.
LangGraph merges partial updates returned by each node.
"""

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class PipelineState(TypedDict):
    # Full conversation history; add_messages merges lists rather than overwriting.
    messages: Annotated[list, add_messages]

    # The user's current question (may be rewritten by a query-rewriter node).
    query: str

    # Raw text chunks returned by the vector store retriever.
    retrieved_docs: list[str]

    # Set to True once the human approves the retrieved context.
    approved: bool

    # Draft answer produced by the Writer agent (before final polish).
    draft: str

    # Final polished answer surfaced to the user.
    final_answer: str

    # Supervisor's routing decision — which agent should run next.
    next_agent: str
