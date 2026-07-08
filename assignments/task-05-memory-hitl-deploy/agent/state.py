"""
agent/state.py — AgentState extended with memory + interrupt fields
"""

from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared across all nodes in the agent graph."""

    # Full conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    # user_id for scoping memories
    user_id: str

    # Memory context injected at session start (preferences + episode summaries)
    memory_context: str

    # Pending HITL interrupt payload (None when no interrupt is active)
    pending_interrupt: dict | None

    # Metadata forwarded to LangSmith
    run_metadata: dict

    # Accumulated tool results for this session
    tool_results: list[dict]
