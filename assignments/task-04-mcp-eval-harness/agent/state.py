"""
agent/state.py — LangGraph agent state schema
"""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared across all nodes in the ReAct agent graph."""

    # Full conversation history; add_messages reducer appends rather than replacing.
    messages: Annotated[list[BaseMessage], add_messages]

    # Metadata forwarded from the caller (used for LangSmith tagging).
    run_metadata: dict
