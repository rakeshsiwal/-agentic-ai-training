"""
graph.py — LangGraph graph definition for the Research Agent.

Graph topology:
  [START]
     ↓
  agent_node
     ↓ (conditional)
  ┌──────────────────────────────┐
  │  tool_node  → agent_node    │  ← loop: tool results feed back to agent
  │  summarizer_node            │  ← produce a clean answer for the user
  │  END                        │  ← user said goodbye / agent finished
  └──────────────────────────────┘
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from .nodes import agent_node, should_continue, summarizer_node, tool_node
from .state import AgentState


def build_graph() -> StateGraph:
    """Compile and return the research agent graph with in-memory checkpointing."""

    builder = StateGraph(AgentState)

    # ── Register nodes ──────────────────────────────────────────────────────
    builder.add_node("agent_node", agent_node)
    builder.add_node("tool_node", tool_node)
    builder.add_node("summarizer_node", summarizer_node)

    # ── Entry point ──────────────────────────────────────────────────────────
    builder.add_edge(START, "agent_node")

    # ── Conditional routing after agent_node ────────────────────────────────
    builder.add_conditional_edges(
        "agent_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "summarizer_node": "summarizer_node",
            "end": END,
        },
    )

    # ── After tool execution, go back to agent to reason about results ───────
    builder.add_edge("tool_node", "agent_node")

    # ── After summarization, the turn is complete ────────────────────────────
    builder.add_edge("summarizer_node", END)

    # ── Compile with MemorySaver for cross-turn conversation memory ──────────
    checkpointer = MemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph
