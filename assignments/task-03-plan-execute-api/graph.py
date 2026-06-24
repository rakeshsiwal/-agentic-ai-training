"""
Full LangGraph pipeline:
  START → planner_node
        → [researcher_node × N]  (Send() fan-out, parallel)
        → aggregator_node
        → writer_node
        → evaluator_node
        → END  or  back to planner_node  (reflection loop)

Compiled with SqliteSaver for cross-restart persistence.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agents.aggregator import aggregator_node
from agents.evaluator import evaluator_node
from agents.planner import planner_node
from agents.researcher import researcher_node
from agents.writer import writer_node
from state import PlanExecuteState

# ── Persistence ─────────────────────────────────────────────────────────────

DB_PATH = Path(__file__).parent / "data" / "checkpoints.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def _make_checkpointer():
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        return SqliteSaver(conn)
    except ImportError:
        # Fallback: in-memory (no persistence)
        from langgraph.checkpoint.memory import MemorySaver
        print("WARNING: langgraph-checkpoint-sqlite not found; using in-memory saver.")
        return MemorySaver()


# ── Conditional edges ────────────────────────────────────────────────────────

def dispatch_researchers(state: PlanExecuteState) -> list[Send]:
    """Fan-out: spawn one researcher_node per sub-task."""
    return [
        Send("researcher_node", {**state, "current_sub_task": task})
        for task in state["sub_tasks"]
    ]


def route_after_evaluation(state: PlanExecuteState) -> str:
    """Decide whether to accept the answer or re-plan."""
    ev = state.get("evaluation") or {}
    if state.get("iteration", 0) >= state.get("max_iterations", 3):
        print("[Router] max iterations reached — finishing.")
        return END
    if ev.get("faithfulness", 0) < 3 or ev.get("relevance", 0) < 3:
        print("[Router] quality too low — re-planning.")
        return "planner_node"
    print("[Router] quality acceptable — finishing.")
    return END


# ── Graph construction ───────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(PlanExecuteState)

    builder.add_node("planner_node", planner_node)
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("aggregator_node", aggregator_node)
    builder.add_node("writer_node", writer_node)
    builder.add_node("evaluator_node", evaluator_node)

    builder.add_edge(START, "planner_node")

    # Fan-out via Send()
    builder.add_conditional_edges(
        "planner_node",
        dispatch_researchers,
        # No explicit destination list needed — Send() handles routing
    )

    # All researcher branches converge at aggregator
    builder.add_edge("researcher_node", "aggregator_node")
    builder.add_edge("aggregator_node", "writer_node")
    builder.add_edge("writer_node", "evaluator_node")

    builder.add_conditional_edges(
        "evaluator_node",
        route_after_evaluation,
        {END: END, "planner_node": "planner_node"},
    )

    checkpointer = _make_checkpointer()
    return builder.compile(checkpointer=checkpointer)


# Singleton graph — imported by server.py and main.py
graph = build_graph()