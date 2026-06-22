"""
graph.py — Wires all agent nodes into a LangGraph StateGraph.

Topology:
    START → supervisor_node
              ↓ conditional on next_agent
         researcher_node → [INTERRUPT] human_review_node
              ↓ (approved)
         writer_node → supervisor_node → END

The human_review_node is an interrupt point: LangGraph pauses execution there
and resumes only after the caller invokes graph.invoke() with a Command that
supplies the human's decision.
"""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.researcher import researcher_node
from agents.supervisor import supervisor_node
from agents.writer import writer_node
from state import PipelineState


# ── Human-in-the-Loop node ────────────────────────────────────────────────────

def human_review_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: pause and ask the human to approve the retrieved context.

    The `interrupt()` call suspends execution.  The graph is resumed by the
    caller (see main.py) passing a Command(resume=<user_input>).

    Accepted inputs:
        "y" or "yes"           → approve; continue to writer
        "n" or "no"            → reject; go back to researcher (re-retrieval)
        "r <new query>"        → replace query and re-retrieve
    """
    docs = state.get("retrieved_docs", [])

    # Build a human-readable preview of the retrieved docs
    preview_lines = []
    for i, chunk in enumerate(docs, 1):
        preview_lines.append(f"\n[Chunk {i}]\n{chunk[:300]}{'…' if len(chunk) > 300 else ''}")
    preview = "\n".join(preview_lines)

    # Suspend execution — the runner must resume with a string decision
    decision: str = interrupt({
        "retrieved_docs": docs,
        "preview": preview,
        "message": (
            "\n" + "═" * 60 + "\n"
            "📄 RETRIEVED CONTEXT\n"
            + "═" * 60
            + preview
            + "\n" + "═" * 60 + "\n"
            "Approve this context?\n"
            "  y / yes          → proceed to Writer\n"
            "  n / no           → re-retrieve (same query)\n"
            "  r <new query>    → replace query and re-retrieve\n"
            + "═" * 60
        ),
    })

    decision = (decision or "").strip().lower()

    if decision in {"y", "yes"}:
        print("[human_review] ✅ Approved.")
        return {"approved": True}

    if decision.startswith("r "):
        new_query = decision[2:].strip()
        print(f"[human_review] 🔄 Query replaced → '{new_query}'")
        return {
            "approved": False,
            "query": new_query,
            "retrieved_docs": [],
            "draft": "",
        }

    # "n" / "no" / anything else → reject
    print("[human_review] ❌ Rejected — will re-retrieve.")
    return {
        "approved": False,
        "retrieved_docs": [],
        "draft": "",
    }


# ── Routing helper ─────────────────────────────────────────────────────────────

def route_from_supervisor(state: PipelineState) -> str:
    """
    Conditional edge: map `next_agent` value to the next node name.
    Returns the node name to visit after supervisor_node.
    """
    destination = state.get("next_agent", "researcher")
    # Guard: after researcher runs, we always pass through human review
    # before writer — but the supervisor may also route directly to writer
    # if approved is already True from a previous loop iteration.
    mapping = {
        "researcher": "researcher_node",
        "writer": "writer_node",
        "end": END,
    }
    return mapping.get(destination, "researcher_node")


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Construct and compile the multi-agent RAG graph.

    Returns:
        A compiled LangGraph app ready to call with .invoke() or .stream().
    """
    builder = StateGraph(PipelineState)

    # Register nodes
    builder.add_node("supervisor_node", supervisor_node)
    builder.add_node("researcher_node", researcher_node)
    builder.add_node("human_review_node", human_review_node)
    builder.add_node("writer_node", writer_node)

    # Entry point
    builder.add_edge(START, "supervisor_node")

    # Conditional routing out of supervisor
    builder.add_conditional_edges(
        "supervisor_node",
        route_from_supervisor,
        {
            "researcher_node": "researcher_node",
            "writer_node": "writer_node",
            END: END,
        },
    )

    # After researcher → always go to human review
    builder.add_edge("researcher_node", "human_review_node")

    # After human review → back to supervisor (which routes to writer or researcher)
    builder.add_edge("human_review_node", "supervisor_node")

    # After writer → back to supervisor (which will route to END)
    builder.add_edge("writer_node", "supervisor_node")

    # Compile with memory checkpointer so interrupt/resume works
    memory = MemorySaver()
    app = builder.compile(
        checkpointer=memory,
        interrupt_before=["human_review_node"],
    )
    return app


# Singleton — import this in main.py
graph = build_graph()
