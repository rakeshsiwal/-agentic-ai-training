"""
main.py — CLI entry point for the multi-agent RAG pipeline.

Usage:
    # 1. Ingest documents first (run once):
    python -m rag.ingest

    # 2. Start the interactive pipeline:
    python main.py

    # 3. Optional: pass a query directly:
    python main.py "What is self-attention?"
"""

from __future__ import annotations

import sys
import uuid

from langgraph.types import Command

from graph import graph


BANNER = """
╔══════════════════════════════════════════════════════════╗
║        Multi-Agent RAG System  (LangGraph)               ║
║  Supervisor → Researcher → [HiTL Review] → Writer        ║
╚══════════════════════════════════════════════════════════╝
"""

def _print_separator(char: str = "─", width: int = 60) -> None:
    print(char * width)


def run_pipeline(query: str) -> None:
    """Execute a full RAG pipeline run with human-in-the-loop for one query."""

    # Each run gets its own thread_id so the checkpointer isolates state
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "messages": [],
        "query": query,
        "retrieved_docs": [],
        "approved": False,
        "draft": "",
        "final_answer": "",
        "next_agent": "",
    }

    print(f"\n🔍 Query: {query}\n")
    _print_separator()

    # ── First invocation ───────────────────────────────────────────────────────
    # Runs until the first interrupt (before human_review_node)
    result = graph.invoke(initial_state, config=config)

    # ── Human-in-the-loop loop ─────────────────────────────────────────────────
    # The graph will interrupt before human_review_node on every retrieval round.
    while True:
        # Check whether execution is suspended at human_review_node
        snapshot = graph.get_state(config)
        if not snapshot.next:
            # No more nodes to run — pipeline is complete
            break

        next_nodes = list(snapshot.next)
        if "human_review_node" not in next_nodes:
            # Suspended at a different node or finished
            break

        # Show the interrupt payload (retrieved docs preview)
        interrupts = snapshot.tasks
        for task in interrupts:
            for interrupt_obj in getattr(task, "interrupts", []):
                payload = interrupt_obj.value
                print(payload.get("message", ""))

        # Collect human decision
        try:
            user_input = input("Your decision: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[main] Pipeline cancelled by user.")
            return

        if not user_input:
            user_input = "y"  # default: approve

        # Resume the graph with the human's decision
        result = graph.invoke(
            Command(resume=user_input),
            config=config,
        )

    # ── Show final answer ──────────────────────────────────────────────────────
    final = result.get("final_answer", "")
    if final:
        _print_separator("═")
        print("\n✅  FINAL ANSWER\n")
        _print_separator("═")
        print(final)
        _print_separator("═")
    else:
        print("\n⚠️  Pipeline finished without producing a final answer.")


def main() -> None:
    print(BANNER)

    # Accept query from CLI argument or interactive prompt
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        print("Type your question (or 'quit' to exit).\n")
        while True:
            try:
                query = input("❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                return
            if query.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                return
            if query:
                break
        print()

    run_pipeline(query)


if __name__ == "__main__":
    main()
