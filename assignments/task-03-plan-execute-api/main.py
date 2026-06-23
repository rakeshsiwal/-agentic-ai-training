"""
CLI entry point for local testing (no API server needed).

Usage:
    python main.py
    python main.py --query "What is the Send() API in LangGraph?" --thread-id my_run_1
"""
from __future__ import annotations

import argparse
import uuid

from graph import graph


def run(query: str, thread_id: str, max_iterations: int = 3) -> str:
    initial_state = {
        "messages": [],
        "original_query": query,
        "sub_tasks": [],
        "retrieved_docs": [],
        "aggregated_context": [],
        "draft": "",
        "final_answer": "",
        "evaluation": {},
        "iteration": 0,
        "max_iterations": max_iterations,
        "current_sub_task": {},
    }

    config = {"configurable": {"thread_id": thread_id}}

    print("\n" + "=" * 70)
    print(f"QUERY: {query}")
    print(f"THREAD ID: {thread_id}")
    print("=" * 70 + "\n")

    final = graph.invoke(initial_state, config=config)

    answer = final.get("final_answer") or final.get("draft", "(no answer produced)")

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(answer)
    print("=" * 70 + "\n")

    return answer


def main():
    parser = argparse.ArgumentParser(description="Plan-and-Execute Research CLI")
    parser.add_argument(
        "--query", "-q",
        default="What is LangGraph's Send() API and how does it enable parallel execution?",
        help="Research question",
    )
    parser.add_argument(
        "--thread-id", "-t",
        default=str(uuid.uuid4()),
        help="Thread ID for checkpoint persistence",
    )
    parser.add_argument(
        "--max-iterations", "-m",
        type=int, default=3,
        help="Max reflection iterations",
    )
    args = parser.parse_args()

    run(args.query, args.thread_id, args.max_iterations)


if __name__ == "__main__":
    main()