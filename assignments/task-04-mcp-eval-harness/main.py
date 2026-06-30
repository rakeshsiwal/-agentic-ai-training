"""
main.py — Interactive CLI entry point
======================================
Starts an interactive conversation with the MCP-backed LangGraph agent.

Usage:
    python main.py

Type 'exit' or 'quit' to end the session.
LangSmith traces are tagged with {"mode": "interactive"}.
"""

from __future__ import annotations

import asyncio

from dotenv import load_dotenv

load_dotenv()


async def main() -> None:
    from agent.graph import agent_session

    print("\n" + "=" * 60)
    print("Task-04 MCP Agent — Interactive Mode")
    print("Knowledge base: AI, ML, Vector DBs, LangGraph, Prompts")
    print("Tools: retrieve_context, calculate, get_current_date, get_current_time")
    print("Type 'exit' or 'quit' to end.")
    print("=" * 60 + "\n")

    print("Initialising agent (loading MCP tools)…")

    async with agent_session() as run:
        print("Ready!\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break

            try:
                response = await run(
                    user_input,
                    metadata={"mode": "interactive"},
                )
            except Exception as exc:
                response = f"[Error: {exc}]"

            print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    asyncio.run(main())