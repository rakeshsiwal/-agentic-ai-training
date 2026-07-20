"""
main.py — Interactive CLI with long-term memory + HITL
=======================================================
Usage:
    python main.py
    python main.py --user-id alice

Memory persists across runs in data/memory.db (SqliteSaver for checkpoints,
InMemoryStore for this demo — swap to a persistent store for full cross-restart
memory; the API server uses SqliteSaver for full persistence).
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


async def main(user_id: str = "default") -> None:
    from langchain_core.messages import HumanMessage
    from langchain_core.runnables import RunnableConfig
    from langgraph.store.memory import InMemoryStore
    from langgraph.types import Command

    from agent.graph import CONFIRMABLE_TOOLS, _get_builder
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    PROJECT_ROOT = str(Path(__file__).parent)
    SERVER_NAMES = ["task04_tools", "time"]

    store = InMemoryStore()

    print(f"\n{'='*60}")
    print(f"Task-05 Agent — Interactive Mode")
    print(f"User: {user_id}")
    print(f"Memory: InMemoryStore (session-scoped)")
    print(f"HITL: enabled for {CONFIRMABLE_TOOLS}")
    print(f"Type 'exit' to quit, 'memories' to view stored memories.")
    print(f"{'='*60}\n")

    print("Initialising agent (loading MCP tools)...")

    client = MultiServerMCPClient(
        {
            "task04_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "env": {"PYTHONPATH": PROJECT_ROOT},
            },
            "time": {
                "command": sys.executable,
                "args": ["-m", "mcp_server_time", "--local-timezone=UTC"],
                "transport": "stdio",
            },
        }
    )

    all_tools = []
    sessions = []

    try:
        for server_name in SERVER_NAMES:
            cm = client.session(server_name)
            session = await cm.__aenter__()
            sessions.append((cm, session))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

        builder = _get_builder(all_tools, store)
        graph = builder.compile()

        print("Ready!\n")
        thread_id = str(uuid.uuid4())

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() == "exit":
                print("Goodbye!")
                break

            if user_input.lower() == "memories":
                from agent.memory_store import load_memories
                prefs = await load_memories(store, ("preferences", user_id))
                episodes = await load_memories(store, ("episodes", user_id))
                print(f"\n--- Memories for {user_id} ---")
                print(f"Preferences ({len(prefs)}): {prefs}")
                print(f"Episodes ({len(episodes)}): {[e.get('summary', '')[:100] for e in episodes]}")
                print("---\n")
                continue

            config = RunnableConfig(
                metadata={"mode": "interactive", "user_id": user_id},
                run_name="task05-interactive",
                configurable={"thread_id": thread_id},
            )

            initial_state = {
                "messages": [HumanMessage(content=user_input)],
                "user_id": user_id,
                "memory_context": "",
                "pending_interrupt": None,
                "run_metadata": {"mode": "interactive", "user_id": user_id},
                "tool_results": [],
            }

            try:
                result = await graph.ainvoke(initial_state, config=config)
                response = result["messages"][-1].content
                print(f"\nAgent: {response}\n")

            except Exception as exc:
                # Check for HITL interrupt
                state = await graph.aget_state(config)
                interrupted = False
                if state and state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            interrupt_data = task.interrupts[0].value
                            tool_name = interrupt_data.get("tool_name", "")
                            tool_args = interrupt_data.get("tool_args", {})

                            print(f"\n⚠️  APPROVAL REQUIRED")
                            print(f"   Tool: {tool_name}")
                            print(f"   Args: {tool_args}")
                            approval = input("   Approve? (y/n): ").strip().lower()
                            approved = approval in ("y", "yes")

                            resume_config = RunnableConfig(
                                metadata={"mode": "hitl_resumed", "thread_id": thread_id},
                                run_name="task05-resumed",
                                configurable={"thread_id": thread_id},
                            )
                            result = await graph.ainvoke(
                                Command(resume={"approved": approved}),
                                config=resume_config,
                            )
                            response = result["messages"][-1].content
                            print(f"\nAgent: {response}\n")
                            interrupted = True
                            break

                if not interrupted:
                    print(f"\n[Error: {exc}]\n")

    finally:
        for cm, session in reversed(sessions):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


if __name__ == "__main__":
    user_id = "default"
    if len(sys.argv) > 2 and sys.argv[1] == "--user-id":
        user_id = sys.argv[2]
    elif len(sys.argv) > 1:
        user_id = sys.argv[1]
    asyncio.run(main(user_id))
