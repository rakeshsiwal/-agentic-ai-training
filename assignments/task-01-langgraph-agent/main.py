#!/usr/bin/env python3
"""
main.py — Interactive entry point for the LangGraph Research Agent.

Run:
    python main.py

Quit by typing: exit, quit, bye, or pressing Ctrl-C / Ctrl-D
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import uuid
import sys
from pathlib import Path

# Load .env before anything else so API keys are available
from dotenv import load_dotenv

load_dotenv()

from langchain_core.messages import HumanMessage

from agent import build_graph

# ── ANSI colour helpers ──────────────────────────────────────────────────────

def _c(text: str, code: str) -> str:
    """Wrap text in an ANSI colour code (no-op if stdout is not a TTY)."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"

BOLD    = lambda t: _c(t, "1")
CYAN    = lambda t: _c(t, "36")
GREEN   = lambda t: _c(t, "32")
YELLOW  = lambda t: _c(t, "33")
DIM     = lambda t: _c(t, "2")


BANNER = f"""
{CYAN('╔══════════════════════════════════════════════════╗')}
{CYAN('║')}   {BOLD('🔍 Personal Research Agent')}  (LangGraph)        {CYAN('║')}
{CYAN('║')}   Type a topic to research, or ask a follow-up.  {CYAN('║')}
{CYAN('║')}   Type {YELLOW("'exit'")} or {YELLOW("'quit'")} to leave.                {CYAN('║')}
{CYAN('╚══════════════════════════════════════════════════╝')}
"""

EXIT_COMMANDS = {"exit", "quit", "bye", "goodbye", ":q"}


def run():
    print(BANNER)

    graph = build_graph()

    # Each run gets a unique thread_id so memory is session-scoped.
    # Swap for a fixed ID to persist across restarts (requires SqliteSaver).
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print(DIM(f"  Session ID: {thread_id[:8]}…\n"))

    while True:
        try:
            user_input = input(f"{GREEN('You')} › ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n" + DIM("Session ended."))
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            print(f"\n{CYAN('Agent')} › Goodbye! Happy researching. 👋\n")
            break

        print()  # breathing room

        # Stream the graph; collect and print each message as it arrives
        final_output = None
        try:
            for event in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values",
            ):
                # 'event' is the full state snapshot after each node
                last_msg = event["messages"][-1]

                # Print tool activity in dim style so it doesn't clutter
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tc in last_msg.tool_calls:
                        args_preview = str(tc["args"])[:80]
                        print(DIM(f"  ⚙  {tc['name']}({args_preview})"))

                final_output = event

        except Exception as exc:
            print(f"\n{YELLOW('⚠  Error:')} {exc}\n")
            print(DIM("  Check your API key and network connection."))
            continue

        # Print the agent's last substantive reply
        if final_output:
            last = final_output["messages"][-1]
            content = last.content if hasattr(last, "content") else ""
            # Handle Groq returning content as a list of blocks
            if isinstance(content, list):
                content = " ".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in content
                )
            if content and not getattr(last, "tool_calls", None):
                print(f"\n{CYAN('Agent')} › {content}\n")
            elif not content:
                # Fallback: find the last message with visible text
                for msg in reversed(final_output["messages"]):
                    c = msg.content if hasattr(msg, "content") else ""
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)
                    if c and not getattr(msg, "tool_calls", None):
                        print(f"\n{CYAN('Agent')} › {c}\n")
                        break

        print(DIM("─" * 52))


if __name__ == "__main__":
    run()
