"""
agent/graph.py — LangGraph agent with long-term memory + HITL
==============================================================
Graph structure:

  [load_memory_node] → [agent_node] ↔ [tool_node_with_hitl]
                                    └→ [save_memory_node] → END

HITL: interrupt() fires before any CONFIRMABLE_TOOL call.
Memory: loaded at start, saved at end via BaseStore.
MCP: tools loaded from two servers using per-server session context managers.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agent.llm import get_llm
from agent.memory_store import (
    build_memory_context,
    detect_preference,
    save_memory,
    summarize_session,
)
from agent.state import AgentState

_PROJECT_ROOT = str(Path(__file__).parent.parent)

# Tools that require human approval before execution
CONFIRMABLE_TOOLS = {"calculate", "get_current_time"}

_BASE_SYSTEM_PROMPT = """\
You are a helpful research assistant with access to a knowledge base about AI,
machine learning, vector databases, LangGraph, and prompt engineering.

When answering questions:
1. Use retrieve_context to look up relevant information from the knowledge base.
2. Use calculate for any arithmetic or math expressions.
3. Use get_current_date or get_current_time to answer date/time questions.
4. Apply any user preferences from memory (formatting style, citation preferences, etc.).
5. Be accurate and cite retrieved passages when they inform your answer.
6. For simple greetings or capability questions, answer directly without using tools.
"""


# ---------------------------------------------------------------------------
# MCP client helpers
# ---------------------------------------------------------------------------

_SERVER_NAMES = ["task04_tools", "time"]


def _get_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "task04_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "env": {"PYTHONPATH": _PROJECT_ROOT},
            },
            "time": {
                "command": sys.executable,
                "args": ["-m", "mcp_server_time", "--local-timezone=UTC"],
                "transport": "stdio",
            },
        }
    )


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def _build_graph(tools: list, store):
    """Compile the LangGraph graph given live MCP tools and a memory store."""

    llm = get_llm(temperature=0.0)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    # ── Node: load_memory_node ──────────────────────────────────────────────
    async def load_memory_node(state: AgentState, config: RunnableConfig) -> dict:
        user_id = state.get("user_id", "default")
        # Use first human message as query for memory relevance
        query = ""
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

        memory_context = await build_memory_context(store, user_id, query)
        return {"memory_context": memory_context}

    # ── Node: agent_node ────────────────────────────────────────────────────
    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        memory_context = state.get("memory_context", "")
        system_content = _BASE_SYSTEM_PROMPT + memory_context

        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_content)] + messages
        else:
            messages[0] = SystemMessage(content=system_content)

        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    # ── Node: tool_node_with_hitl ───────────────────────────────────────────
    async def tool_node_with_hitl(state: AgentState, config: RunnableConfig) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        tool_results = list(state.get("tool_results", []))

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in CONFIRMABLE_TOOLS:
                # Pause and surface approval request to human
                human_response = interrupt(
                    {
                        "action": "approve_tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    }
                )

                if not human_response.get("approved", True):
                    # Human rejected — inject a rejection tool message
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool call '{tool_name}' was declined by the user.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

            # Execute the tool
            if tool_name in tool_map:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args, config=config)
                    content = str(result)
                except Exception as exc:
                    content = f"Error: {exc}"
            else:
                content = f"Unknown tool: {tool_name}"

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"])
            )
            tool_results.append({"tool": tool_name, "args": tool_args, "result": content})

        return {"messages": tool_messages, "tool_results": tool_results}

    # ── Node: save_memory_node ──────────────────────────────────────────────
    async def save_memory_node(state: AgentState, config: RunnableConfig) -> dict:
        user_id = state.get("user_id", "default")
        messages = state["messages"]

        # Detect and save any user preferences stated in this session
        for msg in messages:
            if isinstance(msg, HumanMessage):
                pref = detect_preference(msg.content)
                if pref:
                    await save_memory(
                        store,
                        ("preferences", user_id),
                        key=pref[:50].replace(" ", "_").lower(),
                        value={
                            "value": pref,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )

        # Generate and save episode summary
        if len(messages) > 2:
            summary = await summarize_session(messages, get_llm(temperature=0.0))
            session_key = datetime.now(timezone.utc).strftime("session_%Y-%m-%d_%H-%M-%S")
            topics = []  # Could extract topics with another LLM call if desired
            await save_memory(
                store,
                ("episodes", user_id),
                key=session_key,
                value={
                    "summary": summary,
                    "topics": topics,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        return {}

    # ── Routing ─────────────────────────────────────────────────────────────
    def should_continue(state: AgentState) -> Literal["tools", "save_memory"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "save_memory"

    # ── Build graph ─────────────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node_with_hitl)
    builder.add_node("save_memory", save_memory_node)

    builder.set_entry_point("load_memory")
    builder.add_edge("load_memory", "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "save_memory": "save_memory"},
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("save_memory", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Public session context manager
# ---------------------------------------------------------------------------

@asynccontextmanager
async def agent_session(store, checkpointer=None):
    """
    Async context manager that opens MCP server sessions and keeps them alive.

    Usage:
        async with agent_session(store) as run:
            answer = await run("What is AI?", user_id="alice")
    """
    client = _get_mcp_client()
    all_tools = []
    sessions = []

    try:
        for server_name in _SERVER_NAMES:
            cm = client.session(server_name)
            session = await cm.__aenter__()
            sessions.append((cm, session))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

        graph = _build_graph(all_tools, store)
        if checkpointer:
            graph = _build_graph(all_tools, store)
            # Re-compile with checkpointer for HITL persistence
            builder = _get_builder(all_tools, store)
            graph = builder.compile(checkpointer=checkpointer)

        async def run(
            question: str,
            user_id: str = "default",
            thread_id: str | None = None,
            metadata: dict | None = None,
        ) -> str:
            meta = metadata or {"mode": "interactive", "user_id": user_id}
            config = RunnableConfig(
                metadata=meta,
                run_name="task05-agent",
                configurable={"thread_id": thread_id or user_id},
            )
            state_out = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "user_id": user_id,
                    "memory_context": "",
                    "pending_interrupt": None,
                    "run_metadata": meta,
                    "tool_results": [],
                },
                config=config,
            )
            return state_out["messages"][-1].content

        yield run, graph

    finally:
        for cm, session in reversed(sessions):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass


def _get_builder(tools: list, store):
    """Return a StateGraph builder (for use with external checkpointer)."""
    llm = get_llm(temperature=0.0)
    tool_map = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    async def load_memory_node(state, config):
        user_id = state.get("user_id", "default")
        query = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        memory_context = await build_memory_context(store, user_id, query)
        return {"memory_context": memory_context}

    def agent_node(state, config):
        memory_context = state.get("memory_context", "")
        system_content = _BASE_SYSTEM_PROMPT + memory_context
        messages = list(state["messages"])
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_content)] + messages
        else:
            messages[0] = SystemMessage(content=system_content)
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    async def tool_node_with_hitl(state, config):
        last_message = state["messages"][-1]
        tool_messages = []
        tool_results = list(state.get("tool_results", []))

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in CONFIRMABLE_TOOLS:
                human_response = interrupt(
                    {
                        "action": "approve_tool_call",
                        "tool_name": tool_name,
                        "tool_args": tool_args,
                    }
                )
                if not human_response.get("approved", True):
                    tool_messages.append(
                        ToolMessage(
                            content=f"Tool call '{tool_name}' was declined by the user.",
                            tool_call_id=tool_call["id"],
                        )
                    )
                    continue

            if tool_name in tool_map:
                try:
                    result = await tool_map[tool_name].ainvoke(tool_args, config=config)
                    content = str(result)
                except Exception as exc:
                    content = f"Error: {exc}"
            else:
                content = f"Unknown tool: {tool_name}"

            tool_messages.append(
                ToolMessage(content=content, tool_call_id=tool_call["id"])
            )
            tool_results.append({"tool": tool_name, "args": tool_args, "result": content})

        return {"messages": tool_messages, "tool_results": tool_results}

    async def save_memory_node(state, config):
        user_id = state.get("user_id", "default")
        messages = state["messages"]
        for msg in messages:
            if isinstance(msg, HumanMessage):
                pref = detect_preference(msg.content)
                if pref:
                    await save_memory(
                        store,
                        ("preferences", user_id),
                        key=pref[:50].replace(" ", "_").lower(),
                        value={
                            "value": pref,
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
        if len(messages) > 2:
            summary = await summarize_session(messages, llm)
            session_key = datetime.now(timezone.utc).strftime("session_%Y-%m-%d_%H-%M-%S")
            await save_memory(
                store,
                ("episodes", user_id),
                key=session_key,
                value={
                    "summary": summary,
                    "topics": [],
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        return {}

    def should_continue(state) -> Literal["tools", "save_memory"]:
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return "save_memory"

    builder = StateGraph(AgentState)
    builder.add_node("load_memory", load_memory_node)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node_with_hitl)
    builder.add_node("save_memory", save_memory_node)
    builder.set_entry_point("load_memory")
    builder.add_edge("load_memory", "agent")
    builder.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "save_memory": "save_memory"}
    )
    builder.add_edge("tools", "agent")
    builder.add_edge("save_memory", END)
    return builder
