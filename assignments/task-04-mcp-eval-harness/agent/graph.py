"""
agent/graph.py — LangGraph ReAct-style agent
=============================================
Builds a compiled LangGraph graph whose tools are loaded dynamically from MCP.

As of langchain-mcp-adapters 0.1.0, MultiServerMCPClient cannot be used as
a context manager directly. Instead we open each server session individually
via client.session(server_name) and keep them alive for the whole run.

Graph structure:
    [agent_node] ──(has tool calls)──▶ [tool_node] ──▶ back to agent_node
                └──(no tool calls)──▶ END
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Literal

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agent.llm import get_llm
from agent.mcp_client import get_mcp_client, SERVER_NAMES
from agent.state import AgentState

_SYSTEM_PROMPT = """\
You are a helpful research assistant with access to a knowledge base about AI,
machine learning, vector databases, LangGraph, and prompt engineering, as well
as utilities for math and time.

When answering questions:
1. Use retrieve_context to look up relevant information from the knowledge base.
2. Use calculate for any arithmetic or math expressions.
3. Use get_current_date or get_current_time to answer date/time questions.
4. Cite the retrieved passages when they inform your answer.
5. Be concise and accurate.
6. For simple greetings or capability questions, answer directly without using tools.
"""


def _should_continue(state: AgentState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "__end__"


def _build_graph(tools: list):
    llm = get_llm(temperature=0.0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState, config: RunnableConfig) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=_SYSTEM_PROMPT)] + list(messages)
        response = llm_with_tools.invoke(messages, config=config)
        return {"messages": [response]}

    tool_node = ToolNode(tools)
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.set_entry_point("agent")
    builder.add_conditional_edges(
        "agent", _should_continue, {"tools": "tools", "__end__": END}
    )
    builder.add_edge("tools", "agent")
    return builder.compile()


@asynccontextmanager
async def agent_session():
    """
    Async context manager that opens every MCP server session and keeps them
    alive for the full duration of the block.

    Usage:
        async with agent_session() as run:
            answer = await run("What is AI?")
    """
    client = get_mcp_client()
    all_tools = []

    # Stack of context managers we need to exit cleanly
    sessions = []
    try:
        for server_name in SERVER_NAMES:
            cm = client.session(server_name)
            session = await cm.__aenter__()
            sessions.append((cm, session))
            tools = await load_mcp_tools(session)
            all_tools.extend(tools)

        graph = _build_graph(all_tools)

        async def run(question: str, metadata: dict | None = None) -> str:
            from langchain_core.messages import HumanMessage

            meta = metadata or {"mode": "interactive"}
            config = RunnableConfig(metadata=meta, run_name="task04-agent")
            state_out = await graph.ainvoke(
                {
                    "messages": [HumanMessage(content=question)],
                    "run_metadata": meta,
                },
                config=config,
            )
            return state_out["messages"][-1].content

        yield run

    finally:
        for cm, session in reversed(sessions):
            try:
                await cm.__aexit__(None, None, None)
            except Exception:
                pass