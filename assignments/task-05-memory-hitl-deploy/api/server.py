"""
api/server.py — FastAPI server
==============================
Endpoints:
  GET  /health
  POST /chat                    — start a conversation; may return "interrupted"
  POST /resume                  — resume an interrupted thread with approval decision
  GET  /memories/{user_id}      — list stored memories for a user
  DELETE /memories/{user_id}    — (bonus) wipe a user's memories

Run:
  uvicorn api.server:app --reload --port 8000
"""

from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from agent.graph import CONFIRMABLE_TOOLS, _get_builder
from agent.memory_store import load_memories
from api.schemas import (
    ChatRequest, ChatResponse, HealthResponse,
    MemoriesResponse, PendingApproval,
    ResumeRequest, ResumeResponse,
)

load_dotenv()

_PROJECT_ROOT = str(Path(__file__).parent.parent)
_SERVER_NAMES = ["task04_tools", "time"]

_app_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build graph on startup using MemorySaver (no async context manager needed)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools

    store = InMemoryStore()
    # MemorySaver keeps interrupted thread state in memory —
    # sufficient for the HITL demo; swap to SqliteSaver for full persistence.
    checkpointer = MemorySaver()

    client = MultiServerMCPClient(
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

    all_tools = []
    sessions = []

    for server_name in _SERVER_NAMES:
        cm = client.session(server_name)
        session = await cm.__aenter__()
        sessions.append((cm, session))
        tools = await load_mcp_tools(session)
        all_tools.extend(tools)

    builder = _get_builder(all_tools, store)
    graph = builder.compile(checkpointer=checkpointer)

    _app_state["graph"] = graph
    _app_state["store"] = store
    _app_state["sessions"] = sessions

    yield

    # Cleanup MCP sessions
    for cm, session in reversed(sessions):
        try:
            await cm.__aexit__(None, None, None)
        except Exception:
            pass


app = FastAPI(title="Task-05 Agent API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_interrupt(graph, config) -> PendingApproval | None:
    """Synchronously check graph state for an active interrupt."""
    try:
        state = graph.get_state(config)
        if state and state.tasks:
            for task in state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    data = task.interrupts[0].value
                    return PendingApproval(
                        tool_name=data.get("tool_name", ""),
                        tool_args=data.get("tool_args", {}),
                    )
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    graph = _app_state["graph"]
    thread_id = req.thread_id or str(uuid.uuid4())

    config = RunnableConfig(
        metadata={"mode": "hitl", "user_id": req.user_id, "thread_id": thread_id},
        run_name="task05-api-chat",
        configurable={"thread_id": thread_id},
    )

    initial_state = {
        "messages": [HumanMessage(content=req.query)],
        "user_id": req.user_id,
        "memory_context": "",
        "pending_interrupt": None,
        "run_metadata": {"mode": "interactive", "user_id": req.user_id},
        "tool_results": [],
    }

    try:
        result = await graph.ainvoke(initial_state, config=config)
        answer = result["messages"][-1].content
        return ChatResponse(status="complete", answer=answer, thread_id=thread_id)
    except Exception:
        # Check for interrupt
        pending = _extract_interrupt(graph, config)
        if pending:
            return ChatResponse(
                status="interrupted",
                thread_id=thread_id,
                pending_approval=pending,
            )
        raise


@app.post("/resume", response_model=ResumeResponse)
async def resume(req: ResumeRequest):
    graph = _app_state["graph"]
    thread_id = req.thread_id

    config = RunnableConfig(
        metadata={"mode": "hitl_resumed", "thread_id": thread_id},
        run_name="task05-api-resume",
        configurable={"thread_id": thread_id},
    )

    try:
        result = await graph.ainvoke(
            Command(resume={"approved": req.approved}),
            config=config,
        )
        answer = result["messages"][-1].content
        return ResumeResponse(status="complete", answer=answer, thread_id=thread_id)
    except Exception:
        pending = _extract_interrupt(graph, config)
        if pending:
            return ResumeResponse(
                status="interrupted",
                thread_id=thread_id,
                pending_approval=pending,
            )
        raise


@app.get("/memories/{user_id}", response_model=MemoriesResponse)
async def get_memories(user_id: str):
    store = _app_state["store"]
    preferences = await load_memories(store, ("preferences", user_id))
    episodes = await load_memories(store, ("episodes", user_id))
    return MemoriesResponse(user_id=user_id, preferences=preferences, episodes=episodes)


@app.delete("/memories/{user_id}")
async def delete_memories(user_id: str, x_api_key: str = Header(default="")):
    import os
    expected = os.getenv("MEMORY_DELETE_API_KEY", "changeme")
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

    store = _app_state["store"]
    for namespace in [("preferences", user_id), ("episodes", user_id), ("facts", user_id)]:
        items = await store.asearch(namespace)
        for item in items:
            await store.adelete(namespace, item.key)

    return {"deleted": True, "user_id": user_id}