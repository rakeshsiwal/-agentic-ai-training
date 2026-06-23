"""
FastAPI REST API with Server-Sent Events streaming.

Endpoints:
  POST /research          — start pipeline, stream SSE events
  GET  /research/{tid}    — poll latest state for a thread_id
"""
from __future__ import annotations

import json
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import PipelineStatusResponse, ResearchRequest
from graph import graph

app = FastAPI(title="Plan-Execute Research API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _build_initial_state(body: ResearchRequest) -> dict:
    return {
        "messages": [],
        "original_query": body.query,
        "sub_tasks": [],
        "retrieved_docs": [],
        "aggregated_context": [],
        "draft": "",
        "final_answer": "",
        "evaluation": {},
        "iteration": 0,
        "max_iterations": body.max_iterations,
        "current_sub_task": {},
    }


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


# ── POST /research ────────────────────────────────────────────────────────────

@app.post("/research")
async def research(body: ResearchRequest):
    """Stream pipeline progress as Server-Sent Events."""

    initial_state = _build_initial_state(body)
    config = {"configurable": {"thread_id": body.thread_id}}

    async def event_stream():
        try:
            async for event in graph.astream_events(
                initial_state, config=config, version="v2"
            ):
                event_name = event.get("event", "")
                node_name = event.get("name", "")
                data = event.get("data", {})

                # ── Planner finished ──
                if event_name == "on_chain_end" and node_name == "planner_node":
                    output = data.get("output", {})
                    sub_tasks = output.get("sub_tasks", [])
                    yield _sse({"type": "planner", "sub_tasks": sub_tasks})

                # ── Researcher finished ──
                elif event_name == "on_chain_end" and node_name == "researcher_node":
                    output = data.get("output", {})
                    chunks = output.get("retrieved_docs", [])
                    # Determine sub_task_id from the first chunk tag if available
                    sub_task_id = "unknown"
                    if chunks:
                        tag = chunks[0].split("]")[0].lstrip("[")
                        sub_task_id = tag
                    yield _sse({
                        "type": "researcher",
                        "sub_task_id": sub_task_id,
                        "chunks_found": len(chunks),
                    })

                # ── Aggregator finished ──
                elif event_name == "on_chain_end" and node_name == "aggregator_node":
                    output = data.get("output", {})
                    agg = output.get("aggregated_context", [])
                    # raw count comes from state input; approximate as len(agg) + dedup delta
                    yield _sse({
                        "type": "aggregator",
                        "after_dedup": len(agg),
                    })

                # ── Writer finished ──
                elif event_name == "on_chain_end" and node_name == "writer_node":
                    output = data.get("output", {})
                    draft = output.get("draft", "")
                    yield _sse({"type": "writer", "draft_length": len(draft)})

                # ── Evaluator finished ──
                elif event_name == "on_chain_end" and node_name == "evaluator_node":
                    output = data.get("output", {})
                    ev = output.get("evaluation", {})
                    iteration = output.get("iteration", 0)
                    final_answer = output.get("final_answer", "")

                    if final_answer:
                        yield _sse({
                            "type": "final",
                            "answer": final_answer,
                            "iteration": iteration,
                        })
                    else:
                        yield _sse({
                            "type": "evaluator",
                            "faithfulness": ev.get("faithfulness"),
                            "relevance": ev.get("relevance"),
                            "feedback": ev.get("feedback"),
                            "iteration": iteration,
                        })

            # Emit the final answer from graph state if not already emitted
            final_state = graph.get_state(config)
            if final_state and final_state.values:
                fa = final_state.values.get("final_answer") or final_state.values.get("draft", "")
                if fa:
                    yield _sse({"type": "done", "answer": fa})

        except Exception as exc:
            yield _sse({"type": "error", "message": str(exc)})
            traceback.print_exc()

    return StreamingResponse(event_stream(), media_type="text/event-stream")


# ── GET /research/{thread_id} ─────────────────────────────────────────────────

@app.get("/research/{thread_id}", response_model=PipelineStatusResponse)
async def get_research_state(thread_id: str):
    """Return the latest persisted state for a given thread_id."""
    config = {"configurable": {"thread_id": thread_id}}
    try:
        snapshot = graph.get_state(config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    if snapshot is None or not snapshot.values:
        raise HTTPException(status_code=404, detail="thread_id not found")

    values = snapshot.values
    final_answer = values.get("final_answer") or values.get("draft", "")
    status = "complete" if final_answer else "running"

    return PipelineStatusResponse(
        status=status,
        thread_id=thread_id,
        state={
            "original_query": values.get("original_query"),
            "iteration": values.get("iteration"),
            "evaluation": values.get("evaluation"),
            "final_answer": final_answer,
            "sub_tasks": values.get("sub_tasks"),
        },
    )


# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}