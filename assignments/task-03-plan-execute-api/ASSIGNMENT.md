# Assignment Task 03 — Plan-and-Execute Agent with Parallel Workers, Self-Reflection, and REST API

**Issued:** June 22, 2026
**Due:** June 29, 2026
**Difficulty:** Advanced → Expert
**Branch naming:** `feature/task-03-plan-execute-api`
**Prerequisite:** Task 02 completed and reviewed

---

## 🎯 Objective

Evolve your multi-agent system into a **production-grade agentic pipeline** by adding three capabilities that real deployed systems require:

1. **Plan-and-Execute pattern** — A Planner agent decomposes a complex question into 2–4 focused sub-tasks, then Researcher workers execute those sub-tasks **in parallel** using LangGraph's `Send()` API.
2. **Self-Reflection loop** — An Evaluator agent scores the Writer's draft for faithfulness and relevance. If either score is below threshold the pipeline feeds the feedback back to the Planner and tries again (up to 3 iterations).
3. **REST API with SSE streaming** — The entire pipeline is served via **FastAPI**. A client POSTs a query and receives a real-time **Server-Sent Events** stream of pipeline progress, so the frontend never has to poll.

You will also introduce **`SqliteSaver`** so that pipeline state survives process restarts, enabling true async workflows where a client can disconnect and reconnect to check progress.

---

## 🧠 What You Will Learn

- **`Send()` API** — LangGraph's mechanism for dynamic, data-driven fan-out: spawning N parallel node executions at runtime based on data in state.
- **Fan-in aggregation** — collecting results from N parallel branches back into a single node.
- **Self-reflection / critique loops** — using a second LLM call to score output quality and trigger re-planning.
- **`SqliteSaver`** — persisting graph checkpoints to a SQLite database so sessions survive restarts.
- **FastAPI + SSE** — wrapping an async LangGraph pipeline in an HTTP server that streams events to clients.
- **`graph.astream_events()`** — the async streaming API for real-time event emission.
- **Subgraph reuse** — importing and wiring the RAG retriever from Task 02 as a module, not rewriting it.

---

## 🗂️ Folder Structure (Expected Output)

```
assignments/task-03-plan-execute-api/
│
├── ASSIGNMENT.md                  ← this file (do not modify)
├── README.md                      ← your notes: architecture, setup, how to run
├── requirements.txt
├── .env.example
│
├── api/
│   ├── __init__.py
│   ├── server.py                  ← FastAPI app + SSE streaming endpoint
│   └── schemas.py                 ← Pydantic request/response models
│
├── agents/
│   ├── __init__.py
│   ├── planner.py                 ← Decomposes query into SubTask list
│   ├── researcher.py              ← RAG researcher (one instance per sub-task)
│   ├── aggregator.py              ← Merges parallel retrieved docs into one context
│   ├── writer.py                  ← Drafts the final cited answer
│   └── evaluator.py               ← Scores draft; triggers re-plan if score < 3
│
├── rag/
│   ├── __init__.py
│   ├── ingest.py                  ← Reuse / extend from Task 02
│   └── retriever.py               ← Reuse from Task 02
│
├── data/
│   └── documents/                 ← At least 3 .txt/.md files (expand from Task 02)
│
├── state.py                       ← PlanExecuteState TypedDict
├── graph.py                       ← Full graph with Send(), parallel branches, reflection
└── main.py                        ← CLI entry point for local testing (no API server)
```

---

## ⚙️ Technical Requirements

### 1. Shared Pipeline State

```python
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages


class SubTask(TypedDict):
    id: str          # e.g. "sub_0", "sub_1"
    question: str    # the focused sub-question


class EvaluationResult(TypedDict):
    faithfulness: int  # 1–5: are claims supported by retrieved context?
    relevance: int     # 1–5: does the answer address the original query?
    feedback: str      # one sentence of actionable critique


class PlanExecuteState(TypedDict):
    messages:           Annotated[list, add_messages]
    original_query:     str               # user's original question (never modified)
    sub_tasks:          list[SubTask]     # planner's decomposition
    retrieved_docs:     Annotated[list, operator.add]  # fan-in accumulator
    aggregated_context: list[str]         # deduplicated chunks ready for writer
    draft:              str               # writer's current draft
    final_answer:       str               # approved final output
    evaluation:         EvaluationResult  # latest evaluator scores
    iteration:          int               # reflection loop counter (starts at 0)
    max_iterations:     int               # hard cap (default 3)
```

> **Why `Annotated[list, operator.add]`?** When parallel `researcher_node` branches each append to `retrieved_docs`, LangGraph needs a reducer that concatenates rather than overwrites. `operator.add` does exactly this.

---

### 2. Agent Nodes

#### `agents/planner.py` — Planner Node

Uses structured output to decompose the query:

```python
from pydantic import BaseModel

class Plan(BaseModel):
    sub_tasks: list[SubTask]   # 2–4 items
    reasoning: str

def planner_node(state: PlanExecuteState) -> dict:
    ...
```

- On the **first iteration** (`state["iteration"] == 0`): decompose `original_query`.
- On **re-plan iterations**: incorporate `state["evaluation"]["feedback"]` into the new decomposition — the planner should understand *why* the previous draft failed and ask more targeted questions.
- Always reset `retrieved_docs`, `aggregated_context`, and `draft` to empty so the pipeline re-executes cleanly.

---

#### `agents/researcher.py` — Researcher Node (parallel worker)

Each `researcher_node` receives **one `SubTask`** via `Send()` and writes its results back to the shared `retrieved_docs` list.

```python
def researcher_node(state: PlanExecuteState) -> dict:
    # state will contain the full PlanExecuteState plus the current sub_task
    task: SubTask = state["current_sub_task"]   # injected by Send()
    chunks = retrieve(task["question"], k=4)
    return {"retrieved_docs": chunks}   # reducer appends, not overwrites
```

---

#### `agents/aggregator.py` — Aggregator Node

- Receives all `retrieved_docs` from the fan-in.
- Deduplicates identical chunks (exact match or similarity threshold ≥ 0.95).
- Stores the clean list in `aggregated_context`.
- Optionally re-ranks by relevance to `original_query`.

---

#### `agents/writer.py` — Writer Node

- Identical in spirit to Task 02's writer.
- Uses `aggregated_context` (not `retrieved_docs` directly).
- Must cite `[Chunk N]` inline.
- On re-plan iterations, also receives the previous `evaluation["feedback"]` in its prompt so it can improve the draft's focus.

---

#### `agents/evaluator.py` — Evaluator Node

Uses structured output to score the draft:

```python
class EvaluationResult(BaseModel):
    faithfulness: int = Field(..., ge=1, le=5)
    relevance:    int = Field(..., ge=1, le=5)
    feedback:     str

def evaluator_node(state: PlanExecuteState) -> dict:
    ...
    return {"evaluation": result.model_dump(), "iteration": state["iteration"] + 1}
```

The evaluation prompt must explain the scoring rubric clearly:
- **Faithfulness 1**: Answer invents facts not in the retrieved chunks.
- **Faithfulness 5**: Every claim is directly supported by a cited chunk.
- **Relevance 1**: Answer is off-topic or misses the user's question.
- **Relevance 5**: Answer fully and directly addresses `original_query`.

---

### 3. Parallel Fan-Out with `Send()`

This is the core new concept of Task 03. After `planner_node` runs, instead of routing to a single `researcher_node`, you dynamically spawn **one researcher per sub-task**:

```python
from langgraph.types import Send

def dispatch_researchers(state: PlanExecuteState) -> list[Send]:
    """Conditional edge: fan-out one researcher per sub-task."""
    return [
        Send("researcher_node", {**state, "current_sub_task": task})
        for task in state["sub_tasks"]
    ]
```

Wire it like this:

```python
builder.add_conditional_edges(
    "planner_node",
    dispatch_researchers,
)
```

LangGraph runs all returned `Send` objects in parallel. Each researcher writes its chunks to `retrieved_docs`; the `operator.add` reducer concatenates them automatically. All branches converge at `aggregator_node`.

---

### 4. Self-Reflection Routing

After `evaluator_node`, use a conditional edge to decide whether to stop or re-plan:

```python
def route_after_evaluation(state: PlanExecuteState) -> str:
    ev = state.get("evaluation", {})
    if state["iteration"] >= state["max_iterations"]:
        return "end"   # hard stop — return best draft so far
    if ev.get("faithfulness", 0) < 3 or ev.get("relevance", 0) < 3:
        return "planner_node"  # re-plan with feedback
    return "end"   # quality is acceptable
```

---

### 5. Graph Topology

```
[START]
   ↓
planner_node
   ↓  (dispatch_researchers — Send() fan-out)
[researcher_node × N]  ← run in parallel
   ↓  (all converge here)
aggregator_node
   ↓
writer_node
   ↓
evaluator_node
   ↓  (route_after_evaluation)
┌──────────────────────────────┐
│  score OK or max_iter hit    │→  [END]
│  score too low               │→  planner_node  (loop)
└──────────────────────────────┘
```

Compile with **`SqliteSaver`**:

```python
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)
app = builder.compile(checkpointer=memory)
```

---

### 6. FastAPI REST API

#### `api/server.py`

Expose two endpoints:

```
POST /research
     Body:  { "query": "...", "thread_id": "..." }
     Returns: text/event-stream

GET  /research/{thread_id}
     Returns: { "status": "running|complete|error", "state": {...} }
```

The SSE stream should emit one JSON event per significant pipeline step:

```
data: {"type": "planner",    "sub_tasks": [...]}
data: {"type": "researcher", "sub_task_id": "sub_0", "chunks_found": 4}
data: {"type": "researcher", "sub_task_id": "sub_1", "chunks_found": 3}
data: {"type": "aggregator", "total_chunks": 6, "after_dedup": 5}
data: {"type": "writer",     "draft_length": 1402}
data: {"type": "evaluator",  "faithfulness": 4, "relevance": 3, "iteration": 1}
data: {"type": "final",      "answer": "..."}
```

Implement streaming using `graph.astream_events()`:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()

@app.post("/research")
async def research(body: ResearchRequest):
    async def event_stream():
        async for event in graph.astream_events(initial_state, config, version="v2"):
            if event["event"] == "on_chain_end":
                yield f"data: {json.dumps(event['data'])}\n\n"
    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

---

### 7. Sample Documents

Add at least **3 documents** to `data/documents/` (you may reuse the two from Task 02 and add one more). Topics to consider:

- Vector databases and embedding search
- LangGraph internals and the `Send()` API
- Prompt engineering techniques
- Constitutional AI and alignment approaches

Each file: 200–500 words.

---

## ✅ Acceptance Criteria

Before submitting your Pull Request:

- [ ] `planner_node` correctly produces 2–4 sub-tasks for any query
- [ ] `Send()` fan-out confirmed: logs show researcher nodes running in parallel (different `sub_task_id` values appear before any completes)
- [ ] `aggregator_node` deduplicates chunks — identical chunks do not appear twice in `aggregated_context`
- [ ] `evaluator_node` scores the draft and routes back to `planner_node` at least once during a demo run where you intentionally ask a question with thin coverage
- [ ] Pipeline terminates after `max_iterations` even if scores remain low
- [ ] `SqliteSaver` used — killing the process and restarting still shows the correct state for an existing `thread_id`
- [ ] `POST /research` returns SSE stream with all 5 event types (`planner`, `researcher`, `aggregator`, `writer`, `evaluator`/`final`)
- [ ] `GET /research/{thread_id}` returns the latest pipeline state as JSON
- [ ] `.env.example` present, no real keys committed
- [ ] `requirements.txt` complete and installable
- [ ] `README.md` includes architecture diagram, setup steps, sample curl session, and a screenshot or log of a full pipeline run with at least one reflection loop

---

## 🌟 Bonus Challenges (optional)

| Stars | Bonus | Description |
|---|---|---|
| ⭐ | Query Clarifier | Add a `clarifier_node` before the planner that detects ambiguous queries and asks the user one targeted clarifying question using `interrupt()` |
| ⭐⭐ | Re-ranking | After aggregation, use a cross-encoder (`sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2`) to re-rank chunks by relevance to `original_query` before passing to the writer |
| ⭐⭐ | Async Researcher | Make `researcher_node` async (`async def`) and use `asyncio.gather` inside to query both the vector store and DuckDuckGo web search simultaneously |
| ⭐⭐⭐ | React UI | Build a minimal React/plain-HTML frontend that connects to the SSE endpoint and renders each pipeline step as it arrives in real time |
| ⭐⭐⭐ | LangSmith Tracing | Add `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` and confirm traces appear in LangSmith with custom run names per pipeline stage |
| ⭐⭐⭐⭐ | Docker | Containerize the FastAPI server with Docker and write a `docker-compose.yml` that starts the server and mounts the `data/` volume |

---

## 📦 Dependencies

```txt
# Core
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0

# LLM providers (at least one)
langchain-anthropic>=0.2.0
langchain-openai>=0.2.0
langchain-google-genai>=2.0.0
langchain-groq>=0.2.0

# RAG (reuse from Task 02)
langchain-chroma>=0.1.0
chromadb>=0.5.0
langchain-community>=0.3.0
sentence-transformers>=3.0.0

# Persistence
langgraph-checkpoint-sqlite>=0.1.0   # SqliteSaver

# API
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
httpx>=0.27.0                         # for async HTTP in researcher bonus

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

> **Tip — `langgraph-checkpoint-sqlite`:** If this package is not yet on PyPI under that name, import `SqliteSaver` from `langgraph.checkpoint.sqlite`. Check the LangGraph changelog for the current package name.

---

## 📤 Submission

1. `git checkout -b feature/task-03-plan-execute-api`
2. Build inside `assignments/task-03-plan-execute-api/`.
3. Run the CLI (`python main.py`) end-to-end and confirm at least one reflection loop occurs in the logs.
4. Start the API (`uvicorn api.server:app --reload`) and run `curl` to hit the SSE endpoint.
5. Include both logs in `README.md`.
6. Open a Pull Request to `main` — tag the reviewer.

---

## 📚 Resources

- [LangGraph — Send() API (parallel fan-out)](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [LangGraph — Streaming with astream_events](https://langchain-ai.github.io/langgraph/how-tos/streaming-events-from-within-tools/)
- [LangGraph — SqliteSaver persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [LangGraph — Subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [FastAPI — StreamingResponse + SSE](https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse)
- [LangChain — Structured Output](https://python.langchain.com/docs/how_to/structured_output/)
- [Sentence Transformers — Cross Encoders (bonus)](https://www.sbert.net/docs/cross_encoder/usage/usage.html)

---

## 🗺️ Suggested Day-by-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Set up folder, expand sample docs, confirm Task 02's RAG retriever still works when imported here |
| Day 2 | Build `planner_node` with structured output; test decomposition in isolation |
| Day 3 | Implement `researcher_node` and wire `Send()` fan-out; verify parallel execution in logs |
| Day 4 | Build `aggregator_node` and `writer_node`; run the pipeline CLI end-to-end |
| Day 5 | Add `evaluator_node` and the reflection routing; test with a low-coverage query to trigger re-planning |
| Day 6 | Add `SqliteSaver`; build FastAPI server with SSE streaming; test with `curl` |
| Day 7 | Polish: README, error handling, attempt a bonus |

---

## 🔑 Key Concepts Cheat Sheet

| Concept | Where it appears | Why it matters |
|---|---|---|
| `Send(node, state)` | `dispatch_researchers()` | Spawns N parallel node executions at runtime |
| `Annotated[list, operator.add]` | `retrieved_docs` in state | Reducer that concatenates fan-in results |
| `graph.astream_events()` | `api/server.py` | Async generator yielding fine-grained events for SSE |
| `SqliteSaver` | `graph.py` | Persists checkpoint state to disk across restarts |
| `iteration` counter | `PlanExecuteState` | Prevents infinite reflection loops |
| Evaluator structured output | `agents/evaluator.py` | LLM self-critique with typed, actionable scores |

---

*Task 03 is a significant jump. The `Send()` API and async SSE streaming are the two concepts that will take the most time to understand — read the LangGraph map-reduce how-to guide thoroughly before writing code. The goal is not a perfect system; it is to understand why production agentic pipelines need parallelism, reflection, and an API layer.*
