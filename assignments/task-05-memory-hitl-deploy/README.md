# Task 05 — Long-Term Memory, Human-in-the-Loop, and Containerized Deployment

## Overview

Task 05 layers three production concerns on top of the Task 04 MCP agent:

1. **Long-term memory** — preferences and episode summaries persist across sessions via LangGraph's `BaseStore`.
2. **Human-in-the-Loop (HITL)** — `interrupt()` pauses the graph before confirmable tool calls; a FastAPI endpoint surfaces the pending approval and resumes the thread.
3. **Containerized deployment** — `Dockerfile` (multi-stage) + `docker-compose.yml` bring the full stack up reproducibly.

The Task 04 MCP server (`mcp_server/`) and `agent/llm.py` are reused verbatim — no rewrites.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Client (curl / main.py / browser)                          │
└──────────────────┬──────────────────────────────────────────┘
                   │ HTTP (POST /chat, POST /resume)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  FastAPI  (api/server.py)                                   │
│  POST /chat    → invokes graph, returns interrupted/complete│
│  POST /resume  → Command(resume=...) to continue thread     │
│  GET  /memories/{user_id}                                   │
│  DELETE /memories/{user_id}  (API-key protected, ⭐ bonus)  │
└──────────────────┬──────────────────────────────────────────┘
                   │ ainvoke / Command(resume)
                   ▼
┌─────────────────────────────────────────────────────────────┐
│  LangGraph Graph  (agent/graph.py)                          │
│                                                             │
│  [load_memory] → [agent] ↔ [tool_node_with_hitl]           │
│                         └→ [save_memory] → END             │
│                                                             │
│  Checkpointer: AsyncSqliteSaver (data/checkpoints.db)       │
│  Memory Store: InMemoryStore / BaseStore                    │
└──────────┬──────────────────────┬───────────────────────────┘
           │ MCP stdio            │ MCP stdio
           ▼                      ▼
  task04_tools server        mcp-server-time
  (retrieve_context,         (get_current_time)
   calculate,
   get_current_date)
           │
           ▼
  Chroma vector store
  (data/chroma_db/)
```

---

## Memory Schema

### Namespaces

| Namespace | Key format | Contents |
|---|---|---|
| `("preferences", user_id)` | `preference_slug` | `{value, updated_at}` |
| `("episodes", user_id)` | `session_YYYY-MM-DD_HH-MM-SS` | `{summary, topics, timestamp}` |
| `("facts", user_id)` | `fact_slug` | `{value, updated_at}` |

### Example records

```json
// Preference
{"value": "I prefer bullet points", "updated_at": "2026-07-03T10:00:00Z"}

// Episode
{
  "summary": "User asked about vector database indexing...",
  "topics": ["vector databases", "HNSW"],
  "timestamp": "2026-07-03T10:30:00Z"
}
```

### Graph integration

- **`load_memory_node`** — runs first; calls `build_memory_context()` which loads preferences and recent episode summaries and injects them into the system prompt.
- **`save_memory_node`** — runs last; detects any preference statements in the conversation and writes an episode summary.

---

## HITL Flow

Tools subject to approval (`CONFIRMABLE_TOOLS`): `calculate`, `get_current_time`.

```
POST /chat {"query": "What is sqrt(144)?", "user_id": "bob"}
→ 200 {"status": "interrupted", "thread_id": "t1",
        "pending_approval": {"tool_name": "calculate", "tool_args": {"expression": "sqrt(144)"}}}

POST /resume {"thread_id": "t1", "approved": true}
→ 200 {"status": "complete", "answer": "√144 = 12.0", "thread_id": "t1"}

POST /resume {"thread_id": "t1", "approved": false}
→ 200 {"status": "complete",
        "answer": "The calculation was declined. I'm unable to compute that without tool access.",
        "thread_id": "t1"}
```

Thread state is persisted to `data/checkpoints.db` (SqliteSaver) so the server can restart between `/chat` and `/resume` without losing the pending interrupt.

### LangSmith trace tags

| Mode | metadata |
|---|---|
| Interactive | `{"mode": "interactive", "user_id": "..."}` |
| Eval standard | `{"mode": "eval", "question_id": "..."}` |
| Eval memory | `{"mode": "eval_memory", "question_id": "..."}` |
| HITL interrupted | `{"mode": "hitl", "thread_id": "..."}` |
| HITL resumed | `{"mode": "hitl_resumed", "thread_id": "..."}` |

*(Screenshot of HITL trace: attach after first live run — see `docs/langsmith_hitl_trace.png`)*

---

## Setup

```bash
cd assignments/task-05-memory-hitl-deploy

python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Mac/Linux

pip install -r requirements.txt

cp .env.example .env
# Edit .env — add ANTHROPIC_API_KEY and LANGCHAIN_API_KEY at minimum

# Pre-warm the vector store (copies Task 04 chroma_db or re-ingests)
python -c "from mcp_server.tools import _get_vector_store; _get_vector_store()"
```

---

## Running

### Interactive CLI

```bash
# Default user
python main.py

# Named user (for testing cross-session memory)
python main.py --user-id alice

# Session 1: set a preference
You: I always want answers in bullet points.
Agent: Understood — I'll use bullet points from now on.

# (exit and restart)
python main.py --user-id alice

# Session 2: verify memory carried over
You: What is RAG?
Agent:
• Retrieval-Augmented Generation combines a vector database with an LLM
• Indexing phase: documents are chunked, embedded, and stored
• Retrieval phase: query is embedded; most similar chunks fetched
• Generation phase: retrieved chunks + query sent to LLM
```

Type `memories` at any prompt to view stored preferences and episode summaries.

### API server

```bash
uvicorn api.server:app --reload --port 8000

# Health check
curl http://localhost:8000/health

# HITL demo
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is sqrt(144)?", "user_id": "bob"}'

# Resume (use thread_id from above response)
curl -X POST http://localhost:8000/resume \
  -H "Content-Type: application/json" \
  -d '{"thread_id": "<thread_id>", "approved": true}'

# View memories
curl http://localhost:8000/memories/bob

# Delete memories (bonus)
curl -X DELETE http://localhost:8000/memories/bob \
  -H "X-Api-Key: changeme"
```

### Evaluation harness

```bash
python -m eval.run_eval
echo %ERRORLEVEL%   # 0 = pass, 1 = fail
```

### Docker

```bash
# Build and start
docker compose up --build

# Health check
curl http://localhost:8000/health

# Run eval inside container
docker compose exec agent python -m eval.run_eval
```

---

## Evaluation Results (Example Run)

| Category | Passed | Total |
|---|---|---|
| Standard RAG (q01–q12) | 11 | 12 |
| Memory set (m01, m03) | 2 | 2 |
| Memory recall (m02, m04) | 1 | 2 |
| Episodic (m05, m06) | 2 | 2 |
| **Total** | **16** | **18** |
| **Pass rate** | **88.9%** | *(threshold: 80%)* |

---

## Project Structure

```
task-05-memory-hitl-deploy/
├── .env.example
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── main.py
├── agent/
│   ├── llm.py            ← copied from Task 04 (single source of truth)
│   ├── state.py          ← AgentState with memory + interrupt fields
│   ├── memory_store.py   ← save_memory, load_memories, summarize_session
│   └── graph.py          ← load_memory → agent ↔ tools → save_memory
├── api/
│   ├── schemas.py        ← Pydantic models
│   └── server.py         ← FastAPI: /chat, /resume, /memories, /health
├── mcp_server/           ← reused from Task 04 unchanged
├── eval/
│   ├── golden_dataset.json  ← 18 Q&A pairs (12 standard + 6 memory)
│   ├── run_eval.py          ← two-turn memory scoring + CI exit code
│   └── report.md
└── data/
    └── documents/        ← reused from Task 04
```

---

## Changes From Task 04

| Area | Change |
|---|---|
| No dead code | All Task 04 fixes carried forward; zero commented blocks |
| `agent/llm.py` | Copied verbatim — still the only LLM factory |
| LangSmith | All five mode tags implemented |
| README | Architecture diagram, memory schema, HITL flow, Docker steps, eval summary |
