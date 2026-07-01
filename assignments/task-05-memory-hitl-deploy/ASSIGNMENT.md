# Assignment Task 05 — Long-Term Memory, Human-in-the-Loop, and Containerized Deployment

**Issued:** July 1, 2026
**Due:** July 8, 2026
**Difficulty:** Expert+
**Branch naming:** `feature/task-05-memory-hitl-deploy`
**Prerequisite:** Task 04 completed and reviewed

---

## 🎯 Objective

In every previous task your agent forgot everything the moment the process ended. It also had no mechanism to pause and ask a human "are you sure?" before taking a consequential action. And running it on another machine meant fighting `.env` drift and missing dependencies.

Task 05 fixes all three gaps by adding:

1. **Long-term memory** — the agent remembers user preferences, key facts, and conversation summaries *across* sessions using LangGraph's `BaseStore` with a SQLite-backed persistent store.
2. **Human-in-the-Loop (HITL)** — the agent uses `interrupt()` to pause the graph before high-stakes tool calls, surfaces the pending approval over a FastAPI endpoint, and resumes once the human responds.
3. **Containerized deployment** — a `Dockerfile` and `docker-compose.yml` that bring up the entire stack (agent API + ChromaDB) reproducibly so "it works on my machine" is no longer a valid excuse.

You will **reuse the MCP server and `agent/llm.py` factory from Task 04 verbatim** — do not rewrite them. Task 05 is about layering production concerns on top of a working system, not rebuilding the system.

---

## 📌 Carrying Forward Feedback From Previous Reviews

| Past issue | What to do differently this time |
|---|---|
| Dead/commented-out code accumulating across tasks | Your Task 04 submission was clean — keep that standard. If you change an approach mid-task, delete the old branch entirely from the code. |
| `README.md` missing in Task 03, present but thin in Task 04 | `README.md` must include: architecture diagram, memory schema, HITL flow, Docker setup steps, eval summary, and a LangSmith trace screenshot. |
| LangSmith tracing was required in Task 04 and implemented — good | Tag HITL-interrupted runs with `{"mode": "hitl"}` in addition to the existing `eval` / `interactive` tags. |
| Multi-provider `get_llm()` from Task 04 — correct | Import it from Task 04's `agent/llm.py` via a relative import or copy the file. Do **not** write a third copy. |

---

## 🧠 What You Will Learn

- **LangGraph `BaseStore`** — the memory primitive that lets nodes read and write named key/value records that persist outside the graph's own `State`.
- **Episodic memory** — storing a summary of each conversation at the end, then retrieving relevant past summaries at the start of the next one.
- **`interrupt()`** — LangGraph's mechanism to pause graph execution at a node boundary, return control to the caller, and resume later with human-provided input injected into state.
- **Streaming HITL over HTTP** — combining Task 03's FastAPI/SSE knowledge with `interrupt()` to build an approval workflow endpoint.
- **Docker multi-stage builds** — keeping your image small while installing all dependencies cleanly.
- **`docker-compose`** — declaring the full stack as code, including environment variable injection and service health checks.

---

## 🗂️ Folder Structure (Expected Output)

```
assignments/task-05-memory-hitl-deploy/
│
├── ASSIGNMENT.md                  ← this file (do not modify)
├── README.md                      ← architecture, memory schema, HITL flow, Docker setup, eval summary
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
│
├── agent/
│   ├── __init__.py
│   ├── llm.py                     ← copied/symlinked from Task 04 — one source of truth
│   ├── state.py                   ← AgentState extended with memory + interrupt fields
│   ├── memory_store.py            ← helpers: save_memory(), load_memories(), summarize_session()
│   └── graph.py                   ← graph with memory nodes + interrupt() for HITL
│
├── api/
│   ├── __init__.py
│   ├── server.py                  ← FastAPI: POST /chat, POST /resume, GET /memories/{user_id}
│   └── schemas.py                 ← Pydantic models for requests, responses, and HITL payloads
│
├── mcp_server/                    ← reuse from Task 04, do not modify
│   ├── __init__.py
│   ├── server.py
│   └── tools.py
│
├── eval/
│   ├── __init__.py
│   ├── golden_dataset.json        ← Task 04 dataset + 6 new memory-specific questions (≥ 18 total)
│   ├── run_eval.py                ← extended to also test memory retention across two-turn exchanges
│   └── report.md                  ← generated output (commit one example run)
│
├── data/
│   └── documents/                 ← reuse from Task 04
│
└── main.py                        ← CLI entry point (interactive, no API server needed)
```

---

## ⚙️ Technical Requirements

### 1. Long-Term Memory (`agent/memory_store.py` + `agent/graph.py`)

Use LangGraph's `InMemoryStore` for development and `langgraph-checkpoint-sqlite` for persistence. The memory system must support three memory types:

#### a) User Preferences
Stored when the user states an explicit preference ("I prefer concise answers", "Always cite sources").

```python
# Stored under namespace ("preferences", user_id)
{
  "key": "response_style",
  "value": "concise",
  "updated_at": "2026-07-03T10:00:00Z"
}
```

#### b) Episodic Memory (Session Summaries)
At the end of every session, the agent writes a one-paragraph summary of what was discussed and what was learned about the user's interests.

```python
# Stored under namespace ("episodes", user_id)
{
  "key": "session_2026-07-03",
  "summary": "User asked about vector database indexing strategies...",
  "topics": ["vector databases", "HNSW", "IVF"],
  "timestamp": "2026-07-03T10:30:00Z"
}
```

#### c) Semantic Memory Retrieval
At the start of each session, the agent searches for past episodes relevant to the current query and injects them as context into the system prompt.

```python
# agent/memory_store.py
async def load_relevant_memories(store: BaseStore, user_id: str, query: str, k: int = 3) -> list[dict]:
    """Return the k most relevant past episode summaries for the given query."""
    ...

async def save_memory(store: BaseStore, namespace: tuple, key: str, value: dict) -> None:
    """Write a memory record to the store."""
    ...

async def summarize_session(messages: list, llm) -> str:
    """Summarize a conversation into a one-paragraph episode summary."""
    ...
```

#### Graph Integration

Add two new nodes to the existing `agent_node ↔ tool_node` loop:

```
[load_memory_node] → [agent_node] ↔ [tool_node (with interrupt)]
                                  └→ [save_memory_node] → END
```

- `load_memory_node`: runs first; loads relevant episodes and user preferences, injects them into the system prompt.
- `save_memory_node`: runs last; generates an episode summary and writes it to the store.

---

### 2. Human-in-the-Loop (`interrupt()`)

Use LangGraph's `interrupt()` to pause the graph **before any tool call that modifies state outside the agent** — specifically, before any tool call that writes to disk or calls an external API (as opposed to read-only retrieval calls).

For this task, apply `interrupt()` before every `calculate` and `get_current_time` tool call (treat them as "confirmable" tools to practice the pattern), while `retrieve_context` runs without interruption.

#### How `interrupt()` works in this context

```python
from langgraph.types import interrupt

def tool_node_with_hitl(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]
    for tool_call in last_message.tool_calls:
        if tool_call["name"] in CONFIRMABLE_TOOLS:
            # Pause and surface to human
            human_response = interrupt({
                "action": "approve_tool_call",
                "tool_name": tool_call["name"],
                "tool_args": tool_call["args"],
            })
            if human_response.get("approved") is False:
                # Inject a rejection message and skip the call
                ...
    # proceed with approved calls
    ...
```

#### FastAPI HITL Endpoints

```python
# POST /chat
# Starts or resumes a conversation thread.
# Returns either a final answer or an "interrupted" payload with pending_approval.

# POST /resume
# Body: {"thread_id": "...", "approved": true, "modified_args": {...}}
# Resumes the interrupted graph with the human's decision.

# GET /memories/{user_id}
# Returns all stored memories for a user (preferences + recent episodes).
```

Use LangGraph's `SqliteSaver` (already introduced in Task 03) as the checkpointer so that interrupted threads survive server restarts.

#### Example HITL Flow

```
Client → POST /chat {"query": "What is 144 * 7?", "user_id": "alice"}
Server → {"status": "interrupted", "thread_id": "t1", "pending_approval": {
             "tool_name": "calculate", "tool_args": {"expression": "144 * 7"}}}

Client → POST /resume {"thread_id": "t1", "approved": true}
Server → {"status": "complete", "answer": "144 × 7 = 1008"}
```

---

### 3. Containerized Deployment (`Dockerfile` + `docker-compose.yml`)

#### `Dockerfile`

Use a Python 3.11 base image. Use a **multi-stage build**: a `builder` stage that installs dependencies into a virtual environment, and a `runtime` stage that copies only the venv + source code. This keeps the final image lean.

```dockerfile
# Stage 1 — builder
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2 — runtime
FROM python:3.11-slim AS runtime
WORKDIR /app
COPY --from=builder /install /usr/local
COPY . .
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### `docker-compose.yml`

```yaml
version: "3.9"
services:
  agent:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    volumes:
      - ./data:/app/data          # persist ChromaDB and SQLite across restarts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      retries: 3
```

The `GET /health` endpoint must return `{"status": "ok"}` — Docker health checks require it.

#### What must work after `docker compose up`

```bash
docker compose up --build
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"query": "What is prompt engineering?", "user_id": "alice"}'
```

---

### 4. Evaluation Harness (extended)

Extend Task 04's `eval/golden_dataset.json` with at least **6 new memory-specific questions** (≥ 18 total). Memory questions test a two-turn exchange:

```json
{
  "id": "m01",
  "type": "memory",
  "turn_1": "My name is Alice and I prefer concise, bullet-point answers.",
  "turn_2": "What are the main types of vector database indexes?",
  "expected_facts": [
    "IVF", "HNSW", "flat index", "bullet", "concise"
  ],
  "note": "The answer to turn_2 must reflect the preference stated in turn_1."
}
```

Update `run_eval.py` to handle `"type": "memory"` entries:
- Send `turn_1` and `turn_2` as a two-message conversation to the same thread.
- Score `turn_2`'s answer for both fact coverage **and** preference adherence.

---

### 5. LangSmith Tracing (continued from Task 04)

Tag runs with an additional metadata key:

| Mode | metadata |
|---|---|
| Interactive | `{"mode": "interactive", "user_id": "..."}` |
| Eval — standard | `{"mode": "eval", "question_id": "..."}` |
| Eval — memory | `{"mode": "eval_memory", "question_id": "..."}` |
| HITL interrupted | `{"mode": "hitl", "thread_id": "..."}` |
| HITL resumed | `{"mode": "hitl_resumed", "thread_id": "..."}` |

Include a screenshot or exported JSON trace of one HITL flow (interrupted → resumed) in `README.md`.

---

## 🚀 Example Interactions

### Memory across sessions

```
Session 1:
  User: I always want answers in bullet points.
  Agent: Got it — I'll use bullet points from now on.

Session 2 (new process, same user_id):
  User: What are the main ideas in the prompt engineering document?
  Agent: • Zero-shot prompting: asking the model directly without examples
         • Few-shot prompting: providing 2–5 examples before the question
         • Chain-of-thought: asking the model to reason step-by-step
         ...  ← bullets because it remembered the preference from Session 1
```

### HITL approval via API

```
POST /chat {"query": "What is sqrt(144)?", "user_id": "bob"}
→ {"status": "interrupted", "pending_approval": {"tool_name": "calculate", "tool_args": {"expression": "sqrt(144)"}}}

POST /resume {"thread_id": "t1", "approved": true}
→ {"status": "complete", "answer": "√144 = 12"}

POST /resume {"thread_id": "t2", "approved": false}
→ {"status": "complete", "answer": "You declined the calculation. I'm unable to compute that without tool access."}
```

---

## ✅ Acceptance Criteria

- [ ] `load_memory_node` runs at graph start and correctly injects past episodes + preferences into the system prompt
- [ ] `save_memory_node` runs at graph end and writes an episode summary to the store
- [ ] Memory persists across process restarts (verified by stopping and restarting the server, then asking a follow-up in a new session)
- [ ] `interrupt()` fires before every call to a `CONFIRMABLE_TOOL`; approval and rejection both handled correctly
- [ ] `POST /chat` returns an `"interrupted"` payload when a confirmable tool is pending
- [ ] `POST /resume` resumes the correct thread and returns the final answer
- [ ] `docker compose up --build` succeeds and `GET /health` returns 200
- [ ] `curl` interaction from inside a container reaches the running agent
- [ ] `golden_dataset.json` has ≥ 18 entries (12 from Task 04 + 6 new memory questions)
- [ ] Memory-type eval entries are handled by `run_eval.py` (two-turn scoring including preference adherence)
- [ ] LangSmith traces appear for all five tag modes
- [ ] `agent/llm.py` is the **only** LLM factory — not redefined anywhere in this task
- [ ] No dead/commented-out code
- [ ] `.env.example`, `requirements.txt`, `README.md` all present and complete

---

## 🌟 Bonus Challenges (optional)

| Stars | Bonus | Description |
|---|---|---|
| ⭐ | Memory Deletion | Add `DELETE /memories/{user_id}` endpoint to wipe a user's memories; protect it with a simple API key header |
| ⭐⭐ | Semantic Memory Search | Replace episode retrieval (currently keyword/recency-based) with a proper vector similarity search using the existing Chroma store |
| ⭐⭐ | HITL WebSocket | Replace the two-request HITL flow (POST /chat + POST /resume) with a WebSocket that pushes the interrupt event in real time and waits for the client to reply on the same connection |
| ⭐⭐⭐ | CI + Docker | Add a GitHub Actions workflow: build the Docker image, run `python -m eval.run_eval` inside it, and fail the PR if pass rate < 80% |
| ⭐⭐⭐ | Remote MCP over HTTP | Run the Task 04 MCP server over the `streamable-http` transport inside Docker (instead of stdio subprocess) and update the client config accordingly |

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

# MCP (reuse from Task 04)
mcp>=1.2.0
langchain-mcp-adapters>=0.1.0
mcp-server-time>=0.1.0

# RAG (reuse from Task 04)
langchain-chroma>=0.1.0
chromadb>=0.5.0
langchain-community>=0.3.0
sentence-transformers>=3.0.0

# Memory & Persistence
langgraph-checkpoint-sqlite>=0.1.0

# API
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
sse-starlette>=2.0.0

# Observability
langsmith>=0.1.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

---

## 📤 Submission

1. `git checkout -b feature/task-05-memory-hitl-deploy`
2. Build inside `assignments/task-05-memory-hitl-deploy/`.
3. Run `python main.py` for two interactive sessions with the same `user_id` and confirm memory carries over.
4. Run `curl` against `POST /chat` + `POST /resume` to demonstrate the HITL flow.
5. Run `docker compose up --build` and repeat the curl demo against the containerized server.
6. Run `python -m eval.run_eval` and commit the resulting `eval/report.md`.
7. Open a Pull Request to `main` — tag the reviewer.

---

## 📚 Resources

- [LangGraph — Memory Store](https://langchain-ai.github.io/langgraph/concepts/memory/)
- [LangGraph — Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph — `interrupt()` reference](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/wait-user-input/)
- [LangGraph — Persistence (`SqliteSaver`)](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [Docker — Multi-stage builds](https://docs.docker.com/build/building/multi-stage/)
- [FastAPI — WebSockets](https://fastapi.tiangolo.com/advanced/websockets/)
- [LangSmith — Adding metadata to runs](https://docs.smith.langchain.com/how_to_guides/tracing/add_metadata_tags)

---

## 🗺️ Suggested Day-by-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Read LangGraph memory docs; implement `memory_store.py` helpers; add `load_memory_node` and `save_memory_node` to the graph; test with `main.py` |
| Day 2 | Verify memory persists across process restarts using `SqliteSaver`; add preference-storing logic; test two-session memory recall |
| Day 3 | Implement `interrupt()` in `tool_node`; test approve and reject paths locally from `main.py` |
| Day 4 | Build `api/server.py`: `POST /chat`, `POST /resume`, `GET /memories/{user_id}`, `GET /health`; wire `SqliteSaver` for thread persistence across restarts |
| Day 5 | Write `Dockerfile` (multi-stage); write `docker-compose.yml`; get `docker compose up` working end-to-end |
| Day 6 | Extend `golden_dataset.json` with 6 memory questions; update `run_eval.py` for two-turn memory scoring; get a passing report |
| Day 7 | Wire LangSmith HITL tags; write `README.md` with all required sections; attempt a bonus |

---

## 🔑 Key Concepts Cheat Sheet

| Concept | Where it appears | Why it matters |
|---|---|---|
| `BaseStore` / `InMemoryStore` | `agent/memory_store.py` | The LangGraph primitive for out-of-band persistent key/value storage separate from graph state |
| `store.aput()` / `store.asearch()` | `load_memory_node`, `save_memory_node` | Async API for writing and searching memory records |
| `interrupt()` | `agent/graph.py` tool node | Pauses graph execution and hands control back to the calling layer |
| `Command(resume=...)` | `api/server.py` POST /resume | The object passed to `graph.ainvoke()` to resume an interrupted thread |
| `SqliteSaver` | `api/server.py` checkpointer | Persists graph checkpoints (including interrupted state) to SQLite so the server can restart without losing pending approvals |
| Multi-stage Docker build | `Dockerfile` | Separates dependency installation from the runtime image, producing a smaller, more secure image |

---

*Long-term memory, human oversight, and reproducible deployment are the three gaps that separate a demo agent from one you'd trust in production. Nail all three here and every future project you build will start from a higher baseline. As always: no dead code, README first, eval harness last — and make the Docker build green before you open the PR.*
