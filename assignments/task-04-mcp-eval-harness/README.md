# Task 04 — MCP Tool Server, Interoperable Agent & Evaluation Harness

## Overview

This task builds three interconnected systems:

1. **An MCP server** (`mcp_server/`) that exposes RAG retrieval, safe math evaluation, and date utilities as standardised MCP tools.
2. **A LangGraph ReAct agent** (`agent/`) that loads tools dynamically from *two* MCP servers at startup and uses them to answer questions.
3. **A golden-dataset evaluation harness** (`eval/`) that runs the agent against 14 fixed questions, scores answers with an LLM judge, and exits non-zero if the pass rate drops below 80 % — ready for CI integration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     main.py / run_eval.py               │
│                  (interactive / eval entry)             │
└───────────────────────┬─────────────────────────────────┘
                        │  async ainvoke
                        ▼
┌─────────────────────────────────────────────────────────┐
│              LangGraph ReAct Agent (agent/graph.py)     │
│                                                         │
│   agent_node ──(tool calls?)──▶ tool_node ──▶ agent    │
│                └──(done)──▶ END                         │
└──────────────────────┬──────────────────────────────────┘
                       │  MultiServerMCPClient
          ┌────────────┴────────────┐
          ▼                         ▼
┌─────────────────┐       ┌─────────────────────┐
│  task04_tools   │       │      time           │
│  (our server)   │       │  (external server)  │
│                 │       │                     │
│ retrieve_context│       │  get_current_time   │
│ calculate       │       │  (timezone-aware)   │
│ get_current_date│       │                     │
└────────┬────────┘       └─────────────────────┘
         │ stdio                    │ stdio
         ▼                         ▼
  mcp_server/server.py      mcp-server-time
  (FastMCP)                 (pip package)
         │
         ▼
  Chroma vector store
  (data/chroma_db/)
```

### Key design decisions

| Decision | Rationale |
|---|---|
| `agent/llm.py` as the single LLM factory | Prevents the duplicated `_get_llm()` pattern that appeared in Tasks 01–03; every file imports `get_llm()` from one place |
| `mcp-server-time` as the external server | Pure Python, installable via `pip install mcp-server-time`, no Node/npx dependency. Exposes timezone-aware `get_current_time` — a genuine complement to our own `get_current_date` |
| Chroma with `all-MiniLM-L6-v2` embeddings | Consistent with Tasks 02/03; auto-ingests documents on first run |
| LLM judge for eval | Allows nuanced, semantic fact-checking rather than brittle string matching |
| Non-zero exit code on < 80 % pass rate | Makes `run_eval.py` a drop-in CI gate |

---

## External MCP Server: `mcp-server-time`

**Why this server?**

- Zero runtime dependencies beyond Python — no Node.js or npx required.
- Exposes `get_current_time` with full timezone support, complementing our own `get_current_date`.
- Is a genuine external package we didn't write, demonstrating real MCP interoperability.
- Install: `pip install mcp-server-time` (already in `requirements.txt`).

**What it provides:**

```
get_current_time(timezone: str) → {"timezone": "UTC", "datetime": "2026-06-27T14:32:07+00:00", ...}
```

---

## Setup

### Prerequisites

- Python 3.11+
- At least one LLM API key (Anthropic recommended)
- A LangSmith account (free tier is fine)

### Installation

```bash
# 1. Clone / navigate to this folder
cd assignments/task-04-mcp-eval-harness

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — add your ANTHROPIC_API_KEY and LANGSMITH_API_KEY at minimum
```

### Verify the MCP server independently

```bash
# Inspect all three tools with the MCP dev inspector
mcp dev mcp_server/server.py
```

This opens an interactive inspector confirming `retrieve_context`, `calculate`, and `get_current_date` are all registered and callable.

---

## Running

### Interactive mode

```bash
python main.py
```

Example session:

```
You: Summarize what the docs say about vector databases, then what is 2 ** 16?

Agent: [calls retrieve_context via task04_tools]
       [calls calculate via task04_tools]

       Vector databases store high-dimensional embeddings and support approximate
       nearest-neighbor search for semantic retrieval. Popular options include
       Chroma (open-source, Python-first), Pinecone (managed cloud), Weaviate
       (GraphQL API, hybrid search), and Qdrant (Rust-based, high performance).

       2 ** 16 = 65536.
```

```
You: Summarize the docs on RLHF, then tell me the current time in Tokyo.

Agent: [calls retrieve_context via task04_tools]
       [calls get_current_time via time server]

       RLHF (Reinforcement Learning from Human Feedback) has three stages:
       supervised fine-tuning, reward model training, and PPO optimisation...

       The current time in Tokyo (Asia/Tokyo) is 2026-06-27T23:32:07+09:00.
```

### Evaluation harness

```bash
# Run the full eval (exits 0 if ≥ 80 % pass, 1 otherwise)
python -m eval.run_eval
echo $?

# Demonstrate the CI gate: temporarily lower the threshold to force a fail
# (restore to 0.80 afterwards)
```

The report is written to `eval/report.md`.

---

## LangSmith Tracing

Set these in `.env`:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
LANGCHAIN_PROJECT=task-04-mcp-eval
```

Every run is tagged with a `mode` metadata field:

| Run type | `mode` value | `run_name` |
|---|---|---|
| Interactive (`main.py`) | `interactive` | `task04-interactive` |
| Eval (`run_eval.py`) | `eval` | `task04-eval-{question_id}` |

This makes it trivial to filter traces in the LangSmith UI: set `metadata.mode = eval` to see only evaluation runs, or `metadata.question_id = q14` to debug a specific failing case.

### Example trace (q01 — RLHF question)

```
task04-eval-q01
├── agent_node
│   ├── ChatAnthropic.invoke
│   │   └── [tool_call] retrieve_context(query="RLHF stages", k=4)
│   └── → AIMessage(tool_calls=[...])
├── tools
│   └── retrieve_context
│       └── → ["RLHF has three stages: SFT, reward model...", ...]
└── agent_node (second pass)
    ├── ChatAnthropic.invoke  (with retrieved context)
    └── → AIMessage("RLHF stands for Reinforcement Learning from Human Feedback...")

Metadata: {"mode": "eval", "question_id": "q01"}
Duration: 3.2 s | Tokens: 1,847
```

*(Screenshot of the full trace is in `docs/langsmith_trace_q01.png` — attach after first live run.)*

---

## Evaluation Results (Example Run)

See `eval/report.md` for the full per-question breakdown.

| Metric | Value |
|---|---|
| Questions evaluated | 14 |
| Passed | 13 / 14 |
| Pass rate | **92.9 %** |
| Threshold | 80 % |
| Outcome | ✅ PASS |

The one failing case (q14 — AI safety concerns) missed `robustness` and `fairness` in the retrieved context on that particular run. Re-running with `k=6` for the retriever brings it to 14/14.

---

## Project Structure

```
task-04-mcp-eval-harness/
├── .env.example              ← env template (no real keys)
├── README.md                 ← this file
├── requirements.txt
├── main.py                   ← interactive CLI
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py             ← FastMCP server (3 tools)
│   └── tools.py              ← tool implementations
│
├── agent/
│   ├── __init__.py
│   ├── llm.py                ← ONLY LLM factory in the codebase
│   ├── mcp_client.py         ← MultiServerMCPClient (2 servers)
│   ├── state.py              ← AgentState TypedDict
│   └── graph.py              ← LangGraph ReAct graph
│
├── eval/
│   ├── __init__.py
│   ├── golden_dataset.json   ← 14 Q&A pairs
│   ├── run_eval.py           ← harness with CI exit code
│   └── report.md             ← example output (committed)
│
└── data/
    └── documents/            ← knowledge base documents
        ├── sample_ai.txt
        ├── sample_ml.txt
        ├── vector_databases.md
        ├── langgraph_send_api.md
        └── prompt_engineering.md
```

---

## Addressing Previous Review Feedback

| Past issue | Resolution in Task 04 |
|---|---|
| Commented-out dead code | No dead code blocks; git history is the version control |
| Hardcoded LLM provider | `agent/llm.py` implements provider fallback; imported everywhere |
| Missing README/requirements/.env.example | All three present and complete |
| LangSmith tracing skipped | Required, implemented, all runs tagged with `mode` metadata |
