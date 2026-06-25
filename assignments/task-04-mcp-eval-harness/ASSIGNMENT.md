# Assignment Task 04 — MCP Tool Server, Interoperable Agent, and Evaluation Harness

**Issued:** June 24, 2026
**Due:** July 1, 2026
**Difficulty:** Expert
**Branch naming:** `feature/task-04-mcp-eval-harness`
**Prerequisite:** Task 03 completed and reviewed

---

## 🎯 Objective

So far every tool your agents have used was a Python function you wrote and imported directly. That doesn't scale once multiple teams or agents need to share tools. **MCP (Model Context Protocol)** — the open standard Anthropic created for exactly this problem — lets you expose tools, resources, and prompts behind a stable interface that *any* compliant client can consume, and lets your agent consume tools it didn't write.

In Task 04 you will:

1. Build an **MCP server** that exposes your existing RAG retriever and two utility tools as standardized MCP tools.
2. Build an **MCP client agent** in LangGraph that connects to **your** server *and* to at least one **external** MCP server — proving your agent isn't hardcoded to its own tools.
3. Build a **golden-dataset evaluation harness** that runs your agent against a fixed set of questions and produces a pass/fail report — the difference between "I tried it once and it worked" and "I can prove it works."
4. Wire up **LangSmith tracing as a required feature**.

---

## 📌 Carrying Forward Feedback From Previous Reviews

Before you start, three things from the Task 02/03 reviews apply directly to this task and will be checked again:

| Past issue | What to do differently this time |
|---|---|
| Commented-out dead code left in `ingest.py`, then spread to 4 files in Task 03 | Delete code you're not using. If you switched providers, remove the old branch entirely — git history is where "the old version" belongs, not a comment block. |
| Task 03 silently dropped multi-provider LLM support (hardcoded to OpenAI) | Keep the `_get_llm()` provider-fallback pattern from Tasks 01/02. If you introduce a shared `agent/llm.py` utility this time, even better — stop duplicating it per file. |
| `README.md` / `requirements.txt` / `.env.example` were missing entirely from the Task 03 submission | These three files are **non-negotiable acceptance criteria**, not nice-to-haves. A PR without them does not get reviewed. |
| LangSmith tracing offered as a ⭐⭐⭐ bonus in Task 02 and Task 03, attempted neither time | It is **required** in this task (see §5). |

---

## 🧠 What You Will Learn

- **The Model Context Protocol (MCP)** — the client/server standard for exposing tools, resources, and prompts to LLM applications.
- Building an MCP server with the official `mcp` Python SDK's `FastMCP` class and `@mcp.tool()` decorator.
- Connecting a LangGraph agent to **multiple** MCP servers simultaneously via `langchain-mcp-adapters`.
- Why standardizing the tool interface matters once more than one agent or team needs the same capability.
- Building a **golden dataset** and a repeatable, scriptable **evaluation harness** with a pass/fail exit code (CI-ready).
- **LangSmith tracing** — naming runs, attaching metadata, and using traces to debug a failing eval case.

---

## 🗂️ Folder Structure (Expected Output)

```
assignments/task-04-mcp-eval-harness/
│
├── ASSIGNMENT.md                  ← this file (do not modify)
├── README.md                      ← your notes: architecture, setup, how to run, eval report summary
├── requirements.txt
├── .env.example
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py                  ← FastMCP server: registers tools, runs over stdio
│   └── tools.py                   ← tool implementations (retrieve, calculate, get_current_date)
│
├── agent/
│   ├── __init__.py
│   ├── llm.py                     ← shared multi-provider LLM factory (no more duplicated _get_llm())
│   ├── mcp_client.py              ← MultiServerMCPClient setup — connects to N servers
│   ├── state.py
│   └── graph.py                   ← LangGraph ReAct-style agent using MCP-sourced tools
│
├── eval/
│   ├── __init__.py
│   ├── golden_dataset.json        ← 12+ Q&A pairs with expected facts
│   ├── run_eval.py                ← runs the agent against the dataset, scores, writes report
│   └── report.md                  ← generated output (commit one example run)
│
├── data/
│   └── documents/                 ← reuse/extend documents from Task 02/03
│
└── main.py                        ← CLI entry point for interactive use
```

---

## ⚙️ Technical Requirements

### 1. MCP Server (`mcp_server/`)

Use the official `mcp` SDK's `FastMCP` to expose three tools:

```python
# mcp_server/server.py
from mcp.server.fastmcp import FastMCP
from mcp_server.tools import retrieve, calculate, get_current_date

mcp = FastMCP("task04-tools")

@mcp.tool()
def retrieve_context(query: str, k: int = 4) -> list[str]:
    """Retrieve the top-k relevant chunks from the local document store."""
    return retrieve(query, k)

@mcp.tool()
def calculate(expression: str) -> str:
    """Safely evaluate a math expression."""
    ...

@mcp.tool()
def get_current_date() -> str:
    """Return today's date."""
    ...

if __name__ == "__main__":
    mcp.run(transport="stdio")
```

- `retrieve_context` wraps the **same Chroma vector store** you built in Task 02/03 — do not reimplement RAG from scratch, import and reuse it.
- The server must run standalone: `python -m mcp_server.server`.
- Verify it independently with the MCP inspector before wiring up the client:
  ```bash
  mcp dev mcp_server/server.py
  ```

---

### 2. MCP Client Agent (`agent/`)

Your agent must connect to **at least two** MCP servers:

1. **Your own server** (`mcp_server/server.py`), launched as a subprocess over stdio.
2. **One external/reference MCP server** — e.g. the official filesystem server (`npx -y @modelcontextprotocol/server-filesystem <dir>`) or any other public stdio MCP server of your choice. Document which one you picked and why in `README.md`.

Use `langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient({
    "task04_tools": {
        "command": "python",
        "args": ["-m", "mcp_server.server"],
        "transport": "stdio",
    },
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "./data"],
        "transport": "stdio",
    },
})
tools = await client.get_tools()
```

- Build a LangGraph graph with the familiar `agent_node ↔ tool_node` loop from Task 01 — but the tools are now loaded **dynamically** from MCP at startup instead of hardcoded.
- The agent must demonstrate at least one query that requires a tool from **each** server in the same conversation (e.g., "Summarize what our docs say about X, then list the files in the data directory").

---

### 3. Shared LLM Factory (`agent/llm.py`)

Extract the multi-provider `_get_llm()` pattern from Tasks 01/02 into one shared function, imported everywhere an LLM is needed in this task. No file should define its own copy.

```python
def get_llm(temperature: float = 0.0):
    """Return a chat model based on the first available API key."""
    ...
```

---

### 4. Evaluation Harness (`eval/`)

#### `golden_dataset.json`

At least **12 question/answer pairs**, each shaped like:

```json
{
  "id": "q01",
  "question": "What does RLHF stand for and what are its three stages?",
  "expected_facts": [
    "Reinforcement Learning from Human Feedback",
    "supervised fine-tuning",
    "reward model training",
    "PPO"
  ],
  "source_doc": "sample_ml.txt"
}
```

#### `run_eval.py`

- Iterates the dataset, runs the full agent for each `question`.
- Uses an LLM judge (reuse the Task 03 evaluator pattern) to check whether the agent's answer covers each `expected_facts` entry and is faithful to retrieved context.
- Computes an aggregate pass rate (a question "passes" if faithfulness ≥ 3 **and** relevance ≥ 3 **and** ≥ 70% of `expected_facts` are covered).
- **Exits with a non-zero status code if the aggregate pass rate is below 80%** — this script must be usable as a CI gate.
- Writes a human-readable `eval/report.md` with a per-question breakdown.

```bash
python -m eval.run_eval
echo $?   # 0 if pass rate >= 80%, 1 otherwise
```

---

### 5. LangSmith Tracing (REQUIRED)

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=...
LANGCHAIN_PROJECT=task-04-mcp-eval
```

- Every node execution must appear as a named, inspectable run in your LangSmith project.
- Tag eval-harness runs with metadata `{"mode": "eval", "question_id": "..."}` and interactive runs with `{"mode": "interactive"}` so the two are distinguishable in the LangSmith UI.
- Include a screenshot (or exported trace JSON) of one full trace in `README.md`.

---

## 🚀 Example Interaction

```
You: Summarize what our documents say about vector databases, then tell me
     what files exist in the data directory.

Agent: [calls retrieve_context via task04_tools MCP server]
       [calls list_directory via filesystem MCP server]

       Our documents explain that vector databases store embeddings and
       support approximate nearest-neighbor search for semantic retrieval...

       The data directory contains: sample_ai.txt, sample_ml.txt,
       vector_databases.md, langgraph_send_api.md, prompt_engineering.md
```

---

## ✅ Acceptance Criteria

- [ ] `mcp_server/server.py` starts and passes inspection via `mcp dev`
- [ ] `retrieve_context` tool returns real chunks from the existing vector store (not mocked)
- [ ] Agent successfully loads and calls tools from **two distinct MCP servers** in one conversation
- [ ] `agent/llm.py` is the **only** place an LLM provider is selected — no duplicated `_get_llm()` per file
- [ ] No commented-out/dead implementation blocks left in any file
- [ ] `golden_dataset.json` has ≥ 12 entries covering all sample documents
- [ ] `run_eval.py` runs end-to-end, produces `eval/report.md`, and exits non-zero on a forced failing run (demonstrate this once, then restore passing state)
- [ ] LangSmith traces are visible for both interactive and eval runs, tagged with the `mode` metadata
- [ ] `.env.example` present, no real keys committed
- [ ] `requirements.txt` complete and installable
- [ ] `README.md` explains architecture, which external MCP server you chose and why, setup steps, how to run the eval harness, and includes one example trace

---

## 🌟 Bonus Challenges (optional)

| Stars | Bonus | Description |
|---|---|---|
| ⭐ | Third MCP Server | Connect a third public MCP server (e.g. a search or GitHub MCP server) and use it in the same conversation |
| ⭐⭐ | MCP Resource | Expose your `eval/report.md` as an MCP **resource** (`@mcp.resource()`) so another agent could read your eval history without filesystem access |
| ⭐⭐ | MCP Prompt | Expose your agent's system prompt via `@mcp.prompt()` instead of hardcoding it client-side |
| ⭐⭐⭐ | CI Integration | Add a GitHub Actions workflow that runs `run_eval.py` on every PR and fails the check if the pass rate drops below 80% |
| ⭐⭐⭐ | HTTP/SSE Transport | Run your MCP server over the `streamable-http` transport instead of stdio, so it could be deployed remotely rather than spawned as a local subprocess |

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

# MCP
mcp>=1.2.0
langchain-mcp-adapters>=0.1.0

# RAG (reuse from Task 02/03)
langchain-chroma>=0.1.0
chromadb>=0.5.0
langchain-community>=0.3.0
sentence-transformers>=3.0.0

# Observability
langsmith>=0.1.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.0.0
```

> **Tip:** The external filesystem MCP server runs via `npx`, which requires Node.js installed. If you'd rather avoid a Node dependency, pick any other public stdio-based MCP server written in Python instead — document your choice either way.

---

## 📤 Submission

1. `git checkout -b feature/task-04-mcp-eval-harness`
2. Build inside `assignments/task-04-mcp-eval-harness/`.
3. Run `mcp dev mcp_server/server.py` once to confirm the server is independently valid.
4. Run `python main.py` for an interactive session using tools from both MCP servers.
5. Run `python -m eval.run_eval` and include the resulting `eval/report.md` in your PR.
6. Open a Pull Request to `main` — tag the reviewer.

---

## 📚 Resources

- [Model Context Protocol — Official Docs](https://modelcontextprotocol.io/)
- [MCP Python SDK (FastMCP)](https://github.com/modelcontextprotocol/python-sdk)
- [langchain-mcp-adapters](https://github.com/langchain-ai/langchain-mcp-adapters)
- [MCP — Official Reference Servers](https://github.com/modelcontextprotocol/servers)
- [LangSmith — Tracing Quickstart](https://docs.smith.langchain.com/observability)
- [LangGraph — ReAct Agent Pattern](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)

---

## 🗺️ Suggested Day-by-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Read MCP docs; build `mcp_server/server.py` with the 3 tools; validate with `mcp dev` |
| Day 2 | Build `agent/mcp_client.py`, connect to your own server only, get a basic tool call working |
| Day 3 | Add the second (external) MCP server; get a single conversation to use tools from both |
| Day 4 | Extract `agent/llm.py`; clean up any duplicated/dead code as you go |
| Day 5 | Build `golden_dataset.json` (12+ entries) and `run_eval.py`; get a passing report |
| Day 6 | Wire up LangSmith tracing with mode metadata; capture an example trace for the README |
| Day 7 | Polish: README, error handling, attempt a bonus |

---

## 🔑 Key Concepts Cheat Sheet

| Concept | Where it appears | Why it matters |
|---|---|---|
| `FastMCP` + `@mcp.tool()` | `mcp_server/server.py` | Standardized way to expose a function as an LLM-callable tool |
| stdio transport | both server and client config | The default MCP transport — server runs as a subprocess of the client |
| `MultiServerMCPClient` | `agent/mcp_client.py` | Loads tools from N independent MCP servers into one LangChain tool list |
| Golden dataset + exit code | `eval/run_eval.py` | Turns "it worked when I tried it" into a repeatable, CI-gateable check |
| LangSmith run metadata | every node | Lets you filter traces by `mode` (eval vs. interactive) when debugging a regression |

---

*MCP is the same protocol underpinning Claude Desktop, Claude Code, and the broader Anthropic tool ecosystem — the skills here transfer directly. The evaluation harness is just as important as the protocol work: an agent you can't systematically test is an agent you can't safely change. Treat the "no dead code, no missing README" items as seriously as the MCP wiring itself — they will be checked first.*
