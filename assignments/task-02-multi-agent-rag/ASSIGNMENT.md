# Assignment Task 02 — Multi-Agent RAG System with Human-in-the-Loop

**Issued:** June 12, 2026
**Due:** June 19, 2026
**Difficulty:** Intermediate → Advanced
**Branch naming:** `feature/task-02-multi-agent-rag`
**Prerequisite:** Task 01 completed and reviewed

---

## 🎯 Objective

Build a **multi-agent RAG (Retrieval-Augmented Generation) pipeline** using LangGraph. Instead of a single agent doing everything, you will implement a **Supervisor** agent that delegates work to two specialized sub-agents:

- A **Researcher Agent** — retrieves relevant chunks from a local vector store (RAG).
- A **Writer Agent** — drafts a polished, structured response using retrieved context.

The system must include a **human-in-the-loop** checkpoint that pauses before writing and asks the user to approve the retrieved context.

---

## 🧠 What You Will Learn

- The **Supervisor / Worker** multi-agent pattern — the core architecture used in production agentic systems.
- **RAG fundamentals** — chunking documents, embedding text, storing in a vector store, and retrieving by similarity.
- **Human-in-the-Loop (HiTL)** — using LangGraph's `interrupt` mechanism to pause execution and collect user approval.
- **Subgraph composition** — building each agent as its own `StateGraph` and wiring them inside a parent graph.
- **Shared vs. private state** — understanding what data flows between agents vs. what stays local.
- **Structured output** — forcing the LLM to return JSON/Pydantic objects using `.with_structured_output()`.

---

## 🗂️ Folder Structure (Expected Output)

```
assignments/task-02-multi-agent-rag/
│
├── ASSIGNMENT.md              ← this file (do not modify)
├── README.md                  ← your notes: architecture, setup, how to run
├── requirements.txt
├── .env.example
├── data/
│   └── documents/             ← put .txt or .md files here to index
│       ├── sample_ai.txt      ← at least 2 sample docs (you write them)
│       └── sample_ml.txt
├── agents/
│   ├── __init__.py
│   ├── supervisor.py          ← Supervisor agent node + routing logic
│   ├── researcher.py          ← Researcher agent (RAG retrieval)
│   └── writer.py              ← Writer agent (drafts final answer)
├── rag/
│   ├── __init__.py
│   ├── ingest.py              ← chunk + embed + store documents
│   └── retriever.py           ← similarity search wrapper
├── state.py                   ← shared PipelineState TypedDict
├── graph.py                   ← top-level graph wiring all agents
└── main.py                    ← CLI entry point
```

---

## ⚙️ Technical Requirements

### 1. Shared Pipeline State

Define a `TypedDict` that all agents read from and write to:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class PipelineState(TypedDict):
    messages:        Annotated[list, add_messages]  # conversation history
    query:           str                             # user's original question
    retrieved_docs:  list[str]                       # chunks from vector store
    approved:        bool                            # did user approve the context?
    draft:           str                             # Writer's drafted answer
    final_answer:    str                             # polished final output
    next_agent:      str                             # Supervisor routing decision
```

---

### 2. RAG Pipeline (the `rag/` package)

#### `ingest.py`
- Load `.txt` / `.md` files from `data/documents/`.
- **Chunk** each file into overlapping segments (e.g. 500 chars, 100 overlap).
- **Embed** chunks using one of:
  - `langchain_openai.OpenAIEmbeddings` (if you have an OpenAI key)
  - `langchain_community.embeddings.HuggingFaceEmbeddings` with `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally)
- Store in **ChromaDB** (via `langchain_chroma`), persisted to `data/chroma_db/`.
- The ingest step should be **idempotent** — re-running it should not duplicate documents.

#### `retriever.py`
- Wrap the Chroma collection in a function `retrieve(query: str, k: int = 4) -> list[str]`.
- Returns the top-k most similar chunks as plain strings.

---

### 3. Agent Nodes

#### `agents/supervisor.py` — Supervisor Node
- Receives the user query.
- Uses **structured output** to decide which agent to call next:
  ```python
  from pydantic import BaseModel
  class RouteDecision(BaseModel):
      next: Literal["researcher", "writer", "end"]
      reasoning: str
  ```
- Routing rules:
  - If `retrieved_docs` is empty → route to `"researcher"`
  - If `retrieved_docs` exist but `approved` is `False` → wait for HiTL
  - If `approved` is `True` and `draft` is empty → route to `"writer"`
  - If `draft` is not empty → route to `"end"`

#### `agents/researcher.py` — Researcher Node
- Calls `retrieve(query, k=4)` to fetch relevant chunks.
- Stores them in `state["retrieved_docs"]`.
- Optionally: also calls `search_web` from Task 01's tools if local docs are insufficient.

#### `agents/writer.py` — Writer Node
- Only runs after `approved == True`.
- Receives the query + retrieved docs via state.
- Prompts the LLM to write a **structured, cited answer** (reference doc chunks by index).
- Stores the result in `state["draft"]` and `state["final_answer"]`.

---

### 4. Human-in-the-Loop (REQUIRED)

After the Researcher runs, the graph **must pause** and show the user the retrieved chunks. The user decides:

- **`y` / `yes`** → set `approved = True`, continue to Writer.
- **`n` / `no`** → set `approved = False`, let Supervisor re-route (e.g. refine query and search again).
- **`r <new query>`** → replace the query and restart the researcher.

Implement this using LangGraph's interrupt mechanism:

```python
from langgraph.types import interrupt

def human_review_node(state: PipelineState) -> PipelineState:
    decision = interrupt({
        "retrieved_docs": state["retrieved_docs"],
        "message": "Review the retrieved context above. Approve? (y/n/r <new query>)"
    })
    # parse `decision` and update state accordingly
    ...
```

Compile the graph with `interrupt_before=["human_review_node"]`.

---

### 5. Graph Topology

```
[START]
   ↓
supervisor_node
   ↓ (conditional — next_agent)
┌─────────────────────────────────────────────────┐
│  researcher_node  → human_review_node            │
│       ↓ (approved)                               │
│  writer_node      → supervisor_node → END        │
└─────────────────────────────────────────────────┘
```

The graph must use `add_conditional_edges` from `supervisor_node`, routing based on `state["next_agent"]`.

---

### 6. Sample Documents

Create at least **2 text files** in `data/documents/`. They can be about any AI/ML topic (100–300 words each). The agent will answer questions about these docs.

Example topics:
- Transformer architecture overview
- Reinforcement Learning from Human Feedback (RLHF)
- Vector databases and embeddings
- LangGraph vs. LangChain differences

---

## ✅ Acceptance Criteria

Before submitting your Pull Request:

- [ ] `ingest.py` runs and populates `data/chroma_db/` without errors
- [ ] `retrieve()` returns relevant chunks for a sample query
- [ ] Supervisor correctly routes between researcher → HiTL → writer
- [ ] HiTL node pauses execution and waits for user input
- [ ] Writer only runs after `approved == True`
- [ ] Full pipeline answers a question end-to-end without crashing
- [ ] `retrieved_docs` are cited / referenced in the final answer
- [ ] `.env.example` present, no real keys committed
- [ ] `requirements.txt` complete and installable
- [ ] `README.md` explains the architecture, ingest step, and how to run

---

## 🌟 Bonus Challenges (optional)

| Stars | Bonus | Description |
|---|---|---|
| ⭐ | Re-ranking | After retrieval, use a cross-encoder or LLM call to re-rank chunks by relevance before passing to Writer |
| ⭐⭐ | Query Rewriting | Add a `query_rewriter_node` that rephrases the user query to improve retrieval recall |
| ⭐⭐ | Streaming Writer | Stream the Writer's token output to the terminal in real time |
| ⭐⭐⭐ | Persistent HiTL | Use `SqliteSaver` so an interrupted graph can be resumed from a different terminal session |
| ⭐⭐⭐ | Evaluation Node | After the Writer outputs, add an `evaluator_node` (another LLM call) that scores the answer for faithfulness and relevance on a 1–5 scale |

---

## 📦 Starter Dependencies

```txt
# Core
langgraph>=0.2.0
langchain>=0.3.0
langchain-core>=0.3.0

# LLM providers (at least one)
langchain-openai>=0.2.0
langchain-anthropic>=0.2.0
langchain-google-genai>=2.0.0
langchain-groq>=0.2.0

# RAG
langchain-chroma>=0.1.0         # ChromaDB integration
chromadb>=0.5.0                  # vector store
langchain-community>=0.3.0       # HuggingFaceEmbeddings (free embeddings)
sentence-transformers>=3.0.0     # local embedding model (no API key needed)

# Utilities
python-dotenv>=1.0.0
```

> **Tip — no OpenAI key for embeddings?**
> Use `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")` — it runs
> entirely locally with no API cost. It's slower on first run (model download) but free forever.

---

## 📤 Submission

1. `git checkout -b feature/task-02-multi-agent-rag`
2. Build inside `assignments/task-02-multi-agent-rag/`.
3. Run `python main.py` end-to-end at least once and include a sample session log in your `README.md`.
4. Open a Pull Request to `main` — tag the reviewer.

---

## 📚 Resources

- [LangGraph — Multi-Agent Architectures](https://langchain-ai.github.io/langgraph/concepts/multi_agent/)
- [LangGraph — Human-in-the-Loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
- [LangGraph — Subgraphs](https://langchain-ai.github.io/langgraph/how-tos/subgraph/)
- [LangChain — RAG Quickstart](https://python.langchain.com/docs/tutorials/rag/)
- [ChromaDB Docs](https://docs.trychroma.com/)
- [Sentence Transformers (free embeddings)](https://www.sbert.net/)
- [LangChain — Structured Output](https://python.langchain.com/docs/how_to/structured_output/)

---

## 🗺️ Suggested Day-by-Day Plan

| Day | Focus |
|---|---|
| Day 1 | Set up the folder, write sample docs, get `ingest.py` working (ChromaDB populated) |
| Day 2 | Build `retriever.py` and test retrieval in isolation with a simple script |
| Day 3 | Build `researcher_node` and `supervisor_node` (no HiTL yet — just routing) |
| Day 4 | Add `human_review_node` with interrupt, test pause/resume flow |
| Day 5 | Build `writer_node`, wire the full graph, end-to-end test |
| Day 6 | Polish: README, error handling, clean up, attempt a bonus |
| Day 7 | Buffer / bonus day |

---

*The jump from Task 01 to Task 02 is intentional — you are expected to hit walls. Read the LangGraph docs, experiment in isolation, and ask questions. The goal is to understand how real production agentic systems are structured.*
