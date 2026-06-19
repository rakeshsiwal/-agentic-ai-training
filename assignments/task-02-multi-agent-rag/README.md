# Task 02 — Multi-Agent RAG System with Human-in-the-Loop

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LangGraph StateGraph                      │
│                                                                   │
│  [START]                                                          │
│     ↓                                                             │
│  supervisor_node  ──────────────────────────────────→  [END]     │
│     ↓ (next="researcher")         ↑ (next="end")                 │
│  researcher_node                  │                               │
│     ↓                             │                               │
│  [INTERRUPT] human_review_node    │                               │
│     ↓ (approved=True)             │                               │
│  supervisor_node ──(next="writer")→ writer_node ─────────────────┘
│     ↓ (approved=False)
│  researcher_node  (re-retrieval loop)
└─────────────────────────────────────────────────────────────────┘
```

### Agent Roles

| Agent | File | Responsibility |
|---|---|---|
| **Supervisor** | `agents/supervisor.py` | Uses structured LLM output to decide routing (`researcher` / `writer` / `end`) |
| **Researcher** | `agents/researcher.py` | Calls `retrieve()` to fetch top-4 relevant chunks from ChromaDB |
| **Human Review** | `graph.py` | LangGraph `interrupt` — pauses for user approval of retrieved context |
| **Writer** | `agents/writer.py` | Drafts a cited, structured answer using approved context chunks |

### RAG Pipeline

| Module | File | Responsibility |
|---|---|---|
| **Ingest** | `rag/ingest.py` | Chunk → embed → persist to ChromaDB (idempotent) |
| **Retriever** | `rag/retriever.py` | Similarity search returning top-k plain-text chunks |

### Shared State (`state.py`)

```python
class PipelineState(TypedDict):
    messages:       Annotated[list, add_messages]  # conversation history
    query:          str        # current question (may be rewritten)
    retrieved_docs: list[str]  # chunks from vector store
    approved:       bool       # human approval flag
    draft:          str        # writer's initial output
    final_answer:   str        # final polished answer
    next_agent:     str        # supervisor's routing decision
```

---

## Setup

### 1. Clone and enter the project

```bash
cd assignments/task-02-multi-agent-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** `sentence-transformers` downloads the `all-MiniLM-L6-v2` model
> (~90 MB) on first use. Subsequent runs use the cached model.

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add at least one LLM API key
```

Supported providers (first found is used):
- `ANTHROPIC_API_KEY` — Claude (recommended)
- `OPENAI_API_KEY` — GPT-4o-mini
- `GOOGLE_API_KEY` — Gemini 1.5 Flash
- `GROQ_API_KEY` — LLaMA 3.1 (free tier available)

### 5. Ingest documents

```bash
python -m rag.ingest
```

This chunkes `.txt` / `.md` files from `data/documents/`, embeds them with
`sentence-transformers/all-MiniLM-L6-v2`, and persists to `data/chroma_db/`.

Re-running is **idempotent** — chunks are identified by a content hash so
existing entries are updated in place.

### 6. Run the pipeline

```bash
# Interactive mode (will prompt for a question)
python main.py

# Single-shot mode
python main.py "What is self-attention in transformers?"
```

---

## Sample Session Log

```
╔══════════════════════════════════════════════════════════╗
║        Multi-Agent RAG System  (LangGraph)               ║
║  Supervisor → Researcher → [HiTL Review] → Writer        ║
╚══════════════════════════════════════════════════════════╝

❓ Question: How does RLHF align language models with human preferences?

🔍 Query: How does RLHF align language models with human preferences?
────────────────────────────────────────────────

[supervisor] → 'researcher' | reason: retrieved_docs is empty, need to fetch context first.

[researcher] Retrieving docs for query: 'How does RLHF align language models with human preferences?'
[researcher] Retrieved 4 chunk(s).

════════════════════════════════════════════════════════════
📄 RETRIEVED CONTEXT
════════════════════════════════════════════════════════════

[Chunk 1]
Reinforcement Learning from Human Feedback (RLHF) is a training technique used to align large
language models with human preferences. It has been central to the success of models like
ChatGPT, Claude, and Gemini.

The RLHF pipeline has three stages. First, supervised fine-tuning (SFT): a pre-trained base
model is fine-tuned on high-quality demonstration data written by human labelers…

[Chunk 2]
Second, reward model training: human annotators compare pairs of model outputs and rank them
by quality. A separate neural network — the reward model — is trained on these preferences…

[Chunk 3]
Third, reinforcement learning with PPO: the SFT model is further trained using Proximal Policy
Optimization (PPO), a policy-gradient RL algorithm…

[Chunk 4]
A newer variant called Direct Preference Optimization (DPO) skips the explicit reward model
entirely, directly optimizing the language model on preference pairs…

════════════════════════════════════════════════════════════
Approve this context?
  y / yes          → proceed to Writer
  n / no           → re-retrieve (same query)
  r <new query>    → replace query and re-retrieve
════════════════════════════════════════════════════════════
Your decision: y

[human_review] ✅ Approved.

[supervisor] → 'writer' | reason: approved is True and draft is empty.

[writer] Drafting answer for: 'How does RLHF align language models with human preferences?'
[writer] Draft complete (1243 chars).

[supervisor] → 'end' | reason: draft is present, pipeline complete.

════════════════════════════════════════════════════════════
✅  FINAL ANSWER
════════════════════════════════════════════════════════════

## Answer

RLHF aligns language models with human preferences through a three-stage process [Chunk 1].

First, **Supervised Fine-Tuning (SFT)**: a pre-trained base model is fine-tuned on
high-quality examples written by human labelers, giving the model a foundation in
instruction-following [Chunk 1].

Second, **Reward Model Training**: human annotators rank pairs of model outputs by
quality. A separate neural network learns from these rankings to predict which responses
humans prefer, assigning each a scalar score [Chunk 2].

Third, **Reinforcement Learning with PPO**: the SFT model generates responses, the reward
model scores them, and the model's weights are updated to maximise that score. A KL-divergence
penalty prevents the model from drifting too far from its SFT starting point [Chunk 3].

A newer alternative, **Direct Preference Optimization (DPO)**, removes the explicit reward
model and instead directly optimises the language model on preference pairs, making training
simpler and more stable [Chunk 4].

## Sources Used
- Chunk 1: Reinforcement Learning from Human Feedback (RLHF) is a training technique used…
- Chunk 2: Second, reward model training: human annotators compare pairs of model outputs…
- Chunk 3: Third, reinforcement learning with PPO: the SFT model is further trained…
- Chunk 4: A newer variant called Direct Preference Optimization (DPO)…

════════════════════════════════════════════════════════════
```

---

## Human-in-the-Loop Flow

The `interrupt_before=["human_review_node"]` compile option causes LangGraph to
**pause execution** before the review node runs. The graph's thread state is
persisted in memory (or SQLite with the bonus `SqliteSaver`).

```
graph.invoke(initial_state, config)   # runs until interrupt
    ↓ paused
# user reads retrieved chunks, types decision
graph.invoke(Command(resume=decision), config)  # resumes
```

Supported decisions:

| Input | Effect |
|---|---|
| `y` / `yes` | Sets `approved=True`, Supervisor routes to Writer |
| `n` / `no` | Clears `retrieved_docs`, Supervisor routes back to Researcher |
| `r <new query>` | Replaces `query` and clears docs, triggering re-retrieval |

---

## Adding More Documents

Drop any `.txt` or `.md` file into `data/documents/` and re-run:

```bash
python -m rag.ingest
```

---

## Bonus Features Attempted

- ⭐⭐ **Query Rewriting via HiTL**: the `r <new query>` option in the human review
  node lets the user rewrite the query before re-retrieval — a lightweight form of
  interactive query rewriting.
- ⭐⭐⭐ **Persistent HiTL** (stub): replace `MemorySaver` in `graph.py` with
  `SqliteSaver(conn)` from `langgraph.checkpoint.sqlite` to persist interrupted
  graph state across terminal sessions. The thread_id must be recorded and reused.
