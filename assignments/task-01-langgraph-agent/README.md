# Task 01 — Personal Research Agent (LangGraph)

A stateful, conversational research agent built with **LangGraph** and your choice of LLM. The agent searches the web, answers follow-up questions, and remembers the full conversation within a session.

---

## Architecture

```
[START]
   ↓
agent_node          ← LLM decides: call a tool, summarize, or say goodbye
   ↓ (conditional edge)
┌──────────────────────────────────────────┐
│  tool_node         → back to agent_node  │  tools: search_web, calculate, get_current_date
│  summarizer_node   → END                 │  condenses findings into a clean answer
│  END                                     │  user said goodbye / agent finished
└──────────────────────────────────────────┘
```

State is persisted across turns within a session using LangGraph's `MemorySaver`.

---

## Folder structure

```
task-01-langgraph-research-agent/
├── ASSIGNMENT.md
├── README.md               ← you are here
├── requirements.txt
├── .env.example
├── main.py                 ← run this
└── agent/
    ├── __init__.py
    ├── graph.py            ← StateGraph definition + MemorySaver
    ├── state.py            ← AgentState TypedDict
    ├── nodes.py            ← agent_node, tool_node, summarizer_node, should_continue
    └── tools.py            ← search_web, calculate, get_current_date
```

---

## Setup

### 1. Clone and enter the folder

```bash
cd assignments/task-01-langgraph-research-agent
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure your API key

```bash
cp .env.example .env
```

Open `.env` and fill in **at least one** LLM provider key. The agent auto-detects which key is present and uses that provider. Priority order: OpenAI → Anthropic → Google → Groq.

> **No paid account?** [Groq](https://console.groq.com/) offers a generous free tier.

### 5. Run the agent

```bash
python main.py
```

---

## Example session

```
You › Research the latest trends in multimodal AI models.

  ⚙  search_web({'query': 'latest trends multimodal AI models 2026'})
  ⚙  get_current_date({})

Agent › 📋 Here's what I found on multimodal AI trends:

  1. Real-time video understanding (Gemini 2.0, GPT-5 Vision)
  2. Audio-visual joint embeddings for robotics applications
  3. Unified tokenizers handling text, image, audio, and code
  ...

You › What companies are leading this space?

Agent › Based on the search results, the leading organizations are:
  - Google DeepMind (Gemini series)
  - OpenAI (GPT-4o, Sora)
  - Meta AI (Llama multimodal)
  - Mistral AI (Pixtral)

You › Thanks, I'm done.

Agent › Goodbye! Happy researching. 👋
```

---

## Tools

| Tool | Description |
|---|---|
| `search_web(query)` | DuckDuckGo live search; falls back to simulated results if offline |
| `calculate(expression)` | Safe math eval — supports all `math` module functions |
| `get_current_date()` | Returns today's ISO date for time-aware queries |

---

## Design decisions

- **Auto-provider detection**: the `_get_llm()` helper in `nodes.py` checks env vars in order, so you only need one key. Easy to extend.
- **Summarizer is a separate node**: keeps the reasoning LLM (with tools) decoupled from the presentation LLM. Makes swapping models trivial.
- **Simulated search fallback**: `tools.py` includes `_simulated_search()` so the agent runs even without internet or a DuckDuckGo install — useful for demos and CI.
- **Session-scoped memory**: `MemorySaver` stores state per `thread_id`. Each `main.py` run gets a fresh UUID, so sessions are independent. See the Bonus section below to persist across restarts.

---

## Bonus challenges completed

- [x] *(built-in)* Graceful handling of unanswerable queries — the agent says so clearly.
- [ ] ⭐ Streaming — add `stream=True` to `llm.invoke()` calls in `nodes.py`.
- [ ] ⭐⭐ Interrupt & Resume — add `interrupt_before=["tool_node"]` to `builder.compile()`.
- [ ] ⭐⭐⭐ Persistent storage — swap `MemorySaver` for `SqliteSaver` in `graph.py`.
- [ ] ⭐⭐⭐ Multi-agent — add a `fact_checker_node` that the `summarizer_node` delegates to.

---

## Acceptance checklist

- [x] LangGraph graph compiles and runs without errors
- [x] At least 2 tools implemented (`search_web`, `calculate`, `get_current_date`)
- [x] Agent maintains context across follow-up questions (MemorySaver)
- [x] `.env.example` committed, `.env` in `.gitignore`
- [x] `requirements.txt` complete
- [x] `README.md` explains setup and usage
- [x] No hardcoded API keys, no unused imports
- [x] Agent gracefully handles unanswerable queries
