# Assignment Task 01 — Build a Personal Research Agent with LangGraph

**Issued:** June 5, 2026
**Due:** June 12, 2026
**Difficulty:** Beginner → Intermediate
**Branch naming:** `feature/task-01-langgraph-research-agent`

---

## 🎯 Objective

Build a **stateful, conversational Research Agent** using **LangGraph** and an LLM of your choice (OpenAI GPT-4o, Anthropic Claude, or Google Gemini). The agent should be able to:

1. Accept a research **topic** from the user.
2. **Search the web** (or simulate search results) for information.
3. **Summarize** findings in a clear, structured format.
4. Maintain **conversation memory** so the user can ask follow-up questions.
5. Know **when to stop** — gracefully end the loop when the user is satisfied.

---

## 🧠 What You Will Learn

- How to model an agent as a **graph of nodes and edges** using LangGraph.
- The concept of **agent state** — passing information between steps.
- How to bind **tools** to an LLM (tool calling / function calling).
- Implementing **conditional edges** to decide the next step dynamically.
- Using a **checkpointer** for conversation memory across turns.

---

## 🗂️ Folder Structure (Expected Output)

Place all your work inside this folder:

```
assignments/task-01-langgraph-research-agent/
│
├── ASSIGNMENT.md          ← this file (do not modify)
├── README.md              ← your own notes: what you built, how to run it
├── requirements.txt       ← all Python dependencies
├── .env.example           ← template showing which env vars are needed (NO real keys)
├── agent/
│   ├── __init__.py
│   ├── graph.py           ← LangGraph graph definition
│   ├── state.py           ← AgentState TypedDict
│   ├── nodes.py           ← individual node functions (reason, search, summarize)
│   └── tools.py           ← tool definitions (web search, calculator, etc.)
└── main.py                ← entry point to run the agent interactively
```

---

## ⚙️ Technical Requirements

### 1. LLM Setup (pick ONE)

| Provider | Model Suggestion | Env Variable |
|---|---|---|
| OpenAI | `gpt-4o` or `gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| Google | `gemini-1.5-pro` | `GOOGLE_API_KEY` |
| Groq (free tier) | `llama-3.1-70b-versatile` | `GROQ_API_KEY` |

> **Tip:** Groq offers a generous free tier — great if you don't have paid API access.

---

### 2. Required LangGraph Node Architecture

Your graph **must** contain at least these 4 nodes:

```
[START] → agent_node → tool_node → summarizer_node → [END or back to agent_node]
```

| Node | Responsibility |
|---|---|
| `agent_node` | Calls the LLM; decides whether to use a tool or end |
| `tool_node` | Executes the chosen tool and returns results |
| `summarizer_node` | Condenses tool results into a human-readable answer |
| Conditional edge | Routes back to `agent_node` for follow-ups OR to `END` |

---

### 3. Required Tools (implement at least 2)

| Tool | Description |
|---|---|
| `search_web(query: str)` | Simulated or real web search (use DuckDuckGo via `duckduckgo-search` or `tavily-python`) |
| `calculate(expression: str)` | Evaluate a math expression safely |
| `get_current_date()` | Returns today's date — useful for time-aware queries |

---

### 4. Agent State

Define a `TypedDict` state with at minimum:

```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    topic: str                                 # the research topic
    search_results: list[str]                  # raw results from tools
    final_summary: str                         # the final answer
```

---

### 5. Memory (Checkpointer)

Use LangGraph's built-in `MemorySaver` so the agent remembers previous turns within a session:

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)
```

---

## 🚀 Example Interaction (what it should feel like)

```
You: Research the latest trends in multimodal AI models.

Agent: Searching for "latest trends in multimodal AI models"...
       Found 3 results. Summarizing...

       📋 Summary:
       Multimodal AI models in 2026 are focusing on:
       1. Real-time video understanding (Gemini 2.0, GPT-5 Vision)
       2. Audio-visual joint embeddings for robotics
       3. Unified tokenizers that handle text, image, audio, and code...

You: What companies are leading this space?

Agent: Based on what we found, the leading companies are:
       OpenAI, Google DeepMind, Meta AI, and Mistral AI...

You: Thanks, I'm done.

Agent: Great! Happy to help whenever you need more research. Goodbye! 👋
```

---

## ✅ Acceptance Criteria (Checklist)

Before submitting your Pull Request, make sure:

- [ ] LangGraph graph compiles and runs without errors
- [ ] At least 2 tools are implemented and callable by the LLM
- [ ] Agent maintains context across at least 2 follow-up questions
- [ ] `.env.example` is committed but **never** the actual `.env` file
- [ ] `requirements.txt` is complete and `pip install -r requirements.txt` works
- [ ] `README.md` explains how to set up and run the agent
- [ ] Code is clean — no unused imports, no hardcoded API keys
- [ ] Agent gracefully handles an unknown/unanswerable query

---

## 📦 Dependencies (starter list)

```txt
langgraph>=0.2.0
langchain>=0.3.0
langchain-openai>=0.2.0        # if using OpenAI
langchain-anthropic>=0.2.0     # if using Anthropic
langchain-google-genai>=2.0.0  # if using Gemini
langchain-groq>=0.2.0          # if using Groq
duckduckgo-search>=6.0.0
python-dotenv>=1.0.0
```

---

## 🌟 Bonus Challenges (optional, for extra depth)

| Bonus | Description |
|---|---|
| ⭐ Streaming | Stream the agent's response token-by-token to the terminal |
| ⭐⭐ Interrupt & Resume | Use LangGraph's `interrupt_before` to pause and ask the user to approve a tool call before executing |
| ⭐⭐⭐ Persistent Storage | Swap `MemorySaver` for `SqliteSaver` to persist memory across program restarts |
| ⭐⭐⭐ Multi-Agent | Add a second specialized agent (e.g., a "Fact Checker" agent) that the main agent delegates to |

---

## 📤 Submission

1. Create a branch: `git checkout -b feature/task-01-langgraph-research-agent`
2. Build your agent inside this folder.
3. Commit regularly with meaningful messages.
4. Open a Pull Request to `main` — reference this task in the PR description.
5. Request a code review.

---

## 📚 Resources

- [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/)
- [LangGraph — How to use tools](https://langchain-ai.github.io/langgraph/how-tos/tool-calling/)
- [LangGraph — Memory & Checkpointers](https://langchain-ai.github.io/langgraph/concepts/persistence/)
- [Groq Free API (fast inference)](https://console.groq.com/)
- [DuckDuckGo Search Python](https://pypi.org/project/duckduckgo-search/)
- [python-dotenv docs](https://pypi.org/project/python-dotenv/)

---

*Good luck! The goal isn't a perfect agent on the first try — it's about understanding how nodes, state, and edges work together. Ask questions early and often.*
