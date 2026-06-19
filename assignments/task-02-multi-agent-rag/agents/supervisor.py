"""
agents/supervisor.py — Supervisor node with structured-output routing.

The Supervisor decides which agent should run next based on the current
pipeline state, using an LLM with `.with_structured_output()` to enforce
a typed routing decision.
"""

from __future__ import annotations

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from pydantic import BaseModel, Field

from state import PipelineState

load_dotenv()


# ── Structured routing schema ──────────────────────────────────────────────────

class RouteDecision(BaseModel):
    """The supervisor's routing decision."""
    next: Literal["researcher", "writer", "end"] = Field(
        description="Which agent to invoke next, or 'end' to finish."
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen."
    )


# ── LLM factory ───────────────────────────────────────────────────────────────

def _get_llm():
    """Return a chat LLM based on available API keys (priority order)."""
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    raise EnvironmentError(
        "No LLM API key found. Set one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "GOOGLE_API_KEY, or GROQ_API_KEY in your .env file."
    )


# ── Supervisor node ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a Supervisor agent orchestrating a RAG (Retrieval-Augmented Generation) pipeline.

Your job is to decide which agent should run next based on the current state of the pipeline.

Routing rules (apply in order):
1. If retrieved_docs is empty → route to "researcher" to fetch relevant context.
2. If retrieved_docs exist but approved is False → route to "researcher" 
   (the human has not approved yet; the human_review_node will intercept before writer).
3. If approved is True and draft is empty → route to "writer" to produce the answer.
4. If draft is not empty → route to "end" because the pipeline is complete.

Always return a concise reasoning string explaining your decision."""


def supervisor_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: decide which agent to invoke next.

    Returns a partial state update containing only `next_agent`.
    """
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    # Build a compact state summary for the LLM
    state_summary = (
        f"Query: {state.get('query', '')}\n"
        f"retrieved_docs count: {len(state.get('retrieved_docs') or [])}\n"
        f"approved: {state.get('approved', False)}\n"
        f"draft present: {bool(state.get('draft', ''))}\n"
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        {"role": "user", "content": f"Current pipeline state:\n{state_summary}\n\nWhere should we route next?"},
    ]

    decision: RouteDecision = structured_llm.invoke(messages)
    print(f"\n[supervisor] → '{decision.next}' | reason: {decision.reasoning}")

    return {"next_agent": decision.next}
