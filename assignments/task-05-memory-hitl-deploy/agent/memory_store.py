"""
agent/memory_store.py — Long-term memory helpers
=================================================
Three memory types, all stored in a LangGraph BaseStore:

  preferences  — user style/format preferences ("I prefer concise answers")
  episodes     — per-session summaries written at conversation end
  facts        — explicit facts the user states about themselves

Namespaces:
  ("preferences", user_id)  → key: preference_name
  ("episodes",    user_id)  → key: "session_YYYY-MM-DD_HH-MM-SS"
  ("facts",       user_id)  → key: fact_slug
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage


# ---------------------------------------------------------------------------
# Low-level store helpers
# ---------------------------------------------------------------------------

async def save_memory(store, namespace: tuple, key: str, value: dict) -> None:
    """Write a memory record to the store."""
    await store.aput(namespace, key, value)


async def load_memories(store, namespace: tuple) -> list[dict]:
    """Return all memory records in a namespace."""
    items = await store.asearch(namespace)
    return [item.value for item in items]


async def load_relevant_memories(
    store, user_id: str, query: str, k: int = 3
) -> list[dict]:
    """
    Return the k most recent episode summaries for the given user.
    (Recency-based; semantic search is the ⭐⭐ bonus upgrade.)
    """
    items = await store.asearch(("episodes", user_id))
    # Sort by timestamp descending, take k most recent
    sorted_items = sorted(
        items,
        key=lambda x: x.value.get("timestamp", ""),
        reverse=True,
    )
    return [item.value for item in sorted_items[:k]]


# ---------------------------------------------------------------------------
# Session summary
# ---------------------------------------------------------------------------

async def summarize_session(messages: list[BaseMessage], llm) -> str:
    """
    Summarize a conversation into a one-paragraph episode summary suitable
    for storage as an episodic memory.
    """
    # Build a compact transcript (skip system messages)
    transcript_lines = []
    for msg in messages:
        role = type(msg).__name__.replace("Message", "")
        if role == "System":
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        transcript_lines.append(f"{role}: {content[:300]}")

    transcript = "\n".join(transcript_lines[-20:])  # last 20 turns max

    prompt = f"""Summarize the following conversation in one concise paragraph.
Focus on: what topics were discussed, what the user was trying to learn or accomplish,
and any preferences or facts the user revealed about themselves.

Conversation:
{transcript}

Summary:"""

    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return response.content.strip()


# ---------------------------------------------------------------------------
# Preference extraction
# ---------------------------------------------------------------------------

_PREFERENCE_TRIGGER_WORDS = [
    "prefer", "always", "never", "i like", "i want", "i need",
    "please use", "use bullet", "be concise", "cite sources",
]


def detect_preference(text: str) -> str | None:
    """
    Return a short preference key if the text looks like a user preference
    statement, else None.
    Simple heuristic — good enough for this task.
    """
    lower = text.lower()
    for trigger in _PREFERENCE_TRIGGER_WORDS:
        if trigger in lower:
            return text[:200]
    return None


# ---------------------------------------------------------------------------
# Memory context builder (for injecting into system prompt)
# ---------------------------------------------------------------------------

async def build_memory_context(store, user_id: str, query: str) -> str:
    """
    Build a memory context string to inject into the system prompt.
    Includes relevant past episodes and all stored preferences.
    """
    parts = []

    # Load preferences
    prefs = await load_memories(store, ("preferences", user_id))
    if prefs:
        pref_lines = [f"  - {p.get('value', '')}" for p in prefs]
        parts.append("User preferences from past sessions:\n" + "\n".join(pref_lines))

    # Load relevant episodes
    episodes = await load_relevant_memories(store, user_id, query, k=3)
    if episodes:
        ep_lines = [
            f"  [{e.get('timestamp', '')[:10]}] {e.get('summary', '')}"
            for e in episodes
        ]
        parts.append("Relevant past conversations:\n" + "\n".join(ep_lines))

    if not parts:
        return ""

    return (
        "\n\n--- MEMORY FROM PAST SESSIONS ---\n"
        + "\n\n".join(parts)
        + "\n--- END MEMORY ---\n"
    )
