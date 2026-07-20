"""
agent/llm.py — Shared multi-provider LLM factory
=================================================
This is the ONLY place in the codebase where an LLM provider is selected.
All other modules import get_llm() from here instead of defining their own.

Provider priority (first available API key wins):
  1. Anthropic  — ANTHROPIC_API_KEY
  2. OpenAI     — OPENAI_API_KEY
  3. Google     — GOOGLE_API_KEY
  4. Groq       — GROQ_API_KEY
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature: float = 0.0):
    """
    Return a ChatModel instance based on the first available API key.

    Parameters
    ----------
    temperature:
        Sampling temperature forwarded to the underlying model.

    Returns
    -------
    A LangChain-compatible BaseChatModel.

    Raises
    ------
    EnvironmentError
        When no recognised API key is present in the environment.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model="claude-sonnet-4-6",
            temperature=temperature,
            max_tokens=4096,
        )

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=temperature,
        )

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=temperature,
        )

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=temperature,
        )

    raise EnvironmentError(
        "No LLM API key found. Set at least one of: "
        "ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY, GROQ_API_KEY"
    )
