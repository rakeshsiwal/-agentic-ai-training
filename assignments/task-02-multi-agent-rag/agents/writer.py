"""
agents/writer.py — Writer node: drafts a structured, cited answer using approved context.

Only runs after `approved == True` is confirmed by the human-in-the-loop node.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage

from state import PipelineState

load_dotenv()

WRITER_SYSTEM_PROMPT = """You are an expert technical writer. You will receive:
1. A user question.
2. A numbered list of retrieved context chunks.

Your task:
- Write a clear, well-structured answer using ONLY the provided context.
- Cite evidence using inline references like [Chunk 1], [Chunk 2], etc.
- If the context does not fully answer the question, say so explicitly.
- Keep the answer concise but complete (2–5 paragraphs is usually right).
- Do NOT invent facts beyond what the chunks contain.

Format:
## Answer

<your answer here, with inline [Chunk N] citations>

## Sources Used
- Chunk N: <first 80 chars of that chunk>
"""


def _get_llm():
    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0.3)
    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
    raise EnvironmentError("No LLM API key found. Check your .env file.")


def writer_node(state: PipelineState) -> PipelineState:
    """
    LangGraph node: produce a structured, cited answer from approved context.

    Requires:
        state["approved"] == True
        state["retrieved_docs"] non-empty

    Returns a partial state update with `draft` and `final_answer` set.
    """
    if not state.get("approved"):
        raise RuntimeError("Writer node called before human approval — this is a bug.")

    query = state.get("query", "")
    docs = state.get("retrieved_docs", [])

    if not docs:
        raise RuntimeError("Writer node called with no retrieved documents.")

    # Build the numbered context block
    context_block = "\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(docs)
    )

    user_message = (
        f"**Question:** {query}\n\n"
        f"**Retrieved Context:**\n\n{context_block}"
    )

    print(f"\n[writer] Drafting answer for: '{query}'")
    llm = _get_llm()
    response = llm.invoke([
        SystemMessage(content=WRITER_SYSTEM_PROMPT),
        {"role": "user", "content": user_message},
    ])

    answer_text = response.content
    print(f"[writer] Draft complete ({len(answer_text)} chars).")

    return {
        "draft": answer_text,
        "final_answer": answer_text,
        "messages": [AIMessage(content=answer_text)],
    }
