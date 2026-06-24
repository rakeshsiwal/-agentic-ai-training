"""
Writer agent — synthesises the aggregated context into a cited answer.
On re-plan iterations the evaluator's feedback is included in the prompt
so the writer knows what to improve.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from state import PlanExecuteState

load_dotenv()


# def _get_llm():
#     return ChatAnthropic(
#         model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
#         temperature=0.3,
#         max_tokens=2048,
#     )
from langchain_openai import ChatOpenAI
import os

def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.3,
        max_tokens=2048,
    )

def writer_node(state: PlanExecuteState) -> dict:
    llm = _get_llm()
    query = state["original_query"]
    context_chunks = state.get("aggregated_context", [])
    feedback = (state.get("evaluation") or {}).get("feedback", "")
    iteration = state.get("iteration", 0)

    # Build numbered context
    context_text = "\n\n".join(
        f"[Chunk {i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    feedback_section = ""
    if iteration > 0 and feedback:
        feedback_section = (
            f"\n\nIMPORTANT — The previous draft was rejected with this feedback:\n"
            f"  \"{feedback}\"\n"
            f"Address this feedback explicitly in your new answer."
        )

    prompt = (
        f"You are a knowledgeable research assistant.\n\n"
        f"Answer the following question using ONLY the information in the provided "
        f"context chunks. Cite chunks inline using [Chunk N] notation.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context_text}"
        f"{feedback_section}\n\n"
        f"Rules:\n"
        f"- Every factual claim must reference at least one [Chunk N].\n"
        f"- Do NOT invent information not present in the chunks.\n"
        f"- Be comprehensive but concise.\n"
        f"- If the context is insufficient, say so explicitly."
    )

    response = llm.invoke([HumanMessage(content=prompt)])
    draft = response.content

    print(f"[Writer] iteration={iteration} draft length={len(draft)} chars")

    return {"draft": draft}