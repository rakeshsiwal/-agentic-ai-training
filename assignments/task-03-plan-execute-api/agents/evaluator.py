"""
Evaluator agent — scores the writer's draft for faithfulness and relevance.
Returns structured EvaluationResult and increments the iteration counter.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from state import EvaluationResult, PlanExecuteState

load_dotenv()


# ── Pydantic model ──────────────────────────────────────────────────────────

class EvaluationOutput(BaseModel):
    faithfulness: int = Field(
        ..., ge=1, le=5,
        description=(
            "1 = Answer invents facts not in the retrieved chunks. "
            "5 = Every claim is directly supported by a cited chunk."
        ),
    )
    relevance: int = Field(
        ..., ge=1, le=5,
        description=(
            "1 = Answer is off-topic or misses the user's question. "
            "5 = Answer fully and directly addresses the original query."
        ),
    )
    feedback: str = Field(
        ...,
        description="One sentence of specific, actionable critique for the next iteration.",
    )


# ── LLM ────────────────────────────────────────────────────────────────────

# def _get_llm():
#     return ChatAnthropic(
#         model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
#         temperature=0,
#     ).with_structured_output(EvaluationOutput)
# from langchain_openai import ChatOpenAI
import os

def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).with_structured_output(EvaluationOutput)

# ── Node ───────────────────────────────────────────────────────────────────

def evaluator_node(state: PlanExecuteState) -> dict:
    llm = _get_llm()
    query = state["original_query"]
    draft = state.get("draft", "")
    context_chunks = state.get("aggregated_context", [])
    iteration = state.get("iteration", 0)

    context_text = "\n\n".join(
        f"[Chunk {i + 1}] {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = (
        f"You are a strict quality evaluator for research answers.\n\n"
        f"Original question: {query}\n\n"
        f"Retrieved context:\n{context_text}\n\n"
        f"Draft answer:\n{draft}\n\n"
        f"Score the draft on two dimensions (1-5 each):\n\n"
        f"FAITHFULNESS:\n"
        f"  1 — The draft invents multiple facts not found in the context.\n"
        f"  2 — The draft has significant unsupported claims.\n"
        f"  3 — Most claims are supported but a few are not.\n"
        f"  4 — Nearly all claims are cited and correct.\n"
        f"  5 — Every factual claim is directly supported by a cited chunk.\n\n"
        f"RELEVANCE:\n"
        f"  1 — The draft is completely off-topic or ignores the question.\n"
        f"  2 — The draft partially addresses the question.\n"
        f"  3 — The draft addresses the question but misses key aspects.\n"
        f"  4 — The draft mostly answers the question with minor gaps.\n"
        f"  5 — The draft fully and directly answers the original question.\n\n"
        f"Also provide one sentence of specific, actionable feedback for improvement."
    )

    result: EvaluationOutput = llm.invoke(prompt)

    evaluation: EvaluationResult = {
        "faithfulness": result.faithfulness,
        "relevance": result.relevance,
        "feedback": result.feedback,
    }

    new_iteration = iteration + 1

    print(
        f"[Evaluator] iteration={new_iteration} "
        f"faithfulness={result.faithfulness}/5 relevance={result.relevance}/5"
    )
    print(f"  Feedback: {result.feedback}")

    # If quality is good or we've hit the limit, finalise the answer
    is_acceptable = result.faithfulness >= 3 and result.relevance >= 3
    final_answer = state.get("draft", "") if is_acceptable else ""

    return {
        "evaluation": evaluation,
        "iteration": new_iteration,
        "final_answer": final_answer,
    }