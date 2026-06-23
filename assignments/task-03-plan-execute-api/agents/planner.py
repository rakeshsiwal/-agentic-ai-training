"""
Planner agent — decomposes the user query into 2-4 focused sub-tasks.
On re-plan iterations the evaluator's feedback is incorporated so the
planner asks more targeted questions.
"""
from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from pydantic import BaseModel, Field

from state import PlanExecuteState, SubTask

load_dotenv()


# ── Pydantic structured-output models ──────────────────────────────────────

class Plan(BaseModel):
    sub_tasks: List[SubTask] = Field(
        ..., min_length=2, max_length=4,
        description="List of 2-4 focused sub-questions that together answer the original query."
    )
    reasoning: str = Field(..., description="Brief explanation of the decomposition strategy.")


# ── LLM ────────────────────────────────────────────────────────────────────

# def _get_llm():
#     return ChatAnthropic(
#         model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"),
#         temperature=0,
#     ).with_structured_output(Plan)
from langchain_openai import ChatOpenAI
import os

def _get_llm():
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
    ).with_structured_output(Plan)

# ── Node ───────────────────────────────────────────────────────────────────

def planner_node(state: PlanExecuteState) -> dict:
    llm = _get_llm()
    iteration = state.get("iteration", 0)
    query = state["original_query"]
    feedback = (state.get("evaluation") or {}).get("feedback", "")

    if iteration == 0:
        prompt = (
            f"You are an expert research planner.\n\n"
            f"Decompose the following question into 2-4 focused sub-questions "
            f"that can each be answered independently by a retrieval system.\n\n"
            f"Original question: {query}\n\n"
            f"Rules:\n"
            f"- Each sub-question must be self-contained.\n"
            f"- Cover different aspects of the original question.\n"
            f"- Use clear, specific language."
        )
    else:
        prompt = (
            f"You are an expert research planner performing a re-plan.\n\n"
            f"The previous answer was rejected by the evaluator with this feedback:\n"
            f"  \"{feedback}\"\n\n"
            f"Original question: {query}\n\n"
            f"Decompose the question into 2-4 NEW focused sub-questions that "
            f"address the evaluator's feedback and will produce a better answer.\n"
            f"Make sub-questions more specific and targeted than before."
        )

    plan: Plan = llm.invoke(prompt)

    # Assign deterministic IDs
    sub_tasks: list[SubTask] = [
        {"id": f"sub_{i}", "question": t.question if hasattr(t, "question") else t["question"]}
        for i, t in enumerate(plan.sub_tasks)
    ]

    print(f"[Planner] iteration={iteration} decomposed into {len(sub_tasks)} sub-tasks")
    for st in sub_tasks:
        print(f"  {st['id']}: {st['question']}")

    return {
        "sub_tasks": sub_tasks,
        # Reset downstream state for a clean re-run
        "retrieved_docs": [],
        "aggregated_context": [],
        "draft": "",
    }