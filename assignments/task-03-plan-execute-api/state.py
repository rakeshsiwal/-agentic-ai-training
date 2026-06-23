import operator
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class SubTask(TypedDict):
    id: str        # e.g. "sub_0", "sub_1"
    question: str  # the focused sub-question


class EvaluationResult(TypedDict):
    faithfulness: int  # 1–5
    relevance: int     # 1–5
    feedback: str      # one sentence of actionable critique


class PlanExecuteState(TypedDict):
    messages:           Annotated[list, add_messages]
    original_query:     str
    sub_tasks:          list[SubTask]
    retrieved_docs:     Annotated[list, operator.add]   # fan-in accumulator
    aggregated_context: list[str]
    draft:              str
    final_answer:       str
    evaluation:         EvaluationResult
    iteration:          int
    max_iterations:     int
    current_sub_task:   SubTask   # injected per Send() branch