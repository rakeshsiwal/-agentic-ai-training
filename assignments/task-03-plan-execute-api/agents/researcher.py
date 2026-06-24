"""
Researcher agent — one instance is spawned per sub-task via Send().
Retrieves relevant chunks from the vector store and appends them to
the shared `retrieved_docs` list (operator.add reducer handles merging).
"""
from __future__ import annotations

from rag.retriever import retrieve
from state import PlanExecuteState, SubTask


def researcher_node(state: PlanExecuteState) -> dict:
    task: SubTask = state["current_sub_task"]
    question = task["question"]
    sub_id = task["id"]

    chunks = retrieve(question, k=4)

    # Tag each chunk with its sub-task ID so the aggregator can trace origin
    tagged_chunks = [f"[{sub_id}] {chunk}" for chunk in chunks]

    print(f"[Researcher:{sub_id}] found {len(tagged_chunks)} chunk(s) for: {question!r}")

    return {"retrieved_docs": tagged_chunks}