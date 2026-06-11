from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # full conversation history
    topic: str                               # the research topic
    search_results: list[str]               # raw results from tools
    final_summary: str                       # the final answer
