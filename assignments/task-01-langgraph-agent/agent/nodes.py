"""
nodes.py — Individual node functions wired into the LangGraph graph.

Node flow:
  agent_node  →  tool_node (conditional)  →  summarizer_node  →  END / agent_node
"""

import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from .state import AgentState
from .tools import TOOLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_llm():
    """Instantiate the LLM based on available env vars (lazy import)."""
    import os

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model="gpt-4o-mini", temperature=0)

    if os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

    if os.getenv("GOOGLE_API_KEY"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq
        return ChatGroq(model="llama-3.1-8b-instant", temperature=0)

    raise EnvironmentError(
        "No LLM API key found. Set one of: OPENAI_API_KEY, ANTHROPIC_API_KEY, "
        "GOOGLE_API_KEY, or GROQ_API_KEY in your .env file."
    )


SYSTEM_PROMPT = """You are a helpful research assistant. Your job is to:
1. Understand what the user wants to research.
2. Use the available tools (search_web, calculate, get_current_date) to gather information.
3. Provide clear, well-structured summaries based on your findings.

Guidelines:
- Always search before answering factual or current-events questions.
- If the user says they're done or satisfied, respond with a friendly goodbye and do NOT call any tools.
- If a query is unanswerable or outside scope, say so clearly and politely.
- Maintain context across follow-up questions — refer back to earlier findings when relevant.
"""


# ---------------------------------------------------------------------------
# Node: agent_node
# ---------------------------------------------------------------------------

def agent_node(state: AgentState) -> AgentState:
    """Call the LLM with the current conversation. May request tool calls."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = state["messages"]

    # Prepend system prompt if this is the first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)

    response: AIMessage = llm_with_tools.invoke(messages)

    # Track the topic from the first human message
    topic = state.get("topic", "")
    if not topic:
        first_human = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        topic = first_human[:200]  # cap length

    return {
        "messages": [response],
        "topic": topic,
        "search_results": state.get("search_results", []),
        "final_summary": state.get("final_summary", ""),
    }


# ---------------------------------------------------------------------------
# Node: tool_node
# ---------------------------------------------------------------------------

# Build a lookup dict so we can dispatch by tool name
_TOOL_MAP = {t.name: t for t in TOOLS}


def tool_node(state: AgentState) -> AgentState:
    """Execute all tool calls requested by the last AI message."""
    last_message: AIMessage = state["messages"][-1]
    tool_results: list[ToolMessage] = []
    new_search_results: list[str] = list(state.get("search_results", []))

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        tool_fn = _TOOL_MAP.get(tool_name)
        if tool_fn is None:
            output = f"Unknown tool: {tool_name}"
        else:
            try:
                output = tool_fn.invoke(tool_args)
            except Exception as exc:
                output = f"Tool '{tool_name}' raised an error: {exc}"

        # Accumulate web search results for summarizer context
        if tool_name == "search_web":
            new_search_results.append(output)

        tool_results.append(
            ToolMessage(content=str(output), tool_call_id=tool_id)
        )

    return {
        "messages": tool_results,
        "topic": state.get("topic", ""),
        "search_results": new_search_results,
        "final_summary": state.get("final_summary", ""),
    }


# ---------------------------------------------------------------------------
# Node: summarizer_node
# ---------------------------------------------------------------------------

SUMMARIZER_PROMPT = """You are a research summarizer. Given the conversation history and 
any raw search results, produce a clean, structured summary for the user.

Format your response with:
- A brief intro sentence
- Bullet points or numbered list for key findings
- A closing sentence with any caveats or suggestions for follow-up

Keep it concise but informative. Do NOT call any tools — only summarize.
"""


def summarizer_node(state: AgentState) -> AgentState:
    """Condense tool results and conversation into a readable summary."""
    llm = _get_llm()  # no tools bound here

    # Build a focused prompt
    search_context = ""
    if state.get("search_results"):
        search_context = "\n\nRaw search results collected so far:\n" + "\n---\n".join(
            state["search_results"][-3:]  # last 3 to stay within context
        )

    summarizer_messages = [
        SystemMessage(content=SUMMARIZER_PROMPT + search_context),
        *state["messages"],
    ]

    summary: AIMessage = llm.invoke(summarizer_messages)

    return {
        "messages": [summary],
        "topic": state.get("topic", ""),
        "search_results": state.get("search_results", []),
        "final_summary": summary.content,
    }


# ---------------------------------------------------------------------------
# Conditional edge: should_continue
# ---------------------------------------------------------------------------

def should_continue(state: AgentState) -> Literal["tool_node", "summarizer_node", "end"]:
    """Decide the next step after agent_node.

    - If the last AI message has tool_calls → execute tools
    - If the message content signals the user is done → end
    - Otherwise → summarize
    """
    last_message: AIMessage = state["messages"][-1]

    # Route to tools if any tool calls are present
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"

    # Check if the agent is saying goodbye (no more work to do)
    farewell_signals = ["goodbye", "happy to help", "feel free to ask", "bye", "take care"]
    # content_lower = (last_message.content or "").lower()
    content = last_message.content
    if isinstance(content, list):
        content = " ".join(block.get("text", "") if isinstance(block, dict) else str(block) for block in content)
    content_lower = (content or "").lower()
    if any(signal in content_lower for signal in farewell_signals):
        return "end"

    # return "summarizer_node"
    # Only summarize if there are actual search results to condense
    if state.get("search_results"):
        return "summarizer_node"
    return "end"
