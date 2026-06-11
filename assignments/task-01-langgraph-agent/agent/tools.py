"""
tools.py — Tool definitions bound to the LLM via tool calling.

Three tools are provided:
  - search_web(query)      : DuckDuckGo search (falls back to simulated results)
  - calculate(expression)  : Safe math evaluation
  - get_current_date()     : Returns today's date
"""

import math
import operator
import ast
from datetime import date

from langchain_core.tools import tool


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------

@tool
def search_web(query: str) -> str:
    """Search the web for information about a query.

    Args:
        query: The search query string.

    Returns:
        A string containing up to 5 search result snippets.
    """
    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                title = r.get("title", "No title")
                body = r.get("body", "No snippet available")
                href = r.get("href", "")
                results.append(f"• {title}\n  {body}\n  URL: {href}")

        if results:
            return "\n\n".join(results)
        return f"No results found for query: {query}"

    except ImportError:
        # Graceful fallback if duckduckgo_search is not installed
        return _simulated_search(query)
    except Exception as exc:
        return f"Search failed ({exc}). Falling back to simulated results.\n\n{_simulated_search(query)}"


def _simulated_search(query: str) -> str:
    """Return plausible-looking fake results so the agent still runs offline."""
    return (
        f"[Simulated search results for: '{query}']\n\n"
        "• Result 1: Overview of the topic — This article provides a comprehensive "
        "introduction covering the main concepts, history, and current developments.\n\n"
        "• Result 2: Recent advances — Researchers have published new findings that "
        "expand our understanding of this area in 2025–2026.\n\n"
        "• Result 3: Industry perspective — Leading organizations are investing "
        "heavily in this space, with several major announcements expected in Q3 2026.\n\n"
        "Note: Install `duckduckgo-search` for live results."
    )


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------

# Safe set of names available inside eval
_SAFE_NAMES = {
    k: v for k, v in math.__dict__.items() if not k.startswith("_")
}
_SAFE_NAMES.update({"abs": abs, "round": round, "pow": pow})


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports standard arithmetic, math functions (sin, cos, log, sqrt, etc.),
    and constants (pi, e, inf).

    Args:
        expression: A Python-style math expression, e.g. "sqrt(2) * pi".

    Returns:
        The numeric result as a string, or an error message.
    """
    try:
        # Parse to AST and reject anything non-numeric for safety
        tree = ast.parse(expression, mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Only allow calls to whitelisted names
                if isinstance(node.func, ast.Name) and node.func.id not in _SAFE_NAMES:
                    raise ValueError(f"Function '{node.func.id}' is not allowed.")
            elif isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)):
                raise ValueError("Imports and function definitions are not allowed.")

        result = eval(compile(tree, "<string>", "eval"), {"__builtins__": {}}, _SAFE_NAMES)
        return str(result)
    except Exception as exc:
        return f"Could not evaluate expression: {exc}"


# ---------------------------------------------------------------------------
# Current Date
# ---------------------------------------------------------------------------

@tool
def get_current_date() -> str:
    """Return today's date in ISO 8601 format (YYYY-MM-DD).

    Useful for time-aware queries that need to know what 'current' means.
    """
    return date.today().isoformat()


# ---------------------------------------------------------------------------
# Exported list (used by agent_node and graph)
# ---------------------------------------------------------------------------

TOOLS = [search_web, calculate, get_current_date]
