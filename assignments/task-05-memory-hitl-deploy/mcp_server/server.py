"""
MCP Server — task04-tools
Run standalone: python -m mcp_server.server
"""

import sys
import os

# Pre-load the vector store before the MCP event loop starts.
# This means the embedding model is ready before the first tool call arrives.
print("[server] pre-loading vector store...", file=sys.stderr)
try:
    from mcp_server.tools import _get_vector_store
    _get_vector_store()
    print("[server] vector store ready.", file=sys.stderr)
except Exception as e:
    print(f"[server] warning: could not pre-load vector store: {e}", file=sys.stderr)

from mcp.server.fastmcp import FastMCP
from mcp_server.tools import calculate as _calculate
from mcp_server.tools import get_current_date as _get_current_date
from mcp_server.tools import retrieve as _retrieve

mcp = FastMCP("task04-tools")


@mcp.tool()
def retrieve_context(query: str, k: int = 4) -> list[str]:
    """Retrieve the top-k relevant chunks from the local document store."""
    return _retrieve(query, k)


@mcp.tool()
def calculate(expression: str) -> str:
    """Safely evaluate a math expression. Supports +,-,*,/,**,%, sqrt, log, sin, cos, tan, pi, e."""
    return _calculate(expression)


@mcp.tool()
def get_current_date() -> str:
    """Return today's date in ISO 8601 format (YYYY-MM-DD)."""
    return _get_current_date()


if __name__ == "__main__":
    mcp.run(transport="stdio")