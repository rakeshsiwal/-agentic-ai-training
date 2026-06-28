"""
agent/mcp_client.py — MultiServerMCPClient configuration
"""

from __future__ import annotations

import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient

_PROJECT_ROOT = str(Path(__file__).parent.parent)

SERVER_NAMES: list[str] = ["task04_tools", "time"]


def get_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient(
        {
            "task04_tools": {
                "command": sys.executable,
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "env": {"PYTHONPATH": _PROJECT_ROOT},
            },
            "time": {
                "command": sys.executable,
                "args": ["-m", "mcp_server_time", "--local-timezone=UTC"],
                "transport": "stdio",
            },
        }
    )