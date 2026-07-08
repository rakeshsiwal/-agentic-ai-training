"""
api/schemas.py — Pydantic models for API requests, responses, and HITL payloads
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="The user's question or message")
    user_id: str = Field(default="default", description="User identifier for memory scoping")
    thread_id: str | None = Field(default=None, description="Thread ID for conversation continuity")


class PendingApproval(BaseModel):
    tool_name: str
    tool_args: dict[str, Any]


class ChatResponse(BaseModel):
    status: str  # "complete" | "interrupted"
    answer: str | None = None
    thread_id: str | None = None
    pending_approval: PendingApproval | None = None


class ResumeRequest(BaseModel):
    thread_id: str = Field(..., description="Thread ID of the interrupted conversation")
    approved: bool = Field(..., description="Whether to approve the pending tool call")


class ResumeResponse(BaseModel):
    status: str  # "complete" | "interrupted"
    answer: str | None = None
    thread_id: str
    pending_approval: PendingApproval | None = None


class MemoryRecord(BaseModel):
    namespace: str
    key: str
    value: dict[str, Any]


class MemoriesResponse(BaseModel):
    user_id: str
    preferences: list[dict]
    episodes: list[dict]


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "task05"
