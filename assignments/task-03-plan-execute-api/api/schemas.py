from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):
    query: str = Field(..., description="The research question to answer.")
    thread_id: str = Field(
        ...,
        description="Unique identifier for this pipeline run (used for persistence).",
    )
    max_iterations: int = Field(
        default=3, ge=1, le=5,
        description="Max reflection loops before forcing a final answer.",
    )


class PipelineStatusResponse(BaseModel):
    status: str           # "running" | "complete" | "error"
    thread_id: str
    state: dict | None = None
    error: str | None = None