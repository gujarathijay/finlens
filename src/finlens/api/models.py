"""
API request and response models.
"""

from pydantic import BaseModel, Field

from src.finlens.schemas import ExtractionResult


class ExtractionRequest(BaseModel):
    """What the user sends."""
    filing_text: str = Field(
        description="Raw SEC filing text to extract from",
        min_length=50,
        max_length=10000,
    )


class ExtractionResponse(BaseModel):
    """What we return."""
    status: str = Field(description="success, failed, or flagged")
    extraction: ExtractionResult | None = Field(default=None)
    guardrails_passed: bool = Field(default=True)
    guardrail_failures: list[str] = Field(default_factory=list)
    latency_ms: float = Field(default=0.0)
    request_id: int = Field(description="Database ID for this request")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    model_loaded: bool = False
    database: str = "connected"