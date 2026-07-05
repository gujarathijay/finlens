"""
FinLens extraction schemas.

These models define the exact JSON structure the fine-tuned model
learns to produce. Used in: dataset generation, training, evaluation,
guardrails, and API responses. One definition, used everywhere.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

# ── Enums for constrained values ──


class Severity(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Impact(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RiskCategory(StrEnum):
    REGULATORY = "regulatory"
    MARKET = "market"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    LEGAL = "legal"
    CYBERSECURITY = "cybersecurity"
    ENVIRONMENTAL = "environmental"
    COMPETITIVE = "competitive"


# ── Extracted components ──


class RiskFactor(BaseModel):
    """A single risk factor extracted from the filing."""

    factor: str = Field(description="Description of the risk")
    category: RiskCategory = Field(description="Risk classification")
    severity: Severity = Field(description="How severe this risk is")
    evidence: str = Field(description="Quote or reference from the filing supporting this risk")


class MaterialEvent(BaseModel):
    """A significant event mentioned in the filing."""

    event: str = Field(description="What happened")
    date: str | None = Field(default=None, description="When it happened (YYYY-MM-DD if available)")
    impact: Impact = Field(description="Positive, negative, or neutral impact")
    details: str = Field(description="Additional context about the event")


class FinancialObligation(BaseModel):
    """A financial commitment or liability."""

    obligation: str = Field(description="What is owed")
    amount: str | None = Field(default=None, description="Dollar amount if stated (e.g. '$2.5B')")
    deadline: str | None = Field(default=None, description="Due date if stated (YYYY-MM-DD)")
    category: str = Field(description="Type: debt, lease, legal, pension, etc.")


# ── Top-level extraction result ──


class ExtractionResult(BaseModel):
    """
    Complete extraction output from a single SEC filing section.
    This is what the fine-tuned model learns to produce.
    """

    company_name: str = Field(description="Company name from the filing")
    filing_type: str = Field(default="10-K", description="Filing type (10-K, 10-Q, 8-K)")
    fiscal_year: str = Field(description="Fiscal year covered (e.g. '2024')")
    risk_factors: list[RiskFactor] = Field(default_factory=list)
    material_events: list[MaterialEvent] = Field(default_factory=list)
    financial_obligations: list[FinancialObligation] = Field(default_factory=list)
    summary: str = Field(description="2-3 sentence summary of key findings")


# ── Helper: generate JSON schema for prompts ──


def get_extraction_schema() -> dict:
    """Returns the JSON schema dict. Used in prompt templates and validation."""
    return ExtractionResult.model_json_schema()
